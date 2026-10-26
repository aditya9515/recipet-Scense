"""Bounded crop recovery with immutable base OCR and explicit candidate provenance."""

from __future__ import annotations

from pathlib import Path
from statistics import mean, median
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from .address import address_score, reconstruct_address
from .common import digest, read_json, write_json
from .contracts import ImageRecord, OCRRecord, RecoveryAttempt, RecoveryReport
from .imaging import transform_points
from .ocr_adapter import get_engine
from .reading_order import order_units
from .rules import extract_rules
from .validation import clean_text, parse_amount, parse_date, validate_field

PRIORITY = ("address", "total", "date", "company")


def comparable(field, value):
    if field == "total":
        return parse_amount(value)[0] or clean_text(value or "").casefold()
    if field == "date":
        return parse_date(value)[0] or clean_text(value or "").casefold()
    return clean_text(value or "").casefold()


def recovery_targets(prediction, ocr, config, comparison=None):
    by_id = {u.unit_id: u for u in ocr.units}
    result = []
    for field in PRIORITY:
        item = prediction.fields[field]
        triggers, rank = [], 9
        if item.status in {"missing", "invalid"}:
            triggers.append(item.status)
            rank = 0
        if item.status == "ambiguous" or any("conflict" in r or "multiple" in r for r in item.review_reasons):
            triggers.append("conflicting_candidates")
            rank = min(rank, 1)
        if (
            comparison
            and comparison.fields[field].raw_value
            and item.raw_value
            and comparable(field, item.raw_value) != comparable(field, comparison.fields[field].raw_value)
        ):
            triggers.append("rules_model_disagreement")
            rank = min(rank, 1)
        scores = [
            by_id[i].recognition_score
            for i in item.evidence_unit_ids
            if i in by_id and by_id[i].recognition_score is not None
        ]
        if scores and mean(scores) < config.low_ocr_score:
            triggers.append("weak_ocr_evidence")
            rank = min(rank, 2)
        if (
            item.score_kind == "uncalibrated_first_subword_mean"
            and item.score is not None
            and item.score < config.low_model_score
        ):
            triggers.append("weak_model_score")
            rank = min(rank, 2)
        if item.review_reasons and not triggers:
            triggers.append("existing_review_flag")
            rank = 2
        if triggers:
            result.append((rank, PRIORITY.index(field), field, triggers))
    return sorted(result)


def crop_box(image, ocr, item, field, config):
    width, height = image.processed_size
    selected = [u for u in ocr.units if u.unit_id in item.evidence_unit_ids]
    if selected:
        line_height = median(u.bbox_xyxy[3] - u.bbox_xyxy[1] for u in selected)
        pad = config.crop_padding_heights * line_height
        left = min(u.bbox_xyxy[0] for u in selected) - pad
        top = min(u.bbox_xyxy[1] for u in selected) - pad
        right = max(u.bbox_xyxy[2] for u in selected) + pad
        bottom = max(u.bbox_xyxy[3] for u in selected) + pad
        # Full line width retains TOTAL labels and neighboring address lines.
        left, right = 0, width
        if field == "address":
            bottom += 4 * line_height
    elif field in {"company", "address"}:
        left, top, right, bottom = 0, 0, width, height * config.header_fraction
    elif field == "total":
        left, top, right, bottom = 0, height * config.payment_start, width, height
    else:
        left, top, right, bottom = 0, 0, width, height * 0.65
    left, top = max(0, int(left)), max(0, int(top))
    right, bottom = min(width, int(np.ceil(right))), min(height, int(np.ceil(bottom)))
    if right - left < 24 or bottom - top < 24:
        raise ValueError("Recovery crop is too small")
    return left, top, right, bottom


def prepare_crop(image: ImageRecord, box, preparation, config, artifact_root):
    with Image.open(image.processed_path) as opened:
        crop = opened.convert("RGB").crop(box)
    scale = min(config.upscale, config.max_crop_side / max(crop.size))
    size = (max(24, round(crop.width * scale)), max(24, round(crop.height * scale)))
    crop = crop.resize(size, Image.Resampling.LANCZOS)
    if preparation == "contrast":
        lab = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        crop = Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    sx, sy = size[0] / (box[2] - box[0]), size[1] / (box[3] - box[1])
    matrix = np.array([[sx, 0, -sx * box[0]], [0, sy, -sy * box[1]], [0, 0, 1]], dtype=float)
    fingerprint = digest(
        {
            "source": image.processed_sha256,
            "box": box,
            "preparation": preparation,
            "matrix": matrix.tolist(),
            "configuration": config.fingerprint,
        }
    )
    path = Path(artifact_root) / "recovery_crops" / image.receipt_id / f"{fingerprint}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        crop.save(path, compress_level=1)
    child = image.model_copy(
        update={
            "processed_path": str(path.resolve()),
            "processed_size": size,
            "processed_sha256": digest(path.read_bytes()),
            "preprocessing_fingerprint": fingerprint,
            "display_to_processed": (matrix @ np.asarray(image.display_to_processed)).tolist(),
            "operations": [*image.operations, f"crop:{box}", preparation],
        }
    )
    return child, matrix


def map_retry_units(raw: OCRRecord, parent: ImageRecord, matrix, attempt_id):
    inverse = np.linalg.inv(matrix)
    mapped = []
    width, height = parent.processed_size
    for unit in raw.units:
        points = transform_points(unit.polygon, inverse)
        points[:, 0] = np.clip(points[:, 0], 0, width)
        points[:, 1] = np.clip(points[:, 1], 0, height)
        box = (*points.min(axis=0).tolist(), *points.max(axis=0).tolist())
        mapped.append(
            unit.model_copy(
                update={
                    "unit_id": f"{attempt_id}_{unit.unit_id}",
                    "line_id": f"{attempt_id}_{unit.line_id}",
                    "polygon": [tuple(point) for point in points.tolist()],
                    "bbox_xyxy": box,
                    "attempt_id": attempt_id,
                }
            )
        )
    return mapped


def working_record(base, mapped, box):
    """Replace one region for candidate extraction; never concatenate duplicates."""

    def outside(unit):
        x1, y1, x2, y2 = unit.bbox_xyxy
        return not (box[0] <= (x1 + x2) / 2 <= box[2] and box[1] <= (y1 + y2) / 2 <= box[3])

    return base.model_copy(update={"units": order_units([u for u in base.units if outside(u)] + mapped)})


def accept_candidate(field, original, candidate, work, config, method):
    if not candidate.raw_value or candidate.status != "predicted":
        return False, "candidate_missing_invalid_or_ambiguous"
    if comparable(field, candidate.raw_value) == comparable(field, original.raw_value):
        return False, "same_value"
    evidence = [u for u in work.units if u.unit_id in candidate.evidence_unit_ids]
    if not evidence or not any(u.attempt_id != "base" for u in evidence):
        return False, "candidate_has_no_retry_evidence"
    scores = [u.recognition_score for u in evidence if u.recognition_score is not None]
    if scores and mean(scores) < config.low_ocr_score:
        return False, "candidate_ocr_support_is_weak"
    if (
        candidate.score_kind == "uncalibrated_first_subword_mean"
        and (candidate.score or 0) < config.low_model_score
    ):
        return False, "candidate_model_support_is_weak"
    if original.status in {"missing", "invalid", "ambiguous"}:
        return True, "valid_unambiguous_recovery"
    if field == "address" and address_score(candidate.raw_value) >= max(
        config.address_min_score, address_score(original.raw_value) + config.address_gain
    ):
        return True, "address_structural_gain"
    if method == "layoutlmv3":
        corroboration = extract_rules(work).fields[field]
        if corroboration.status == "predicted" and comparable(field, corroboration.raw_value) == comparable(
            field, candidate.raw_value
        ):
            return True, "rules_model_agreement"
    return False, "usable_original_retained_without_structural_corroboration"


def enhance_prediction(
    image, base_ocr, baseline, *, mode, config, extract, method, artifact_root, use_cache=True, engine=None
):
    started = perf_counter()
    prediction = baseline.model_copy(deep=True)
    report = RecoveryReport(
        mode=mode,
        configuration_version=config.version,
        configuration_hash=config.fingerprint,
        baseline_fields={f: v.model_copy(deep=True) for f, v in baseline.fields.items()},
    )
    catalog_units = list(base_ocr.units)
    if mode in {"address", "combined"}:
        field, decision = reconstruct_address(base_ocr, prediction.fields["address"], config)
        prediction.fields["address"] = validate_field("address", field)
        report.address_decision = decision
    comparison = extract_rules(base_ocr) if method == "layoutlmv3" else None
    completed = set()
    current_field = None
    current_box = None
    for number in range(config.max_attempts if mode in {"retry", "combined"} else 0):
        catalog = base_ocr.model_copy(update={"units": catalog_units})
        targets = [
            t for t in recovery_targets(prediction, catalog, config, comparison) if t[2] not in completed
        ]
        if not targets:
            break
        same_field = current_field is not None and any(t[2] == current_field for t in targets)
        target = next(t for t in targets if t[2] == current_field) if same_field else targets[0]
        _, _, field_name, triggers = target
        preparation = "contrast" if same_field else "rgb"
        original = prediction.fields[field_name].model_copy(deep=True)
        attempt_id = f"retry{number + 1}"
        attempt_start = perf_counter()
        # Initialize trace before any crop/OCR work so failures remain inspectable.
        attempt = RecoveryAttempt(
            attempt_id=attempt_id,
            field=field_name,
            triggers=triggers,
            crop_bbox=(0, 0, *image.processed_size),
            parent_to_crop=np.eye(3).tolist(),
            preparation=preparation,
            cache_key="",
            before=original.raw_value,
            after=original.raw_value,
            outcome="error",
            reason="not_started",
        )
        try:
            box = current_box if same_field else crop_box(image, catalog, original, field_name, config)
            child, matrix = prepare_crop(image, box, preparation, config, artifact_root)
            attempt.crop_bbox, attempt.parent_to_crop = box, matrix.tolist()
            active_engine = engine or get_engine()
            key = digest(
                {
                    "source": image.processed_sha256,
                    "crop": child.processed_sha256,
                    "preprocessing": child.preprocessing_fingerprint,
                    "engine": active_engine.fingerprint,
                    "policy": config.fingerprint,
                }
            )
            attempt.cache_key = key
            cache = Path(artifact_root) / "recovery_ocr" / f"{key}.json"
            if use_cache and cache.exists():
                raw = OCRRecord.model_validate(read_json(cache))
                attempt.cache_hit = True
            else:
                attempt.ocr_called = True
                raw = active_engine.recognize(child)
                write_json(cache, raw.model_dump(mode="json"))
            if (
                raw.receipt_id != image.receipt_id
                or raw.ocr_fingerprint != active_engine.fingerprint
                or (raw.image_width, raw.image_height) != child.processed_size
            ):
                raise ValueError("Retry OCR provenance/coordinate mismatch")
            mapped = map_retry_units(raw, image, matrix, attempt_id)
            catalog_units.extend(mapped)
            if not mapped:
                attempt.outcome, attempt.reason = "rejected", "empty_retry_ocr"
            else:
                work = working_record(base_ocr, mapped, box)
                candidate_prediction = extract(work)
                candidate = candidate_prediction.fields[field_name]
                if field_name == "address" and mode == "combined":
                    candidate, _ = reconstruct_address(work, candidate, config)
                candidate = validate_field(field_name, candidate)
                attempt.candidate = candidate.raw_value
                attempt.evidence_unit_ids = candidate.evidence_unit_ids
                accepted, reason = accept_candidate(field_name, original, candidate, work, config, method)
                attempt.reason = reason
                if accepted:
                    prediction.fields[field_name] = candidate
                    attempt.outcome, attempt.after = "accepted", candidate.raw_value
                else:
                    attempt.outcome = "unchanged" if reason == "same_value" else "rejected"
                    if candidate.raw_value and comparable(field_name, candidate.raw_value) != comparable(
                        field_name, original.raw_value
                    ):
                        prediction.fields[field_name].alternatives = list(
                            dict.fromkeys([*original.alternatives, candidate.raw_value])
                        )[:4]
                        prediction.fields[field_name].review_reasons = sorted(
                            set(original.review_reasons + ["recovery_conflict"])
                        )
            current_field, current_box = field_name, box
        except Exception as exc:
            attempt.reason = f"{type(exc).__name__}: {exc}"
            prediction.fields[field_name].review_reasons = sorted(
                set(original.review_reasons + ["recovery_failed"])
            )
            current_field, current_box = None, None
            completed.add(field_name)
        attempt.elapsed_seconds = perf_counter() - attempt_start
        report.attempts.append(attempt)
        if attempt.outcome == "accepted" or same_field:
            completed.add(field_name)
            current_field, current_box = None, None
    report.logical_attempts = len(report.attempts)
    report.uncached_ocr_calls = sum(a.ocr_called for a in report.attempts)
    report.elapsed_seconds = perf_counter() - started
    fingerprint = digest({"base": base_ocr.ocr_fingerprint, "mode": mode, "policy": config.fingerprint})
    catalog = OCRRecord(
        receipt_id=base_ocr.receipt_id,
        image_width=base_ocr.image_width,
        image_height=base_ocr.image_height,
        ocr_fingerprint=fingerprint,
        engine={
            **base_ocr.engine,
            "base_fingerprint": base_ocr.ocr_fingerprint,
            "evidence_catalog": True,
            "enhancement": mode,
            "configuration": config.fingerprint,
        },
        units=[u.model_copy(update={"reading_order": i}) for i, u in enumerate(catalog_units)],
    )
    prediction.recovery = report
    prediction.ocr_fingerprint = fingerprint
    prediction.configuration_hash = digest(
        {"baseline": baseline.configuration_hash, "mode": mode, "configuration": config.fingerprint}
    )
    prediction.check_evidence(catalog)
    return catalog, prediction
