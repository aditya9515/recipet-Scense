import json

import numpy as np
import pytest
from conftest import make_ocr

from receiptsense.address import reconstruct_address
from receiptsense.common import FIELDS
from receiptsense.contracts import FieldPrediction, OCRRecord
from receiptsense.imaging import prepare_image, transform_points
from receiptsense.pipeline import process_receipt
from receiptsense.recovery import (
    accept_candidate,
    map_retry_units,
    prepare_crop,
    recovery_targets,
    working_record,
)
from receiptsense.recovery_config import RecoveryConfig
from receiptsense.review_store import ReviewStore
from receiptsense.rules import extract_rules
from receiptsense.validation import validate_fields


def test_address_graph_joins_postal_only_line_and_stops_at_contact():
    ocr = make_ocr(
        ["EXAMPLE STORE", "12 MAIN ROAD", "UDUPI KARNATAKA", "576104", "TEL 1234567890", "17 BACK ROAD"]
    )
    result, decision = reconstruct_address(ocr, FieldPrediction(), RecoveryConfig())
    assert result.raw_value == "12 MAIN ROAD\nUDUPI KARNATAKA\n576104"
    assert decision["outcome"] == "accepted"
    assert "17 BACK" not in result.raw_value


def test_address_graph_does_not_cross_far_columns_or_select_prices():
    ocr = make_ocr(["12 MAIN ROAD", "UDUPI 576104", "TOTAL 576104.00"])
    for unit in ocr.units:
        if unit.line_id == "l1":
            unit.bbox_xyxy = tuple(v + (650 if i % 2 == 0 else 0) for i, v in enumerate(unit.bbox_xyxy))
            unit.polygon = [(x + 650, y) for x, y in unit.polygon]
    result, _ = reconstruct_address(ocr, FieldPrediction(), RecoveryConfig())
    assert "TOTAL" not in (result.raw_value or "")
    assert not ("MAIN" in (result.raw_value or "") and "UDUPI" in result.raw_value)


def test_address_stops_at_glued_ocr_date_and_tax_labels():
    ocr = make_ocr(["24RIVERROAD", "CENTRAL CITY560002", "DATE03/04/2025", "GSTIN29ABCDE"])
    result, _ = reconstruct_address(ocr, FieldPrediction(), RecoveryConfig())
    assert result.raw_value == "24RIVERROAD\nCENTRAL CITY560002"


@pytest.mark.parametrize("variant", ["identity", "deskew", "perspective"])
def test_crop_evidence_roundtrip_including_parent_transform(receipt_bytes, tmp_path, variant):
    kwargs = (
        {"deskew_angle": 7}
        if variant == "deskew"
        else {"corners": [[30, 30], [590, 40], [570, 860], [40, 870]]}
        if variant == "perspective"
        else {}
    )
    image = prepare_image(receipt_bytes, variant=variant, artifact_root=tmp_path, **kwargs)
    box = (10, 20, min(400, image.processed_size[0]), min(500, image.processed_size[1]))
    crop, matrix = prepare_crop(image, box, "contrast", RecoveryConfig(), tmp_path)
    original_points = [[50, 80], [100, 90], [100, 120], [50, 120]]
    parent_points = transform_points(original_points, image.display_to_processed)
    crop_points = transform_points(parent_points, matrix)
    from receiptsense.contracts import OCRUnit

    raw = OCRRecord(
        receipt_id=image.receipt_id,
        image_width=crop.processed_size[0],
        image_height=crop.processed_size[1],
        ocr_fingerprint="fixture",
        units=[
            OCRUnit(
                unit_id="u",
                text="ROAD",
                polygon=crop_points.tolist(),
                bbox_xyxy=(*crop_points.min(0), *crop_points.max(0)),
                line_id="l",
            )
        ],
    )
    mapped = map_retry_units(raw, image, matrix, "retry1")[0]
    restored = transform_points(mapped.polygon, np.linalg.inv(image.display_to_processed))
    np.testing.assert_allclose(restored, original_points, atol=1e-6)
    assert mapped.unit_id == "retry1_u" and mapped.attempt_id == "retry1"
    assert max(crop.processed_size) <= 1600


def test_working_ocr_replaces_region_instead_of_duplicating():
    base = make_ocr(["A", "B", "C"])
    replacement = base.units[1].model_copy(
        update={"text": "FIXED", "unit_id": "retry1_x", "attempt_id": "retry1"}
    )
    work = working_record(base, [replacement], (0, 60, 1000, 110))
    assert [u.text for u in work.units] == ["A", "FIXED", "C"]
    assert [u.text for u in base.units] == ["A", "B", "C"]


class FixtureEngine:
    fingerprint = "fixture-v1"

    def __init__(self, base, empty=False, fail=False):
        self.base, self.empty, self.fail, self.calls = base, empty, fail, 0

    def recognize(self, image):
        self.calls += 1
        if self.fail and self.calls > 1:
            raise RuntimeError("OCR unavailable")
        units = [] if self.empty else self.base.units
        return self.base.model_copy(
            update={
                "receipt_id": image.receipt_id,
                "image_width": image.processed_size[0],
                "image_height": image.processed_size[1],
                "units": units,
            }
        )


def test_two_retry_budget_and_warm_cache_accounting(receipt_bytes, tmp_path):
    engine = FixtureEngine(make_ocr([]), empty=True)
    first = process_receipt(receipt_bytes, artifact_root=tmp_path, engine=engine, enhancement="combined")
    assert first.prediction.recovery.logical_attempts == 2
    assert first.prediction.recovery.uncached_ocr_calls == 2
    assert engine.calls == 3
    second = process_receipt(receipt_bytes, artifact_root=tmp_path, engine=engine, enhancement="combined")
    assert second.prediction.recovery.logical_attempts == 2
    assert second.prediction.recovery.uncached_ocr_calls == 0 and engine.calls == 3
    assert first.ocr.ocr_fingerprint == second.ocr.ocr_fingerprint
    assert all(a.reason == "empty_retry_ocr" for a in second.prediction.recovery.attempts)


def test_retry_error_keeps_original_values_and_evidence(receipt_bytes, tmp_path):
    engine = FixtureEngine(
        make_ocr(["EXAMPLE STORE", "12 MAIN ROAD", "03/04/2025", "TOTAL 10.00"]), fail=True
    )
    result = process_receipt(receipt_bytes, artifact_root=tmp_path, engine=engine, enhancement="retry")
    recovery = result.prediction.recovery
    assert 0 < recovery.uncached_ocr_calls <= 2
    assert any(a.outcome == "error" for a in recovery.attempts)
    assert {f: v.raw_value for f, v in result.prediction.fields.items()} == {
        f: v.raw_value for f, v in recovery.baseline_fields.items()
    }
    result.prediction.check_evidence(result.ocr)


def test_priority_is_missing_then_conflict_then_weak(clear_ocr):
    prediction = validate_fields(extract_rules(clear_ocr))
    prediction.fields["company"] = FieldPrediction()
    prediction.fields["address"].status = "ambiguous"
    prediction.fields["total"].score_kind, prediction.fields["total"].score = (
        "uncalibrated_first_subword_mean",
        0.2,
    )
    assert [t[2] for t in recovery_targets(prediction, clear_ocr, RecoveryConfig())][:3] == [
        "company",
        "address",
        "total",
    ]


def test_legacy_prediction_defaults_do_not_break_immutability(clear_ocr, tmp_path):
    prediction = validate_fields(extract_rules(clear_ocr))
    payload = prediction.model_dump(mode="json")
    payload.pop("recovery")
    store = ReviewStore(tmp_path / "legacy.sqlite3")
    with store.connect() as db:
        db.execute(
            "INSERT INTO predictions VALUES (?,?,?,?)",
            (prediction.prediction_id, prediction.receipt_id, prediction.created_at, json.dumps(payload)),
        )
    restored = store.get_prediction(prediction.prediction_id)
    store.save_prediction(restored)
    review = store.save_review(
        restored.prediction_id, {f: restored.fields[f].raw_value for f in FIELDS}, confirm=True
    )
    assert review.state == "confirmed" and restored.recovery is None


def test_invalid_mode_and_budget_rejected(receipt_bytes, tmp_path):
    with pytest.raises(ValueError, match="enhancement"):
        process_receipt(receipt_bytes, enhancement="unbounded", artifact_root=tmp_path)
    with pytest.raises(ValueError):
        RecoveryConfig(max_attempts=3)


def test_model_metadata_without_weights_is_not_available(tmp_path):
    from receiptsense.common import LABELS
    from receiptsense.model import trained_bundle_available

    (tmp_path / "receiptsense_model.json").write_text(json.dumps({"trained": True, "labels": LABELS}))
    assert not trained_bundle_available(tmp_path)


def test_higher_ocr_score_alone_cannot_overwrite_usable_total():
    ocr = make_ocr(["TOTAL 99.00"])
    for unit in ocr.units:
        unit.attempt_id = "retry1"
    original = FieldPrediction(raw_value="10.00", status="predicted", evidence_unit_ids=["base"])
    candidate = validate_fields(extract_rules(ocr)).fields["total"]
    accepted, reason = accept_candidate("total", original, candidate, ocr, RecoveryConfig(), "rules")
    assert not accepted and reason == "usable_original_retained_without_structural_corroboration"
    accepted, _ = accept_candidate("total", FieldPrediction(), candidate, ocr, RecoveryConfig(), "rules")
    assert accepted


def test_accepted_retry_keeps_both_base_and_retry_evidence(receipt_bytes, tmp_path):
    from receiptsense.export import json_export

    class RecoveringEngine(FixtureEngine):
        def recognize(self, image):
            self.calls += 1
            content = (
                make_ocr(["EXAMPLE STORE"]) if self.calls == 1 else make_ocr(["12 MAIN ROAD", "UDUPI 576104"])
            )
            units = []
            for unit in content.units:
                scale_x, scale_y = image.processed_size[0] / 1000, image.processed_size[1] / 2000
                polygon = [(x * scale_x, y * scale_y) for x, y in unit.polygon]
                xs, ys = zip(*polygon, strict=True)
                units.append(
                    unit.model_copy(
                        update={"polygon": polygon, "bbox_xyxy": (min(xs), min(ys), max(xs), max(ys))}
                    )
                )
            return OCRRecord(
                receipt_id=image.receipt_id,
                image_width=image.processed_size[0],
                image_height=image.processed_size[1],
                ocr_fingerprint=self.fingerprint,
                units=units,
            )

    result = process_receipt(
        receipt_bytes, artifact_root=tmp_path, engine=RecoveringEngine(make_ocr([])), enhancement="combined"
    )
    assert result.prediction.recovery.attempts[0].outcome == "accepted"
    assert result.prediction.fields["address"].raw_value == "12 MAIN ROAD\nUDUPI 576104"
    assert any(u.attempt_id == "base" for u in result.ocr.units)
    assert any(u.attempt_id == "retry1" for u in result.ocr.units)
    result.prediction.check_evidence(result.ocr)
    assert json.loads(json_export(result.prediction))["prediction"]["recovery"]["logical_attempts"] <= 2
