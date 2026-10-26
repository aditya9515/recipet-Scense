"""Validation-only policy selection, paired ablations and isolated CPU timing."""

from __future__ import annotations

import itertools
import random
import re
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np

from .address import reconstruct_address
from .common import (
    FIELDS,
    ROOT,
    digest,
    read_json,
    read_jsonl,
    source_fingerprint,
    utc_now,
    write_json,
    write_jsonl,
)
from .contracts import ImageRecord, OCRRecord
from .evaluation import field_correct, paired_bootstrap, score_predictions
from .model import get_extractor
from .ocr_adapter import get_engine
from .pipeline import process_receipt
from .recovery_config import ENHANCEMENTS, load_config
from .rules import extract_rules
from .validation import validate_field, validate_fields

CHECKPOINT = ROOT / "artifacts/models/receiptsense-layoutlmv3"
POLICY_FILES = (
    "address.py",
    "recovery.py",
    "recovery_config.py",
    "rules.py",
    "validation.py",
    "reading_order.py",
    "imaging.py",
    "model.py",
    "pipeline.py",
    "contracts.py",
)


def policy_fingerprint():
    return digest({name: digest((ROOT / "src/receiptsense" / name).read_bytes()) for name in POLICY_FILES})


def checkpoint_identity(checkpoint=CHECKPOINT, verify_weights=False):
    path = Path(checkpoint)
    metadata = read_json(path / "receiptsense_model.json")
    if verify_weights and digest((path / "model.safetensors").read_bytes()) != metadata["weight_sha256"]:
        raise ValueError("Frozen checkpoint weight hash mismatch")
    return {
        "weight_sha256": metadata["weight_sha256"],
        "manifest_sha256": digest((path / "receiptsense_model.json").read_bytes()),
        "adaptation": metadata["adaptation"],
        "selected_epoch": metadata["selected_epoch"],
    }


def address_token_f1(predicted, reference):
    left = Counter(re.findall(r"\w+", (predicted or "").casefold()))
    right = Counter(re.findall(r"\w+", reference.casefold()))
    overlap = sum((left & right).values())
    return 2 * overlap / (sum(left.values()) + sum(right.values())) if left and right else 0.0


def merchant_bootstrap(entries, first, second, samples=1000):
    if not all(e.get("merchant_id") for e in entries):
        return None
    a, b = {r["receipt_id"]: r for r in first["rows"]}, {r["receipt_id"]: r for r in second["rows"]}
    groups = {}
    for entry in entries:
        rid = entry["receipt_id"]
        groups.setdefault(entry["merchant_id"], []).append(
            mean(b[rid]["correct"].values()) - mean(a[rid]["correct"].values())
        )
    if len(groups) < 2:
        return None
    rng, keys = random.Random(42), sorted(groups)
    draws = [
        mean(value for key in rng.choices(keys, k=len(keys)) for value in groups[key]) for _ in range(samples)
    ]
    return {
        "difference": mean(v for values in groups.values() for v in values),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "n_merchants": len(keys),
        "samples": samples,
        "seed": 42,
        "scope": "merchant-cluster bootstrap; small-pilot uncertainty, not a risk certificate",
    }


def tune_address(manifest, *, records=ROOT / "data/processed/records/identity", checkpoint=CHECKPOINT):
    entries = read_jsonl(manifest)
    if not entries or any(e.get("split") != "validation" for e in entries):
        raise ValueError("Policy selection requires validation records only")
    model = get_extractor(str(checkpoint))
    baseline = []
    for entry in entries:
        raw = read_json(Path(records) / f"{entry['receipt_id']}.json")
        image, ocr = ImageRecord.model_validate(raw["image"]), OCRRecord.model_validate(raw["ocr"])
        for method in ("rules", "layoutlmv3"):
            prediction = extract_rules(ocr) if method == "rules" else model.predict(image, ocr)
            baseline.append((entry, ocr, validate_fields(prediction), method))
    config, trials = load_config(), []
    for gap, overlap in itertools.product((1.5, 2.0, 2.5), (0.2, 0.4)):
        candidate = config.model_copy(update={"address_gap_heights": gap, "address_overlap": overlap})
        addresses, all_fields, changes = [], [], 0
        for entry, ocr, prediction, _ in baseline:
            new, _ = reconstruct_address(ocr, prediction.fields["address"], candidate)
            new = validate_field("address", new)
            addresses.append(
                field_correct("address", new.raw_value, entry["reference"]["address"], "strict-v1")
            )
            all_fields.extend(
                field_correct(
                    f,
                    new.raw_value if f == "address" else prediction.fields[f].raw_value,
                    entry["reference"][f],
                    "strict-v1",
                )
                for f in FIELDS
            )
            changes += new.raw_value != prediction.fields["address"].raw_value
        trials.append(
            {
                "gap": gap,
                "overlap": overlap,
                "address_em": mean(addresses),
                "macro_em": mean(all_fields),
                "changed_fields": changes,
            }
        )
    selected = max(
        trials, key=lambda t: (t["address_em"], t["macro_em"], -t["changed_fields"], -t["gap"], t["overlap"])
    )
    frozen = config.model_copy(
        update={"address_gap_heights": selected["gap"], "address_overlap": selected["overlap"]}
    )
    write_json(ROOT / "configs/recovery-v1.json", frozen.model_dump(mode="json"))
    report = {
        "created_at": utc_now(),
        "n_validation_receipts": len(entries),
        "selection": "mean strict address EM over both methods; macro EM, fewer changes, smaller gap, larger overlap break ties",
        "configuration_hash": frozen.fingerprint,
        "policy_fingerprint": policy_fingerprint(),
        "trials": trials,
        "selected": selected,
        "checkpoint": checkpoint_identity(checkpoint, True),
        "base_ocr_fingerprint": get_engine().fingerprint,
        "split_hash": digest([e["receipt_id"] for e in entries]),
    }
    write_json(ROOT / "results/novelty/policy_selection.json", report)
    return report


def compare_outputs(entries, outputs, baseline):
    strict = score_predictions(entries, outputs)
    normalized = score_predictions(entries, outputs, "normalized-v1")
    paired = paired_bootstrap(baseline, strict)
    previous = {r["receipt_id"]: r for r in baseline["rows"]}
    corrected = {f: 0 for f in FIELDS}
    introduced = {f: 0 for f in FIELDS}
    for row in strict["rows"]:
        for field, correct in row["correct"].items():
            old = previous[row["receipt_id"]]["correct"][field]
            corrected[field] += correct and not old
            introduced[field] += old and not correct
    mapped = {p.receipt_id: p for p in outputs}
    reports = [p.recovery for p in outputs if p.recovery]
    timings = [p.timings["total_seconds"] for p in outputs]
    return {
        "strict-v1": {k: v for k, v in strict.items() if k != "rows"},
        "normalized-v1": {k: v for k, v in normalized.items() if k != "rows"},
        "address_token_f1": mean(
            address_token_f1(
                mapped[e["receipt_id"]].fields["address"].raw_value if e["receipt_id"] in mapped else None,
                e["reference"]["address"],
            )
            for e in entries
        ),
        "corrected_fields": corrected,
        "introduced_errors": introduced,
        "paired_receipt_bootstrap": paired,
        "paired_merchant_bootstrap": merchant_bootstrap(entries, baseline, strict),
        "retry_rate": sum(bool(p.recovery and p.recovery.logical_attempts) for p in outputs) / len(entries),
        "logical_retry_attempts": sum(r.logical_attempts for r in reports),
        "uncached_retry_ocr_calls": sum(r.uncached_ocr_calls for r in reports),
        "uncached_base_ocr_calls": sum(int(not bool(p.timings["ocr_cache_hit"])) for p in outputs),
        "pipeline_seconds": {
            "n": len(timings),
            "median": float(np.median(timings)) if timings else None,
            "p95": float(np.quantile(timings, 0.95)) if timings else None,
            "scope": "accuracy run; caches allowed; not an uncached latency claim",
        },
    }


def run_ablation(manifest, *, name, checkpoint=CHECKPOINT, config=None, confirmation_lock=None):
    entries = read_jsonl(manifest)
    config = config or load_config()
    if confirmation_lock:
        from .confirmation import verify_confirmation

        if config.fingerprint != load_config().fingerprint:
            raise ValueError("Confirmation cannot override the frozen configuration")
        verify_confirmation(manifest, confirmation_lock, checkpoint=checkpoint)
        purpose = "fresh confirmation pilot"
    elif entries and all(e.get("split") == "validation" for e in entries):
        purpose = "validation development; not independent confirmation"
    else:
        raise ValueError("Use validation data, or a verified frozen confirmation manifest")
    folder = ROOT / "artifacts/novelty" / name
    folder.mkdir(parents=True, exist_ok=False)
    report = {
        "created_at": utc_now(),
        "complete": False,
        "purpose": purpose,
        "n_receipts": len(entries),
        "configuration": config.model_dump(mode="json"),
        "configuration_hash": config.fingerprint,
        "policy_fingerprint": policy_fingerprint(),
        "source_fingerprint": source_fingerprint(),
        "base_ocr_fingerprint": get_engine().fingerprint,
        "checkpoint": checkpoint_identity(checkpoint, True),
        "split_hash": digest([e["receipt_id"] for e in entries]),
        "fresh_data_published": False,
        "results": {},
        "failures": [],
    }
    for method in ("rules", "layoutlmv3"):
        base_metrics = None
        for mode in ENHANCEMENTS:
            predictions, failure_count = [], 0
            for index, entry in enumerate(entries):
                try:
                    path = Path(entry["image_path"])
                    result = process_receipt(
                        path.read_bytes(),
                        path.name,
                        method=method,
                        checkpoint=checkpoint,
                        enhancement=mode,
                        recovery_config=config,
                    )
                    predictions.append(result.prediction)
                except Exception as exc:
                    failure_count += 1
                    # Detailed, potentially sensitive failure records stay local.
                    with (folder / "failures.jsonl").open("a", encoding="utf-8") as stream:
                        import json

                        stream.write(
                            json.dumps(
                                {
                                    "receipt_id": entry["receipt_id"],
                                    "method": method,
                                    "mode": mode,
                                    "error": str(exc),
                                }
                            )
                            + "\n"
                        )
                print(f"Ablation {method}/{mode} {index + 1}/{len(entries)}", flush=True)
            write_jsonl(
                folder / f"{method}_{mode}_predictions.jsonl",
                [p.model_dump(mode="json") for p in predictions],
            )
            if mode == "none":
                base_metrics = score_predictions(entries, predictions)
            metrics = compare_outputs(entries, predictions, base_metrics)
            report["results"][f"{method}/{mode}"] = metrics
            if failure_count:
                report["failures"].append({"method": method, "mode": mode, "count": failure_count})
            write_json(folder / "progress.json", report)
    report["complete"] = True
    report["combined_default_gate"] = {
        method: report["results"][f"{method}/combined"]["strict-v1"]["per_field"]["address"]["exact_match"]
        > report["results"][f"{method}/none"]["strict-v1"]["per_field"]["address"]["exact_match"]
        and report["results"][f"{method}/combined"]["strict-v1"]["macro_field_em"]
        >= report["results"][f"{method}/none"]["strict-v1"]["macro_field_em"]
        for method in ("rules", "layoutlmv3")
    }
    write_json(folder / "summary.json", report)
    return report


def benchmark_enhancements(manifest, *, name, limit=10, checkpoint=CHECKPOINT):
    entries = read_jsonl(manifest)[:limit]
    if not entries or any(e.get("split") != "validation" for e in entries):
        raise ValueError("The development timing study uses validation receipts")
    conditions = list(itertools.product(("rules", "layoutlmv3"), ENHANCEMENTS))
    config = load_config()
    # Warm engines/models and each code path; every timed OCR still bypasses caches.
    path = Path(entries[0]["image_path"])
    for method, mode in conditions:
        process_receipt(
            path.read_bytes(),
            path.name,
            method=method,
            enhancement=mode,
            checkpoint=checkpoint,
            use_cache=False,
        )
    rows = []
    for index, entry in enumerate(entries):
        order = conditions[index % len(conditions) :] + conditions[: index % len(conditions)]
        path = Path(entry["image_path"])
        for method, mode in order:
            result = process_receipt(
                path.read_bytes(),
                path.name,
                method=method,
                enhancement=mode,
                checkpoint=checkpoint,
                use_cache=False,
            )
            rows.append({"method": method, "mode": mode, **result.prediction.timings})
        print(f"Uncached enhancement latency {index + 1}/{len(entries)}", flush=True)
    report = {
        "created_at": utc_now(),
        "device": "cpu",
        "n_receipts": len(entries),
        "warmup_per_condition": 1,
        "scope": "warm uncached image preparation + base OCR + enhancement + extraction + validation; excludes startup and persistence",
        "condition_order": "rotating per receipt to reduce fixed-order bias",
        "configuration_hash": config.fingerprint,
        "policy_fingerprint": policy_fingerprint(),
        "results": {},
        "rows": rows,
    }
    for method, mode in conditions:
        subset = [r for r in rows if r["method"] == method and r["mode"] == mode]
        times = [r["total_seconds"] for r in subset]
        report["results"][f"{method}/{mode}"] = {
            "n": len(times),
            "median_seconds": float(np.median(times)),
            "p95_seconds": float(np.quantile(times, 0.95)),
            "mean_ocr_calls": mean(r["uncached_ocr_calls"] for r in subset),
        }
    write_json(ROOT / "artifacts/novelty" / name / "latency.json", report)
    return report
