"""Receipt-level strict/normalized metrics; missing predictions remain failures."""

from __future__ import annotations

import random
from pathlib import Path
from statistics import mean
from time import perf_counter
from uuid import uuid4

import numpy as np

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
from .contracts import ImageRecord, OCRRecord, PredictionRecord
from .model import get_extractor
from .rules import extract_rules
from .validation import clean_text, parse_amount, parse_date, validate_fields


def strict_value(value: str | None) -> str | None:
    return clean_text(value) if value is not None and value.strip() else None


def normalized_value(field: str, value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    if field == "date":
        return parse_date(value)[0]
    if field == "total":
        return parse_amount(value)[0]
    return clean_text(value).casefold()


def field_correct(field: str, prediction: str | None, reference: str | None, policy: str) -> bool:
    if policy not in {"strict-v1", "normalized-v1"}:
        raise ValueError("Unknown metric policy")
    left = strict_value(prediction) if policy == "strict-v1" else normalized_value(field, prediction)
    right = strict_value(reference) if policy == "strict-v1" else normalized_value(field, reference)
    return left is not None and right is not None and left == right


def score_predictions(
    entries: list[dict], predictions: list[PredictionRecord | dict], policy: str = "strict-v1"
) -> dict:
    mapped = {}
    for prediction in predictions:
        prediction = (
            prediction
            if isinstance(prediction, PredictionRecord)
            else PredictionRecord.model_validate(prediction)
        )
        if prediction.receipt_id in mapped:
            raise ValueError("Duplicate prediction receipt ID; evaluate one method at a time")
        mapped[prediction.receipt_id] = prediction
    if not entries or len({e["receipt_id"] for e in entries}) != len(entries):
        raise ValueError("Evaluation manifest must be nonempty and have unique receipt IDs")
    rows = []
    for entry in entries:
        prediction = mapped.get(entry["receipt_id"])
        eligible = [field for field in FIELDS if entry["reference"].get(field)]
        correct = {
            f: field_correct(
                f, prediction.fields[f].raw_value if prediction else None, entry["reference"][f], policy
            )
            for f in eligible
        }
        flagged = (
            prediction is None
            or bool(prediction.errors)
            or any(f.review_reasons or f.status != "predicted" for f in prediction.fields.values())
        )
        rows.append(
            {
                "receipt_id": entry["receipt_id"],
                "source_key": entry.get("source_key"),
                "correct": correct,
                "all_four": len(eligible) == 4 and all(correct.values()),
                "has_four_references": len(eligible) == 4,
                "flagged": bool(flagged),
                "missing_prediction": prediction is None,
                "reference": entry["reference"],
                "predicted": {f: prediction.fields[f].raw_value if prediction else None for f in FIELDS},
            }
        )
    per_field = {}
    for field in FIELDS:
        subset = [r["correct"][field] for r in rows if field in r["correct"]]
        per_field[field] = {
            "correct": sum(subset),
            "n": len(subset),
            "exact_match": mean(subset) if subset else None,
        }
    available = [v["exact_match"] for v in per_field.values() if v["exact_match"] is not None]
    four = [r for r in rows if r["has_four_references"]]
    wrong = [r for r in rows if not all(r["correct"].values())]
    unflagged = [r for r in rows if not r["flagged"]]
    return {
        "policy": policy,
        "n_receipts": len(rows),
        "per_field": per_field,
        "macro_field_em": mean(available) if available else None,
        "all_four_em": mean(r["all_four"] for r in four) if four else None,
        "all_four_denominator": len(four),
        "missing_predictions": sum(r["missing_prediction"] for r in rows),
        "review_rate": mean(r["flagged"] for r in rows),
        "review_error_recall": mean(r["flagged"] for r in wrong) if wrong else None,
        "unflagged_accuracy": mean(all(r["correct"].values()) for r in unflagged) if unflagged else None,
        "rows": rows,
    }


def paired_bootstrap(first: dict, second: dict, samples: int = 1000, seed: int = 42) -> dict:
    a = {r["receipt_id"]: r for r in first["rows"]}
    b = {r["receipt_id"]: r for r in second["rows"]}
    if set(a) != set(b) or first["policy"] != second["policy"]:
        raise ValueError("Paired comparison requires identical receipts and metric policy")
    differences = [mean(b[i]["correct"].values()) - mean(a[i]["correct"].values()) for i in sorted(a)]
    rng = random.Random(seed)
    estimates = [mean(rng.choices(differences, k=len(differences))) for _ in range(samples)]
    return {
        "difference": mean(differences),
        "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "bootstrap_receipts": len(differences),
        "samples": samples,
        "seed": seed,
    }


def evaluate_manifest(
    manifest: str | Path,
    *,
    records: str | Path = ROOT / "data/processed/records/identity",
    methods: tuple[str, ...] = ("rules", "layoutlmv3"),
    checkpoint: str | Path | None = None,
    limit: int | None = None,
    run_name: str | None = None,
) -> dict:
    entries = read_jsonl(manifest)
    entries = entries[:limit] if limit else entries
    folder = ROOT / "artifacts/runs" / (run_name or f"evaluation-{uuid4().hex[:10]}")
    folder.mkdir(parents=True, exist_ok=False)
    extractor = (
        get_extractor(str(checkpoint or ROOT / "artifacts/models/receiptsense-layoutlmv3"))
        if "layoutlmv3" in methods
        else None
    )
    outputs = {method: [] for method in methods}
    failures = []
    elapsed = {method: [] for method in methods}
    fingerprints = set()
    for index, entry in enumerate(entries):
        try:
            record = read_json(Path(records) / f"{entry['receipt_id']}.json")
            image, ocr = ImageRecord.model_validate(record["image"]), OCRRecord.model_validate(record["ocr"])
            fingerprints.add(ocr.ocr_fingerprint)
        except Exception as exc:
            failures.append({"receipt_id": entry["receipt_id"], "method": "all", "error": str(exc)})
            continue
        for method in methods:
            start = perf_counter()
            try:
                prediction = extract_rules(ocr) if method == "rules" else extractor.predict(image, ocr)
                prediction = validate_fields(prediction)
                prediction.check_evidence(ocr)
                elapsed[method].append(perf_counter() - start)
                outputs[method].append(prediction.model_dump(mode="json"))
            except Exception as exc:
                failures.append({"receipt_id": entry["receipt_id"], "method": method, "error": str(exc)})
        print(
            f"Evaluate {index + 1}/{len(entries)} {entry.get('source_key', entry['receipt_id'])}", flush=True
        )
    if len(fingerprints) > 1:
        raise ValueError("Mixed OCR fingerprints in one experiment; separate engine conditions first")
    report = {
        "created_at": utc_now(),
        "manifest": str(Path(manifest).resolve()),
        "source_fingerprint": source_fingerprint(),
        "split_hash": digest([e["receipt_id"] for e in entries]),
        "record_directory": str(Path(records).resolve()),
        "n_receipts": len(entries),
        "partial_subset": limit is not None,
        "ocr_fingerprints": sorted(fingerprints),
        "failures": failures,
        "results": {},
        "latency_scope": "cached OCR extraction+validation only; excludes OCR and startup",
    }
    for method, predictions in outputs.items():
        write_jsonl(folder / f"{method}_predictions.jsonl", predictions)
        report["results"][method] = {}
        for policy in ["strict-v1", "normalized-v1"]:
            metrics = score_predictions(entries, predictions, policy)
            write_json(folder / f"{method}_{policy}.json", metrics)
            report["results"][method][policy] = {k: v for k, v in metrics.items() if k != "rows"}
        timings = elapsed[method]
        report["results"][method]["cached_latency"] = {
            "n": len(timings),
            "median_seconds": float(np.median(timings)) if timings else None,
            "p95_seconds": float(np.quantile(timings, 0.95)) if timings else None,
        }
    if set(methods) == {"rules", "layoutlmv3"}:
        report["paired_strict"] = paired_bootstrap(
            read_json(folder / "rules_strict-v1.json"), read_json(folder / "layoutlmv3_strict-v1.json")
        )
    write_json(folder / "summary.json", report)
    return report
