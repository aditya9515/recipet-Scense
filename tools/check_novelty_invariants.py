"""Audit saved ablations against the published baseline and recovery invariants."""

from receiptsense.common import FIELDS, ROOT, read_json, read_jsonl, write_json
from receiptsense.contracts import OCRRecord, PredictionRecord
from receiptsense.novelty_evaluation import checkpoint_identity, policy_fingerprint


def main():
    folder = ROOT / "artifacts/novelty/validation-v1"
    summary = read_json(folder / "summary.json")
    preserved = read_json(ROOT / "results/novelty/baseline_manifest.json")
    assert checkpoint_identity(verify_weights=True) == preserved["checkpoint"]
    assert summary["policy_fingerprint"] == policy_fingerprint()
    comparisons = {
        "layoutlmv3": ROOT / "artifacts/models/receiptsense-layoutlmv3/validation_predictions.jsonl",
        "rules": ROOT / "artifacts/runs/validation-rules-v1/rules_predictions.jsonl",
    }
    agreement = {}
    for method, path in comparisons.items():
        before = {r["receipt_id"]: r for r in read_jsonl(path)}
        after = read_jsonl(folder / f"{method}_none_predictions.jsonl")
        assert len(before) == len(after) == 61
        differences = []
        for prediction in after:
            for field in FIELDS:
                if (
                    prediction["fields"][field]["raw_value"]
                    != before[prediction["receipt_id"]]["fields"][field]["raw_value"]
                ):
                    differences.append((prediction["receipt_id"], field))
        assert not differences, differences
        agreement[method] = {"receipts": 61, "raw_field_agreement": 1.0}
    records, attempts, max_attempts = 0, 0, 0
    for path in folder.glob("*_predictions.jsonl"):
        fingerprints = set()
        for raw in read_jsonl(path):
            prediction = PredictionRecord.model_validate(raw)
            source = ROOT / "artifacts/predictions" / prediction.prediction_id / "ocr.json"
            ocr = OCRRecord.model_validate(read_json(source))
            prediction.check_evidence(ocr)
            fingerprints.add(prediction.ocr_fingerprint)
            records += 1
            if prediction.recovery:
                recovery = prediction.recovery
                assert recovery.logical_attempts == len(recovery.attempts) <= 2
                assert recovery.uncached_ocr_calls == sum(a.ocr_called for a in recovery.attempts) <= 2
                assert all(not (a.cache_hit and a.ocr_called) for a in recovery.attempts)
                ids = {u.unit_id for u in ocr.units}
                assert all(set(a.evidence_unit_ids) <= ids for a in recovery.attempts)
                attempts += recovery.logical_attempts
                max_attempts = max(max_attempts, recovery.logical_attempts)
        assert len(fingerprints) == 1
    assert records == 488
    write_json(
        ROOT / "results/novelty/invariants.json",
        {
            "passed": True,
            "checkpoint_unchanged": True,
            "baseline_validation_reproduction": agreement,
            "predictions_audited": records,
            "retry_attempts_audited": attempts,
            "maximum_attempts_observed": max_attempts,
            "all_evidence_resolves": True,
            "one_fingerprint_per_condition": True,
        },
    )
    print("Verified frozen head, all 488 predictions, retry budget, evidence and original baseline fields")


if __name__ == "__main__":
    main()
