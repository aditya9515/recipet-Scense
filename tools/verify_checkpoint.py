"""Reload the selected bundle and compare deployed validation predictions."""

from receiptsense.common import FIELDS, ROOT, digest, read_json, read_jsonl, utc_now, write_json
from receiptsense.contracts import ImageRecord, OCRRecord
from receiptsense.evaluation import score_predictions
from receiptsense.model import LayoutExtractor
from receiptsense.validation import validate_fields


def main():
    checkpoint = ROOT / "artifacts/models/receiptsense-layoutlmv3"
    metadata = read_json(checkpoint / "receiptsense_model.json")
    actual_hash = digest((checkpoint / "model.safetensors").read_bytes())
    if metadata["weight_sha256"] != actual_hash:
        raise ValueError("Saved checkpoint weight hash differs from its manifest")
    extractor = LayoutExtractor(checkpoint)
    entries = read_jsonl(ROOT / "data/manifests/validation.jsonl")
    cached = {row["receipt_id"]: row for row in read_jsonl(checkpoint / "validation_predictions.jsonl")}
    predictions, differences = [], []
    for index, entry in enumerate(entries):
        record = read_json(ROOT / "data/processed/records/identity" / f"{entry['receipt_id']}.json")
        image, ocr = ImageRecord.model_validate(record["image"]), OCRRecord.model_validate(record["ocr"])
        prediction = validate_fields(extractor.predict(image, ocr))
        predictions.append(prediction)
        for field in FIELDS:
            before = cached[entry["receipt_id"]]["fields"][field]["raw_value"]
            after = prediction.fields[field].raw_value
            if before != after:
                differences.append(
                    {"receipt_id": entry["receipt_id"], "field": field, "cached": before, "reloaded": after}
                )
        print(f"Checkpoint reload {index + 1}/{len(entries)}", flush=True)
    report = {
        "created_at": utc_now(),
        "checkpoint": str(checkpoint),
        "weight_hash_verified": True,
        "n_validation_receipts": len(entries),
        "cached_feature_precision": "float16 storage, float32 classifier",
        "deployed_inference_precision": "float32",
        "cached_selection_macro_field_em": metadata["best_validation_macro_field_em"],
        "reloaded_validation_metrics": score_predictions(entries, predictions),
        "raw_field_differences": differences,
        "raw_field_agreement": 1 - len(differences) / (len(entries) * len(FIELDS)),
        "note": "Validation-only reload diagnostic; no checkpoint reselection or test tuning.",
    }
    write_json(ROOT / "artifacts/checkpoint_reload_check.json", report)
    print({"raw_field_agreement": report["raw_field_agreement"], "differences": len(differences)})


if __name__ == "__main__":
    main()
