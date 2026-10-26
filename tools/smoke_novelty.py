"""Real CPU enhanced API workflow with synthetic inputs and an isolated DB."""

import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

import api.main as module
from receiptsense.common import ROOT, utc_now, write_json
from receiptsense.contracts import ImageRecord, OCRRecord, PredictionRecord
from receiptsense.imaging import evidence_image
from receiptsense.model import trained_bundle_available
from receiptsense.review_store import ReviewStore
from receiptsense.samples import create_samples


def main():
    paths = create_samples()
    conditions = [(p, "rules") for p in paths]
    if trained_bundle_available(ROOT / "artifacts/models/receiptsense-layoutlmv3"):
        conditions.append((next(p for p in paths if "clear" in p.name), "layoutlmv3"))
    previous = module.get_store
    checks = []
    with tempfile.TemporaryDirectory() as directory:
        local_store = ReviewStore(Path(directory) / "reviews.sqlite3")
        module.get_store = lambda: local_store
        try:
            with TestClient(module.app) as client:
                for path, method in conditions:
                    upload = client.post(
                        "/receipts", files={"file": (path.name, path.read_bytes(), "image/png")}
                    )
                    assert upload.status_code == 201, upload.text
                    extracted = client.post(
                        f"/receipts/{upload.json()['receipt_id']}/extract",
                        json={"method": method, "enhancement": "combined"},
                    )
                    assert extracted.status_code == 200, extracted.text
                    prediction = PredictionRecord.model_validate(extracted.json())
                    report = prediction.recovery
                    assert report and report.logical_attempts <= 2 and report.uncached_ocr_calls <= 2
                    artifact = ROOT / "artifacts/predictions" / prediction.prediction_id
                    image = ImageRecord.model_validate_json((artifact / "image.json").read_text())
                    ocr = OCRRecord.model_validate_json((artifact / "ocr.json").read_text())
                    prediction.check_evidence(ocr)
                    overlay = evidence_image(image, ocr, prediction.fields["address"].evidence_unit_ids)
                    assert overlay.size == image.display_size
                    if method == "rules" and "clear" in path.name:
                        assert prediction.fields["company"].raw_value == "EXAMPLE STORE"
                        assert prediction.fields["total"].raw_value == "493.50"
                    values = {f: value.raw_value for f, value in prediction.fields.items()}
                    values["company"], values["date"] = "SYNTHETIC REVIEWED STORE", "2025-04-03"
                    reviewed = client.post(
                        f"/predictions/{prediction.prediction_id}/reviews",
                        json={"values": values, "confirm": True},
                    )
                    assert reviewed.status_code == 201, reviewed.text
                    review_id = reviewed.json()["review_id"]
                    exported = client.get(f"/reviews/{review_id}/export?format=json")
                    csv_result = client.get(f"/reviews/{review_id}/export?format=csv")
                    assert exported.json()["review_status"] == "confirmed"
                    assert exported.json()["fields"]["company"] == values["company"]
                    assert exported.json()["prediction"]["recovery"]["mode"] == "combined"
                    assert csv_result.status_code == 200 and "combined" in csv_result.text
                    original = client.get(f"/predictions/{prediction.prediction_id}").json()
                    assert original["fields"] == extracted.json()["fields"]
                    checks.append(
                        {
                            "synthetic_input": path.name,
                            "method": method,
                            "enhancement": "combined",
                            "logical_attempts": report.logical_attempts,
                            "uncached_retry_calls": report.uncached_ocr_calls,
                            "evidence_overlay": "passed",
                            "confirmed_export": "passed",
                            "original_preserved": True,
                        }
                    )
        finally:
            module.get_store = previous
    write_json(
        ROOT / "artifacts/novelty/workflow_check.json",
        {
            "checked_at": utc_now(),
            "passed": True,
            "device": "cpu",
            "synthetic_only": True,
            "database_cleanup": "passed",
            "checks": checks,
        },
    )
    print("Real enhanced CPU API, evidence overlay, corrections, exports and immutable history: passed")


if __name__ == "__main__":
    main()
