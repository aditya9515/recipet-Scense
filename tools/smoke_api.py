"""Exercise the real local API workflow with a synthetic receipt and isolated DB."""

import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

import api.main as module
from receiptsense.common import ROOT, utc_now, write_json
from receiptsense.review_store import ReviewStore
from receiptsense.samples import create_samples


def main():
    path = next(p for p in create_samples() if "clear" in p.name)
    previous = module.get_store
    with tempfile.TemporaryDirectory() as folder:
        store = ReviewStore(Path(folder) / "api.sqlite3")
        module.get_store = lambda: store
        try:
            with TestClient(module.app) as client:
                upload = client.post("/receipts", files={"file": (path.name, path.read_bytes(), "image/png")})
                assert upload.status_code == 201, upload.text
                receipt_id = upload.json()["receipt_id"]
                response = client.post(f"/receipts/{receipt_id}/extract", json={"method": "rules"})
                assert response.status_code == 200, response.text
                prediction = response.json()
                values = {name: field["raw_value"] for name, field in prediction["fields"].items()}
                assert values["company"] == "EXAMPLE STORE" and values["total"] == "493.50", values
                values["company"] = "API SMOKE REVIEWED"
                response = client.post(
                    f"/predictions/{prediction['prediction_id']}/reviews",
                    json={"values": values, "confirm": True},
                )
                assert response.status_code == 201, response.text
                review_id = response.json()["review_id"]
                exported = client.get(f"/reviews/{review_id}/export?format=json")
                csv_result = client.get(f"/reviews/{review_id}/export?format=csv")
                original = client.get(f"/predictions/{prediction['prediction_id']}")
                assert exported.json()["fields"]["company"] == "API SMOKE REVIEWED"
                assert original.json()["fields"]["company"]["raw_value"] == "EXAMPLE STORE"
                assert csv_result.status_code == 200 and "API SMOKE REVIEWED" in csv_result.text
                evidence = {
                    "checked_at": utc_now(),
                    "real_ocr": True,
                    "synthetic_input": path.name,
                    "upload_status": upload.status_code,
                    "extraction_status": 200,
                    "review_status": response.status_code,
                    "json_export_status": exported.status_code,
                    "csv_export_status": csv_result.status_code,
                    "original_preserved": True,
                    "store_scope": "temporary isolated database",
                }
        finally:
            module.get_store = previous
    evidence["temporary_database_cleanup_passed"] = True
    write_json(ROOT / "artifacts/api_smoke_check.json", evidence)
    print("Real API upload, OCR, review, exports, original preservation and DB cleanup: passed")


if __name__ == "__main__":
    main()
