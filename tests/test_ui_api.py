from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from api.main import app
from receiptsense.common import ROOT
from receiptsense.review_store import ReviewStore
from receiptsense.rules import extract_rules
from receiptsense.validation import validate_fields


def test_api_health_review_and_export(clear_ocr, tmp_path):
    store = ReviewStore(tmp_path / "api.sqlite3")
    prediction = validate_fields(extract_rules(clear_ocr))
    store.save_prediction(prediction)
    app.dependency_overrides.clear()
    # Route functions use the cached provider; replace the cache object in module.
    import api.main as api_module

    previous = api_module.get_store
    api_module.get_store = lambda: store
    try:
        client = TestClient(app)
        assert client.get("/health").status_code == 200
        assert client.get("/predictions/unknown").status_code == 404
        response = client.post(
            f"/predictions/{prediction.prediction_id}/reviews",
            json={"values": {n: f.raw_value for n, f in prediction.fields.items()}, "confirm": True},
        )
        assert response.status_code == 201
        review = response.json()
        exported = client.get(f"/reviews/{review['review_id']}/export?format=json")
        assert exported.status_code == 200 and exported.json()["review_status"] == "confirmed"
        assert (
            client.post("/receipts", files={"file": ("bad.png", b"not image", "image/png")}).status_code
            == 422
        )
    finally:
        api_module.get_store = previous


def test_streamlit_initial_screen_and_pages():
    app_test = AppTest.from_file(ROOT / "app/streamlit_app.py", default_timeout=30).run()
    assert not app_test.exception
    assert app_test.title[0].value == "From receipt to record."
    app_test.sidebar.radio[0].set_value("Review history").run()
    assert not app_test.exception
    app_test.sidebar.radio[0].set_value("Experiments").run()
    assert not app_test.exception
    app_test.sidebar.radio[0].set_value("Model & data").run()
    assert not app_test.exception


def test_history_reopens_latest_correction(monkeypatch, tmp_path, receipt_bytes, clear_ocr):
    import streamlit as st

    from receiptsense import common, review_store
    from receiptsense.pipeline import process_receipt

    class FixtureEngine:
        fingerprint = "fixture-v1"

        def recognize(self, image):
            return clear_ocr.model_copy(
                update={
                    "receipt_id": image.receipt_id,
                    "image_width": image.processed_size[0],
                    "image_height": image.processed_size[1],
                }
            )

    result = process_receipt(receipt_bytes, artifact_root=tmp_path / "artifacts", engine=FixtureEngine())
    local_store = ReviewStore(tmp_path / "reviews.sqlite3")
    local_store.save_prediction(result.prediction)
    values = {n: f.raw_value for n, f in result.prediction.fields.items()}
    values["company"] = "Reviewed Store"
    local_store.save_review(result.prediction.prediction_id, values, confirm=True)
    app_path = ROOT / "app/streamlit_app.py"
    st.cache_resource.clear()
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(review_store, "ReviewStore", lambda: local_store)
    try:
        app_test = AppTest.from_file(app_path, default_timeout=30).run()
        app_test.sidebar.radio[0].set_value("Review history").run()
        assert any("Reviewed Store" in item.value for item in app_test.markdown)
        next(b for b in app_test.button if b.label == "Open record").click().run()
        assert not app_test.exception
        assert next(w for w in app_test.text_input if w.label == "Company").value == "Reviewed Store"
    finally:
        st.cache_resource.clear()
