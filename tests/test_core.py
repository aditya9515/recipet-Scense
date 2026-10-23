import csv
import io
import json

import numpy as np
import pytest
from conftest import make_ocr
from pydantic import ValidationError

from receiptsense.alignment import align_fields
from receiptsense.common import FIELDS
from receiptsense.contracts import OCRRecord
from receiptsense.decoding import decode_fields
from receiptsense.encoding import window_owners, window_ranges
from receiptsense.evaluation import field_correct, score_predictions
from receiptsense.export import csv_export, json_export
from receiptsense.imaging import ImageInputError, prepare_image, transform_points
from receiptsense.pipeline import process_receipt
from receiptsense.review_store import ReviewStore
from receiptsense.rules import extract_rules
from receiptsense.validation import parse_amount, parse_date, validate_fields


@pytest.mark.parametrize(
    "text,expected",
    [
        ("493.50", "493.50"),
        ("RM 1,234.50", "1234.50"),
        ("1.234,50", "1234.50"),
        ("1,234", None),
        ("493.SO", None),
        ("", None),
        ("-2.00", "-2.00"),
        ("12.345", None),
        ("RM8.50", "8.50"),
    ],
)
def test_decimal_parsing_does_not_hide_ambiguity(text, expected):
    assert parse_amount(text)[0] == expected


def test_date_ambiguity_and_two_digit_years():
    assert parse_date("03/04/2025")[0] is None
    assert len(parse_date("03/04/2025")[1]) == 2
    assert parse_date("14/02/2025")[0] == "2025-02-14"
    assert parse_date("14/02/25")[0] is None


def test_rules_choose_total_instead_of_larger_cash(clear_ocr):
    result = validate_fields(extract_rules(clear_ocr))
    assert result.fields["total"].raw_value == "493.50"
    assert result.fields["company"].raw_value == "EXAMPLE STORE"
    assert result.fields["address"].raw_value == "12 MAIN ROAD\nCENTRAL CITY"
    assert result.fields["date"].normalized_value == "2025-02-14"
    result.check_evidence(clear_ocr)


def test_empty_ocr_is_missing_not_guessed():
    ocr = make_ocr([])
    result = validate_fields(extract_rules(ocr))
    assert ocr.status == "empty_ocr"
    assert all(item.status == "missing" for item in result.fields.values())


def test_duplicate_and_out_of_frame_geometry_rejected(clear_ocr):
    raw = clear_ocr.model_dump()
    raw["units"].append(raw["units"][0])
    with pytest.raises(ValidationError, match="Duplicate"):
        OCRRecord.model_validate(raw)
    raw = clear_ocr.model_dump()
    raw["units"][0]["polygon"][0] = (-50, 0)
    with pytest.raises(ValidationError, match="outside"):
        OCRRecord.model_validate(raw)


def test_transform_round_trip_and_upload_validation(receipt_bytes, tmp_path):
    record = prepare_image(receipt_bytes, variant="deskew", deskew_angle=8, artifact_root=tmp_path)
    points = [[100, 200], [500, 800]]
    mapped = transform_points(points, record.display_to_processed)
    restored = transform_points(mapped, np.linalg.inv(record.display_to_processed))
    np.testing.assert_allclose(restored, points, atol=1e-6)
    assert record.processed_size[0] > record.display_size[0]
    with pytest.raises(ImageInputError):
        prepare_image(b"not a receipt", artifact_root=tmp_path)


def test_perspective_corner_roundtrip(receipt_bytes, tmp_path):
    corners = [[50, 50], [550, 70], [530, 850], [60, 830]]
    record = prepare_image(receipt_bytes, variant="perspective", corners=corners, artifact_root=tmp_path)
    mapped = transform_points(corners, record.display_to_processed)
    width, height = record.processed_size
    np.testing.assert_allclose(
        mapped, [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], atol=1e-4
    )


def test_alignment_masks_uncertain_negatives_with_repeated_total():
    ocr = make_ocr(["EXAMPLE STORE", "12 MAIN ROAD", "2025-02-14", "ITEM 9.00", "TOTAL 9.00"])
    result = align_fields(
        ocr, {"company": "EXAMPLE STORE", "address": "12 MAIN ROAD", "date": "2025-02-14", "total": "9.00"}
    )
    assert result["usable_for_training"] and not result["fully_aligned"]
    assert "B-TOTAL" not in result["labels"] and "O" not in result["labels"]
    assert result["fields"]["total"]["status"] == "ambiguous"


def test_exact_alignment_and_window_ownership(clear_ocr):
    reference = {
        "company": "EXAMPLE STORE",
        "address": "12 MAIN ROAD CENTRAL CITY",
        "date": "2025-02-14",
        "total": "493.50",
    }
    aligned = align_fields(clear_ocr, reference)
    assert aligned["usable_for_training"]
    assert "B-TOTAL" in aligned["labels"]
    ranges = window_ranges([1] * 800)
    owners = window_owners(ranges, 800)
    assert ranges[-1][1] == 800 and len(owners) == 800
    assert all(ranges[owner][0] <= i < ranges[owner][1] for i, owner in owners.items())


def test_bio_decoder_repairs_orphan_inside_without_merging_competing_spans():
    ocr = make_ocr(["A", "B", "C"])
    result = decode_fields(ocr, ["I-COMPANY", "O", "B-COMPANY"], [0.9, 0.9, 0.7])
    assert result["company"].raw_value == "A"
    assert result["company"].alternatives == ["C"]


def test_scores_count_missing_predictions_and_parse_failures(clear_ocr):
    prediction = validate_fields(extract_rules(clear_ocr))
    entries = [
        {"receipt_id": "sample", "reference": {n: prediction.fields[n].raw_value for n in FIELDS}},
        {"receipt_id": "missing", "reference": {n: "required" for n in FIELDS}},
    ]
    result = score_predictions(entries, [prediction])
    assert result["macro_field_em"] == 0.5
    assert result["missing_predictions"] == 1
    assert not field_correct("date", "03/04/2025", "03/04/2025", "normalized-v1")
    assert not field_correct("total", None, None, "normalized-v1")
    assert field_correct("company", "Shop", "SHOP", "normalized-v1")
    assert not field_correct("company", "Shop", "SHOP", "strict-v1")


def test_review_is_append_only_and_csv_roundtrips(clear_ocr, tmp_path):
    prediction = validate_fields(extract_rules(clear_ocr))
    store = ReviewStore(tmp_path / "reviews.sqlite3")
    store.save_prediction(prediction)
    values = {n: f.raw_value for n, f in prediction.fields.items()}
    values["company"] = '=HYPERLINK("malicious")'
    values["address"] = 'A, B\n"C"'
    review = store.save_review(prediction.prediction_id, values, confirm=True)
    assert store.get_prediction(prediction.prediction_id).fields["company"].raw_value == "EXAMPLE STORE"
    assert store.get_review(review.review_id).state == "confirmed"
    exported = json.loads(json_export(prediction, review))
    assert exported["fields"]["company"].startswith("=")
    row = next(csv.DictReader(io.StringIO(csv_export(prediction, review).decode("utf-8-sig"))))
    assert row["company"].startswith("'=") and row["address"] == values["address"]
    edited = prediction.model_copy(deep=True)
    edited.fields["total"].raw_value = "0.00"
    with pytest.raises(ValueError, match="immutable"):
        store.save_prediction(edited)
    edited_review = store.save_review(prediction.prediction_id, {**values, "total": "USD1.00"})
    assert edited_review.state == "edited"
    assert store.latest_review(prediction.prediction_id).review_id == edited_review.review_id
    assert json.loads(json_export(prediction, edited_review))["currency"] == "USD"
    reverted = store.save_review(prediction.prediction_id, {**values, "company": "EXAMPLE STORE"})
    company_event = next(e for e in reverted.edits if e["field"] == "company")
    assert company_event["previous_value"] == values["company"]
    assert company_event["edited_value"] == "EXAMPLE STORE"


def test_review_database_closes_connections_and_rolls_back(tmp_path):
    import sqlite3

    store = ReviewStore(tmp_path / "resources.sqlite3")
    with store.connect() as connection:
        connection.execute("CREATE TABLE resource_check (value INTEGER)")
        connection.execute("INSERT INTO resource_check VALUES (1)")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    with pytest.raises(ValueError, match="abort"):
        with store.connect() as connection:
            connection.execute("INSERT INTO resource_check VALUES (2)")
            raise ValueError("abort")
    with store.connect() as connection:
        assert connection.execute("SELECT value FROM resource_check").fetchall() == [(1,)]


def test_pipeline_cache_keys_and_end_to_end_export(receipt_bytes, clear_ocr, tmp_path):
    class FixtureEngine:
        fingerprint = "fixture-v1"
        calls = 0

        def recognize(self, image):
            self.calls += 1
            return clear_ocr.model_copy(
                update={
                    "receipt_id": image.receipt_id,
                    "image_width": image.processed_size[0],
                    "image_height": image.processed_size[1],
                }
            )

    engine = FixtureEngine()
    first = process_receipt(receipt_bytes, engine=engine, artifact_root=tmp_path)
    second = process_receipt(receipt_bytes, engine=engine, artifact_root=tmp_path)
    assert engine.calls == 1 and second.cache_hit
    assert first.prediction.fields["total"].normalized_value == "493.50"
    assert json.loads(json_export(first.prediction))["review_status"] == "unreviewed"


def test_untrained_base_checkpoint_is_not_advertised_as_receipt_model(tmp_path):
    from receiptsense.model import LayoutExtractor

    with pytest.raises(ValueError, match="No trained"):
        LayoutExtractor(tmp_path)
