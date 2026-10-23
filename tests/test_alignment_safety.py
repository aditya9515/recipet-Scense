from conftest import make_ocr

from receiptsense.alignment import align_fields


def test_unique_cash_match_is_not_total_supervision():
    ocr = make_ocr(["EXAMPLE STORE", "12 MAIN ROAD", "2025-02-14", "TOTAL:9.00", "CASH 9.00"])
    labels = align_fields(
        ocr, {"company": "EXAMPLE STORE", "address": "12 MAIN ROAD", "date": "2025-02-14", "total": "9.00"}
    )
    assert labels["fields"]["total"]["status"] == "unmatched"
    assert "B-TOTAL" not in labels["labels"]


def test_negative_reference_does_not_label_positive_amount():
    ocr = make_ocr(["EXAMPLE STORE", "12 MAIN ROAD", "2025-02-14", "TOTAL 1.73"])
    labels = align_fields(
        ocr, {"company": "EXAMPLE STORE", "address": "12 MAIN ROAD", "date": "2025-02-14", "total": "-1.73"}
    )
    assert labels["fields"]["total"]["status"] == "unmatched"
