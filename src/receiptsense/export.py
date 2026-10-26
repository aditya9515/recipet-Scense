"""JSON keeps original text; spreadsheet CSV uses explicit text-cell escaping."""

from __future__ import annotations

import csv
import io
import json

from .common import FIELDS
from .contracts import PredictionRecord, ReviewedRecord
from .validation import parse_amount


def json_export(prediction: PredictionRecord, review: ReviewedRecord | None = None) -> bytes:
    if review and review.prediction_id != prediction.prediction_id:
        raise ValueError("Review does not belong to this prediction")
    payload = {
        "schema_version": "1.0",
        "receipt_id": prediction.receipt_id,
        "review_status": review.state if review else "unreviewed",
        "fields": review.values if review else {n: f.raw_value for n, f in prediction.fields.items()},
        "normalized_fields": review.normalized_values
        if review
        else {n: f.normalized_value for n, f in prediction.fields.items()},
        "currency": parse_amount(review.values["total"])[1]
        if review
        else prediction.fields["total"].currency,
        "prediction": prediction.model_dump(mode="json"),
        "review": review.model_dump(mode="json") if review else None,
    }
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def csv_safe(value: str | None) -> str:
    if value is None:
        return ""
    return "'" + value if value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) else value


def csv_export(prediction: PredictionRecord, review: ReviewedRecord | None = None) -> bytes:
    if review and review.prediction_id != prediction.prediction_id:
        raise ValueError("Review does not belong to this prediction")
    values = review.values if review else {name: prediction.fields[name].raw_value for name in FIELDS}
    normalized = (
        review.normalized_values
        if review
        else {name: prediction.fields[name].normalized_value for name in FIELDS}
    )
    row = {
        "receipt_id": prediction.receipt_id,
        "prediction_id": prediction.prediction_id,
        "review_id": review.review_id if review else "",
        "company": csv_safe(values["company"]),
        "address": csv_safe(values["address"]),
        "date_raw": csv_safe(values["date"]),
        "date_normalized": normalized["date"] or "",
        "total_raw": csv_safe(values["total"]),
        "total_normalized": normalized["total"] or "",
        "currency": (parse_amount(values["total"])[1] if review else prediction.fields["total"].currency)
        or "",
        "review_status": review.state if review else "unreviewed",
        "extraction_method": prediction.method,
        "processing_mode": prediction.recovery.mode if prediction.recovery else "none",
        "logical_retry_attempts": prediction.recovery.logical_attempts if prediction.recovery else 0,
        "uncached_retry_ocr_calls": prediction.recovery.uncached_ocr_calls if prediction.recovery else 0,
    }
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
    return stream.getvalue().encode("utf-8-sig")
