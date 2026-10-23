"""BIO predictions reconstructed into source-bearing receipt fields."""

from __future__ import annotations

from .common import FIELDS
from .contracts import FieldPrediction, OCRRecord


def decode_fields(ocr: OCRRecord, labels: list[str], scores: list[float]) -> dict[str, FieldPrediction]:
    if not (len(labels) == len(scores) == len(ocr.units)):
        raise ValueError("Unit predictions, scores and OCR lengths differ")
    spans: dict[str, list[list[int]]] = {field: [] for field in FIELDS}
    active_field = None
    for i, label in enumerate(labels):
        if label == "O":
            active_field = None
            continue
        prefix, name = label.split("-", 1)
        field = name.lower()
        if field not in spans or prefix not in {"B", "I"}:
            raise ValueError(f"Unsupported model label: {label}")
        if prefix == "B" or active_field != field:
            spans[field].append([])
        spans[field][-1].append(i)
        active_field = field
    fields = {}
    for field, candidates in spans.items():
        if not candidates:
            fields[field] = FieldPrediction(review_reasons=[f"missing_{field}"])
            continue
        candidates.sort(key=lambda span: (-sum(scores[i] for i in span) / len(span), span[0]))
        selected = candidates[0]

        def join(span, field=field):
            result = ""
            previous = None
            for i in span:
                unit = ocr.units[i]
                separator = (
                    "\n" if field == "address" and previous and previous.line_id != unit.line_id else " "
                )
                result += (separator if result else "") + unit.text
                previous = unit
            return result

        score = sum(scores[i] for i in selected) / len(selected)
        reasons = ["low_model_margin"] if score < 0.6 else []
        if len(candidates) > 1:
            reasons.append(f"multiple_{field}_candidates")
        fields[field] = FieldPrediction(
            raw_value=join(selected),
            status="ambiguous" if len(candidates) > 1 else "predicted",
            evidence_unit_ids=[ocr.units[i].unit_id for i in selected],
            score=score,
            score_kind="uncalibrated_first_subword_mean",
            review_reasons=reasons,
            alternatives=[join(c) for c in candidates[1:4]],
        )
    return fields
