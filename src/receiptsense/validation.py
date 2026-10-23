"""Conservative field parsing. Syntax checks never choose a replacement OCR span."""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from decimal import Decimal, InvalidOperation

from .contracts import FieldPrediction, PredictionRecord


def clean_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def parse_amount(value: str | None) -> tuple[str | None, str | None]:
    if not value:
        return None, None
    text = clean_text(value)
    currency = None
    symbols = [
        (r"\b(?:MYR|RM)(?=\s*-?\d|\b)", "MYR"),
        (r"\bUSD(?=\s*-?\d|\b)", "USD"),
        (r"\bINR(?=\s*-?\d|\b)|₹", "INR"),
        (r"€|\bEUR(?=\s*-?\d|\b)", "EUR"),
        (r"£|\bGBP(?=\s*-?\d|\b)", "GBP"),
    ]
    for pattern, code in symbols:
        if re.search(pattern, text, re.I):
            if currency is not None and currency != code:
                return None, None
            currency = code
            text = re.sub(pattern, "", text, flags=re.I).strip()
    text = text.strip("$ ")  # '$' alone does not establish which dollar currency.
    if not re.fullmatch(r"-?\d[\d., ]*", text):
        return None, currency
    text = text.replace(" ", "")
    if "." in text and "," in text:
        decimal_sep = "." if text.rfind(".") > text.rfind(",") else ","
        group_sep = "," if decimal_sep == "." else "."
        integer, fraction = text.rsplit(decimal_sep, 1)
        if len(fraction) not in {1, 2} or not re.fullmatch(
            r"-?\d{1,3}(?:" + re.escape(group_sep) + r"\d{3})+", integer
        ):
            return None, currency
        text = integer.replace(group_sep, "") + "." + fraction
    elif "," in text:
        if text.count(",") == 1 and len(text.split(",")[1]) in {1, 2}:
            text = text.replace(",", ".")
        elif text.count(",") > 1 and re.fullmatch(r"-?\d{1,3}(?:,\d{3})+", text):
            text = text.replace(",", "")
        else:
            return None, currency
    elif text.count(".") > 1 or ("." in text and len(text.rsplit(".", 1)[1]) not in {1, 2}):
        return None, currency
    try:
        amount = Decimal(text)
        return format(amount.quantize(Decimal("0.01")), "f"), currency
    except (InvalidOperation, ValueError):
        return None, currency


def parse_date(value: str | None) -> tuple[str | None, list[str]]:
    if not value:
        return None, []
    text = clean_text(value).replace(",", "")
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%d.%m.%Y",
        "%m.%d.%Y",
        "%d %b %Y",
        "%d %B %Y",
        "%b %d %Y",
        "%B %d %Y",
    ]
    candidates = set()
    # Two-digit years intentionally require a user/policy decision.
    for fmt in formats:
        try:
            parsed = datetime.strptime(text, fmt)
            candidates.add(parsed.date().isoformat())
        except ValueError:
            pass
    ordered = sorted(candidates)
    return (ordered[0] if len(ordered) == 1 else None), ordered


def validate_field(name: str, field: FieldPrediction) -> FieldPrediction:
    result = field.model_copy(deep=True)
    if not result.raw_value or not result.raw_value.strip():
        result.raw_value = None
        result.normalized_value = None
        result.status = "missing"
        result.review_reasons = sorted(set(result.review_reasons + [f"missing_{name}"]))
        return result
    cleaned = clean_text(result.raw_value)
    result.normalized_value = cleaned
    result.normalization_actions = ["NFC", "whitespace"]
    if name == "total":
        result.normalized_value, result.currency = parse_amount(cleaned)
        if result.normalized_value is None:
            result.status = "invalid"
            result.review_reasons.append("amount_parse_failed")
        else:
            result.normalization_actions.append("decimal_parse")
    elif name == "date":
        result.normalized_value, alternatives = parse_date(cleaned)
        if result.normalized_value is None:
            result.status = "ambiguous" if len(alternatives) > 1 else "invalid"
            result.alternatives = alternatives or result.alternatives
            result.review_reasons.append("ambiguous_date" if len(alternatives) > 1 else "date_parse_failed")
        else:
            result.normalization_actions.append("ISO_date")
    if result.status == "missing":
        result.status = "predicted"
    result.review_reasons = sorted(set(result.review_reasons))
    return result


def validate_fields(prediction: PredictionRecord) -> PredictionRecord:
    result = prediction.model_copy(deep=True)
    result.fields = {name: validate_field(name, field) for name, field in result.fields.items()}
    return result
