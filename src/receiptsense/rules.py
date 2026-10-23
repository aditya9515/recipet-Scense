"""Deterministic, evidence-bearing field candidates for a transparent baseline."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .common import FIELDS, digest
from .contracts import FieldPrediction, OCRRecord, OCRUnit, PredictionRecord
from .reading_order import grouped_lines
from .validation import parse_amount

RULES_VERSION = "rules-v1"
DATE_PATTERN = re.compile(
    r"\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b"
)
AMOUNT_PATTERN = re.compile(r"(?<![\w/])(?:RM\s*|MYR\s*|USD\s*|INR\s*|[$€£₹]\s*)?-?\d[\d,.]*(?![\w/])", re.I)


@dataclass
class Candidate:
    value: str
    units: list[OCRUnit]
    score: float
    reason: str


def text_of(units: list[OCRUnit]) -> str:
    return " ".join(u.text for u in units)


def span_units(line: list[OCRUnit], start: int, end: int) -> list[OCRUnit]:
    cursor = 0
    selected = []
    for unit in line:
        if cursor < end and cursor + len(unit.text) > start:
            selected.append(unit)
        cursor += len(unit.text) + 1
    return selected


def prediction_field(candidates: list[Candidate], name: str) -> FieldPrediction:
    if not candidates:
        return FieldPrediction(review_reasons=[f"missing_{name}"])
    candidates = sorted(candidates, key=lambda c: (-c.score, c.value, [u.unit_id for u in c.units]))
    first = candidates[0]
    alternatives = list(dict.fromkeys(c.value for c in candidates[1:] if c.value != first.value))[:4]
    ambiguous = bool(
        alternatives
        and any(c.value != first.value and abs(c.score - first.score) < 1 for c in candidates[1:])
    )
    warnings = (
        ["multiple_total_candidates" if name == "total" else f"multiple_{name}_candidates"]
        if ambiguous
        else []
    )
    if any(u.recognition_score is not None and u.recognition_score < 0.65 for u in first.units):
        warnings.append("low_ocr_evidence")
    if any(u.box_source == "replicated_line" for u in first.units):
        # Geometry precision is displayed separately; coarse boxes alone need not flag every receipt.
        pass
    return FieldPrediction(
        raw_value=first.value,
        evidence_unit_ids=[u.unit_id for u in first.units],
        status="ambiguous" if ambiguous else "predicted",
        score=first.score,
        score_kind="heuristic",
        alternatives=alternatives,
        review_reasons=warnings,
    )


def extract_rules(ocr: OCRRecord) -> PredictionRecord:
    lines = grouped_lines(ocr.units)
    candidates: dict[str, list[Candidate]] = {name: [] for name in FIELDS}
    texts = [text_of(line) for line in lines]
    company_index = None
    best_company = -float("inf")
    for i, (line, text) in enumerate(zip(lines, texts, strict=True)):
        upper = text.upper()
        # Header-only clues; never allow payment rows to become a merchant.
        if i < max(7, len(lines) // 4) and any(ch.isalpha() for ch in text):
            score = 6 - i * 0.7
            if re.search(
                r"\b(SDN|BHD|LTD|LIMITED|STORE|MART|TRADING|RESTAURANT|ENTERPRISE|PHARMACY)\b", upper
            ):
                score += 4
            if re.search(
                r"\b(RECEIPT|INVOICE|TEL|PHONE|FAX|GST|CASHIER|DATE|CUSTOMER|NO\.?|JALAN|ROAD|STREET)\b",
                upper,
            ):
                score -= 7
            if sum(ch.isdigit() for ch in text) > len(text) * 0.25:
                score -= 4
            if score > 0:
                candidates["company"].append(Candidate(text, line, score, "header_name"))
                if score > best_company:
                    company_index, best_company = i, score
        for match in DATE_PATTERN.finditer(text):
            selected = span_units(line, match.start(), match.end())
            score = 3.0 + (4 if re.search(r"\b(DATE|TARIKH)\b", upper) else 0) - i * 0.01
            candidates["date"].append(Candidate(match.group(), selected, score, "date_pattern"))
        for match in AMOUNT_PATTERN.finditer(text):
            value, _ = parse_amount(match.group())
            if value is None:
                continue
            # Integer identifiers and item quantities require explicit final-total context.
            total_label = bool(
                re.search(
                    r"\b(?:GRAND\s+TOTAL|TOTAL(?:\s+(?:AMOUNT|DUE|SALES|PAYABLE))?|AMOUNT\s+DUE|NET\s+AMOUNT)\b",
                    upper,
                )
            )
            penalty = bool(
                re.search(
                    r"\b(?:SUB\s*TOTAL|SUBTOTAL|CASH|CHANGE|TENDERED|TAX|GST|SST|ROUNDING|DISCOUNT)\b", upper
                )
            )
            score = 1 + 2 * i / max(1, len(lines) - 1)
            if total_label:
                score += 10
            if penalty:
                score -= 12
            if not re.search(r"[.,]\d{1,2}\b", match.group()) and not total_label:
                score -= 6
            if (
                i
                and not total_label
                and re.search(r"\b(?:TOTAL|AMOUNT DUE)\b", texts[i - 1].upper())
                and not re.search(r"SUB\s*TOTAL", texts[i - 1].upper())
            ):
                score += 7
            if score > 0:
                candidates["total"].append(
                    Candidate(
                        match.group().strip(),
                        span_units(line, match.start(), match.end()),
                        score,
                        "amount_context",
                    )
                )
    if company_index is not None:
        address_lines = []
        for j in range(company_index + 1, min(len(lines), company_index + 7)):
            text = texts[j]
            if re.search(
                r"\b(?:TEL|PHONE|FAX|GST|SST|INVOICE|RECEIPT|DATE|CASHIER|REG\b|TAX|WWW|EMAIL)\b|@",
                text,
                re.I,
            ):
                break
            if DATE_PATTERN.search(text) or not any(ch.isalpha() for ch in text):
                break
            address_lines.append(lines[j])
        if address_lines:
            flattened = [u for line in address_lines for u in line]
            value = "\n".join(text_of(line) for line in address_lines)
            score = 4 + (
                2
                if re.search(r"\b(?:ROAD|JALAN|STREET|TAMAN|NO\b|LOT|FLOOR|LORONG|PERSIARAN)\b", value, re.I)
                else 0
            )
            candidates["address"].append(Candidate(value, flattened, score, "consecutive_header_address"))
    prediction = PredictionRecord(
        receipt_id=ocr.receipt_id,
        method="rules",
        fields={name: prediction_field(candidates[name], name) for name in FIELDS},
        ocr_fingerprint=ocr.ocr_fingerprint,
        model_revision=RULES_VERSION,
        configuration_hash=digest({"rules": RULES_VERSION, "ocr": ocr.ocr_fingerprint}),
    )
    prediction.check_evidence(ocr)
    return prediction
