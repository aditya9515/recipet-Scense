"""Geometry graph for evidence-bearing merchant address blocks."""

import re
from dataclasses import dataclass
from statistics import median

from .contracts import FieldPrediction, OCRRecord, OCRUnit
from .reading_order import grouped_lines
from .recovery_config import RecoveryConfig
from .validation import clean_text

STREET = re.compile(
    r"\b(?:ROAD|RD|STREET|ST|LANE|NAGAR|CHOWK|MARG|COLONY|SECTOR|JALAN|JLN|TAMAN|LOT|LORONG|PERSIARAN|FLOOR|BUILDING|COMPLEX|LAYOUT|AVENUE|BLOCK|BAZAR|BAZAAR)\b",
    re.I,
)
POSTAL = re.compile(r"(?<!\d)\d{5,6}(?!\d)")
NUMBER = re.compile(r"^\s*(?:NO\.?\s*)?\d[\w/,-]*\s*[,#/-]?\s*[A-Za-z]", re.I)
STOP = re.compile(
    r"\b(?:TEL|TELEPHONE|PHONE|MOBILE|FAX|GSTIN|GST|SST|TIN|CIN|REGISTRATION|REG|INVOICE|RECEIPT|BILL|DATE|CASHIER|CUSTOMER|QTY|QUANTITY|DESCRIPTION|ITEM|TOTAL|SUBTOTAL|SUB\s+TOTAL|CASH|CHANGE|TENDER|TAX|WWW|EMAIL)\b|@|https?://",
    re.I,
)
MERCHANT = re.compile(r"\b(?:SDN|BHD|LTD|LIMITED|STORE|RESTAURANT|ENTERPRISE|TRADING)\b", re.I)
PLACE = re.compile(
    r"\b(?:INDIA|DELHI|MUMBAI|BENGALURU|BANGALORE|HYDERABAD|CHENNAI|PUNE|MANIPAL|UDUPI|KARNATAKA|SELANGOR|PERAK|MALAYSIA|KUALA|JOHOR|PENANG)\b",
    re.I,
)


@dataclass
class Line:
    units: list[OCRUnit]
    text: str
    box: tuple[float, float, float, float]


def line_nodes(ocr: OCRRecord) -> list[Line]:
    """Split distant columns while retaining the OCR engine's coarse boxes."""
    nodes = []
    for line in grouped_lines(ocr.units):
        groups: list[list[OCRUnit]] = [[]]
        for unit in sorted(line, key=lambda u: (u.bbox_xyxy[0], u.reading_order)):
            if groups[-1]:
                previous = groups[-1][-1]
                height = max(1, previous.bbox_xyxy[3] - previous.bbox_xyxy[1])
                if unit.bbox_xyxy[0] - previous.bbox_xyxy[2] > max(40, 2.5 * height):
                    groups.append([])
            groups[-1].append(unit)
        for units in groups:
            if units:
                box = (
                    min(u.bbox_xyxy[0] for u in units),
                    min(u.bbox_xyxy[1] for u in units),
                    max(u.bbox_xyxy[2] for u in units),
                    max(u.bbox_xyxy[3] for u in units),
                )
                nodes.append(Line(units, " ".join(u.text for u in units), box))
    return sorted(nodes, key=lambda n: (n.box[1], n.box[0], n.text))


def blocked(text: str) -> bool:
    return bool(
        STOP.search(text)
        or re.search(
            r"(?:^|\s)(?:DATE|TARIKH|TEL|PHONE|GSTIN|INVOICE|RECEIPT|TOTAL|SUBTOTAL)(?=\d|[:#])", text, re.I
        )
        or re.search(r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b", text)
        or re.search(r"(?<!\w)\d+[.,]\d{2}\b", text)
    )


def address_score(value: str | None, y_fraction: float = 0) -> float:
    if not value or blocked(value):
        return -100
    lines = value.splitlines()
    return (
        3 * bool(STREET.search(value))
        + 3 * bool(POSTAL.search(value))
        + 2 * bool(NUMBER.search(lines[0]))
        + bool(PLACE.search(value))
        + 0.75 * min(4, len(lines))
        - y_fraction
        - 4 * bool(MERCHANT.search(value))
    )


def reconstruct_address(ocr: OCRRecord, original: FieldPrediction, config: RecoveryConfig):
    nodes = line_nodes(ocr)[:120]
    height = median([n.box[3] - n.box[1] for n in nodes]) if nodes else 1
    header = config.header_fraction * ocr.image_height
    valid = [i for i, n in enumerate(nodes) if n.box[1] < header and not blocked(n.text)]

    def connected(a, b):
        x1, y1, x2, y2 = a.box
        u1, v1, u2, v2 = b.box
        if (v1 + v2 - y1 - y2) / 2 < height * 0.3 or v1 - y2 > height * config.address_gap_heights:
            return False
        overlap = max(0, min(x2, u2) - max(x1, u1)) / max(1, min(x2 - x1, u2 - u1))
        if overlap < config.address_overlap and abs(x1 - u1) > height * 1.5:
            return False
        # A contact/payment row is a barrier, not a line to skip across.
        return not any(
            blocked(n.text) and y2 <= n.box[1] < v1 and min(x2, n.box[2]) > max(x1, n.box[0]) for n in nodes
        )

    edges = {i: [j for j in valid if j > i and connected(nodes[i], nodes[j])] for i in valid}
    candidates = {}
    visited = 0

    def visit(path):
        nonlocal visited
        visited += 1
        if visited > 4096:
            return
        text = "\n".join(nodes[i].text for i in path)
        score = address_score(text, nodes[path[0]].box[1] / ocr.image_height)
        if score >= config.address_min_score:
            key = clean_text(text).casefold()
            candidates[key] = (score, text, [u.unit_id for i in path for u in nodes[i].units])
        if len(path) < config.address_max_lines:
            # Keep search bounded and follow immediate neighbors, not arbitrary skips.
            following = edges[path[-1]]
            if following:
                first_y = min(nodes[j].box[1] for j in following)
                for j in following:
                    if nodes[j].box[1] <= first_y + 0.3 * height:
                        visit(path + [j])

    for i in valid:
        if STREET.search(nodes[i].text) or NUMBER.search(nodes[i].text) or POSTAL.search(nodes[i].text):
            visit([i])
    ranked = sorted(candidates.values(), key=lambda c: (-c[0], -len(c[2]), c[1]))
    decision = {
        "before": original.raw_value,
        "after": original.raw_value,
        "outcome": "unchanged",
        "reason": "no_supported_address_block",
        "candidates": [],
    }
    if not ranked:
        return original.model_copy(deep=True), decision
    best_score, value, ids = ranked[0]
    alternatives = [
        c
        for c in ranked[1:]
        if clean_text(c[1]) not in clean_text(value) and clean_text(value) not in clean_text(c[1])
    ]
    decision["candidates"] = [
        {"value": c[1], "score": c[0], "evidence_unit_ids": c[2]} for c in [ranked[0], *alternatives[:3]]
    ]
    ambiguous = bool(alternatives and best_score - alternatives[0][0] < config.address_margin)
    result = original.model_copy(deep=True)
    if clean_text(value) == clean_text(original.raw_value or ""):
        decision["reason"] = "same_address"
    elif ambiguous:
        result.alternatives = list(
            dict.fromkeys([*result.alternatives, value, *[c[1] for c in alternatives[:2]]])
        )[:4]
        result.review_reasons = sorted(set(result.review_reasons + ["address_graph_conflict"]))
        decision.update(outcome="rejected", reason="competing_address_blocks")
    elif (
        original.raw_value
        and original.status == "predicted"
        and best_score < address_score(original.raw_value) + config.address_gain
    ):
        decision.update(outcome="rejected", reason="insufficient_structural_gain")
    else:
        result = FieldPrediction(
            raw_value=value,
            evidence_unit_ids=ids,
            status="predicted",
            score=best_score,
            score_kind="address_graph_heuristic",
            alternatives=[c[1] for c in alternatives[:3]],
        )
        decision.update(after=value, outcome="accepted", reason="supported_address_block")
    return result, decision
