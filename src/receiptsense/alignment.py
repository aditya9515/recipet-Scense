"""Conservative weak-label construction with ambiguity masking and provenance."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

from .common import FIELDS, ROOT, read_json, read_jsonl, write_json, write_jsonl
from .contracts import OCRRecord
from .reading_order import grouped_lines


def matching_text(value: str, preserve_sign: bool = False) -> str:
    pattern = r"[^\w-]" if preserve_sign else r"[^\w]"
    return re.sub(pattern, "", unicodedata.normalize("NFKC", value).casefold())


def align_fields(ocr: OCRRecord, reference: dict[str, str]) -> dict:
    units = sorted(ocr.units, key=lambda u: u.reading_order)
    labels = ["IGNORE"] * len(units)
    states = {}
    candidates_by_field = {}
    lines = grouped_lines(units)
    line_context = {line[0].line_id: " ".join(u.text for u in line).upper() for line in lines}
    # Ground truth is read only for train/validation label construction.
    for field in FIELDS:
        pieces = [matching_text(u.text, preserve_sign=field == "total") for u in units]
        target = matching_text(reference.get(field, ""), preserve_sign=field == "total")
        candidates = []
        if target:
            for start in range(len(units)):
                joined = ""
                for end in range(start, min(len(units), start + 80)):
                    joined += pieces[end]
                    if joined == target:
                        candidates.append(list(range(start, end + 1)))
                        break
                    if len(joined) > len(target) or not target.startswith(joined):
                        break
        candidates_by_field[field] = candidates
        if field == "total":
            # The same reference amount can occur as cash/change. Unique string
            # matching alone is not semantic evidence for the TOTAL label.
            candidates = [
                span
                for span in candidates
                if not any(
                    re.search(
                        r"\b(?:CASH|CHANGE|TENDERED|TENDER|SUBTOTAL|SUB\s+TOTAL)\b",
                        line_context[units[i].line_id],
                    )
                    and not re.search(
                        r"\b(?:GRAND TOTAL|TOTAL PAYABLE|NETT? TOTAL)\b", line_context[units[i].line_id]
                    )
                    for i in span
                )
            ]
            candidates_by_field[field] = candidates
        if len(candidates) == 1:
            span = candidates[0]
            states[field] = {"status": "exact_match", "unit_ids": [units[i].unit_id for i in span]}
        else:
            states[field] = {
                "status": "ambiguous" if candidates else "unmatched",
                "unit_ids": [],
                "candidate_count": len(candidates),
            }
    # Unknown positives may be anywhere. On partially aligned receipts, supervise
    # only verified-positive spans and ignore all other units. O is justified
    # only when all four reference fields have an unambiguous located span.
    fully_aligned = all(states[f]["status"] == "exact_match" for f in FIELDS)
    if fully_aligned:
        labels = ["O"] * len(units)
    usable = any(states[f]["status"] == "exact_match" for f in FIELDS)
    occupied = set()
    for field in FIELDS:
        spans = candidates_by_field[field]
        if len(spans) == 1:
            for offset, index in enumerate(spans[0]):
                if index in occupied:
                    usable = False
                occupied.add(index)
                labels[index] = ("B-" if offset == 0 else "I-") + field.upper()
    return {
        "receipt_id": ocr.receipt_id,
        "unit_ids": [u.unit_id for u in units],
        "labels": labels,
        "usable_for_training": usable,
        "fully_aligned": fully_aligned,
        "fields": states,
        "policy": "exact-span-partial-supervision-v2",
        "reference": reference,
        "label_provenance": "exact_match_weak_labels_not_manually_verified",
    }


def build_alignments(
    manifest: str | Path,
    records: str | Path = ROOT / "data/processed/records/identity",
    output: str | Path | None = None,
) -> dict:
    entries = read_jsonl(manifest)
    if any(e.get("split") == "test" for e in entries):
        raise ValueError("Training alignment is restricted to train/validation manifests")
    output = Path(output or ROOT / "data/processed" / f"labels_{Path(manifest).stem}.jsonl")
    rows, coverage, missing = [], {field: Counter() for field in FIELDS}, []
    for entry in entries:
        path = Path(records) / f"{entry['receipt_id']}.json"
        if not path.exists():
            missing.append(entry["receipt_id"])
            continue
        record = read_json(path)
        aligned = align_fields(OCRRecord.model_validate(record["ocr"]), entry["reference"])
        aligned["record_path"] = str(path.resolve())
        aligned["split"] = entry["split"]
        rows.append(aligned)
        for field in FIELDS:
            coverage[field][aligned["fields"][field]["status"]] += 1
    write_jsonl(output, rows)
    summary = {
        "receipts": len(rows),
        "usable_receipts": sum(r["usable_for_training"] for r in rows),
        "fully_aligned_receipts": sum(r["fully_aligned"] for r in rows),
        "per_field": {f: dict(c) for f, c in coverage.items()},
        "missing_ocr": missing,
        "policy": "exact-span-partial-supervision-v2",
        "manual_audit": "required; exact string alignment is not semantic proof",
    }
    write_json(output.with_suffix(".summary.json"), summary)
    return summary
