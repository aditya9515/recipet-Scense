"""Describe frozen held-out errors without changing extraction or training."""

from collections import Counter

from receiptsense.alignment import matching_text
from receiptsense.common import FIELDS, ROOT, read_json, read_jsonl, utc_now, write_json
from receiptsense.evaluation import field_correct


def span_count(words, value, field):
    pieces = [matching_text(word, preserve_sign=field == "total") for word in words]
    target = matching_text(value, preserve_sign=field == "total")
    count = 0
    if not target:
        return 0
    for start in range(len(pieces)):
        joined = ""
        for end in range(start, min(start + 80, len(pieces))):
            joined += pieces[end]
            if joined == target:
                count += 1
                break
            if len(joined) > len(target) or not target.startswith(joined):
                break
    return count


def main():
    run = ROOT / "artifacts/runs/held-out-v1"
    summary = read_json(run / "summary.json")
    entries = read_jsonl(ROOT / "data/manifests/test.jsonl")
    availability, ledger = {}, []
    coverage = {field: Counter() for field in FIELDS}
    for entry in entries:
        record = read_json(ROOT / "data/processed/records/identity" / f"{entry['receipt_id']}.json")
        words = [unit["text"] for unit in sorted(record["ocr"]["units"], key=lambda u: u["reading_order"])]
        availability[entry["receipt_id"]] = {}
        for field in FIELDS:
            count = span_count(words, entry["reference"][field], field)
            availability[entry["receipt_id"]][field] = count
            coverage[field]["none" if count == 0 else "one" if count == 1 else "multiple"] += 1
    buckets = {}
    for method in summary["results"]:
        buckets[method] = {field: Counter() for field in FIELDS}
        metrics = read_json(run / f"{method}_strict-v1.json")
        for row in metrics["rows"]:
            for field, correct in row["correct"].items():
                if correct:
                    continue
                predicted, reference = row["predicted"][field], row["reference"][field]
                spans = availability[row["receipt_id"]][field]
                if row["missing_prediction"]:
                    bucket = "missing_prediction"
                elif field_correct(field, predicted, reference, "normalized-v1"):
                    bucket = "normalization_resolves"
                elif predicted is None:
                    bucket = "missing_field"
                elif spans:
                    bucket = "reference_span_available"
                else:
                    bucket = "reference_span_not_found"
                buckets[method][field][bucket] += 1
                ledger.append(
                    {
                        "receipt_id": row["receipt_id"],
                        "source_key": row["source_key"],
                        "method": method,
                        "field": field,
                        "reference": reference,
                        "predicted": predicted,
                        "reference_span_count": spans,
                        "diagnostic_bucket": bucket,
                    }
                )
    evidence = {
        "created_at": utc_now(),
        "run": "held-out-v1",
        "n_receipts": len(entries),
        "policy": "post-hoc diagnostics only; no model or rule updates",
        "span_policy": "contiguous whole OCR units, up to 80; NFKC/casefold/punctuation removal, amount sign retained",
        "limitations": "String availability is not semantic correctness or a measured OCR CER/WER. Missing spans can reflect OCR errors, punctuation/annotation differences, granularity or reading order. Buckets do not establish causal responsibility.",
        "reference_span_coverage": {field: dict(c) for field, c in coverage.items()},
        "strict_error_buckets": {
            method: {field: dict(c) for field, c in fields.items()} for method, fields in buckets.items()
        },
        "ledger": ledger,
    }
    write_json(ROOT / "artifacts/error_analysis.json", evidence)
    lines = [
        "# ReceiptSense - frozen held-out error analysis",
        "",
        f"Run `held-out-v1`: {len(entries)} receipts, both methods using the same OCR. This analysis is post-hoc; it does not change the evaluated checkpoint, rules, normalization or thresholds.",
        "",
        "## Reference text availability in OCR",
        "",
        "The diagnostic searches contiguous whole OCR units after NFKC/case/punctuation cleanup (amount signs retained). It allows up to 80 units. A found span establishes string availability, not that the region is semantically the desired field. Multiple equal amounts can be total, cash or change.",
        "",
        "| Field | No matching span | One span | Multiple spans |",
        "|---|---:|---:|---:|",
    ]
    for field, counts in coverage.items():
        lines.append(f"| {field} | {counts['none']} | {counts['one']} | {counts['multiple']} |")
    lines += [
        "",
        "Missing spans can result from recognition errors, transcript formatting, word boundaries, reading order or reference differences. This table is not an OCR character/word error rate and cannot separate detection from recognition failures.",
        "",
        "## Strict field errors",
        "",
        "Each strict error enters one category in this order: missing complete prediction; resolved by the declared normalized policy; missing field; reference span available; reference span not found. The categories are diagnostics rather than proven causes.",
        "",
        "| Method | Field | Missing prediction | Normalization resolves | Missing field | Reference available | Reference not found | Total strict errors |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    names = [
        "missing_prediction",
        "normalization_resolves",
        "missing_field",
        "reference_span_available",
        "reference_span_not_found",
    ]
    for method, fields in buckets.items():
        for field, counts in fields.items():
            cells = [method, field] + [str(counts[name]) for name in names] + [str(sum(counts.values()))]
            lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## How to use these findings",
        "",
        "Use train/validation receipts to investigate label coverage, OCR geometry, competing spans and multi-line address boundaries. Richer verified labels and full-backbone adaptation are possible future experiments. Keep the held-out results above frozen; any changes motivated by these test errors need a new untouched confirmation set or a disclosure that the old set became development data.",
        "",
        "[Full metrics and report](IMPLEMENTATION_REPORT.md) · [Complete error ledger](../results/artifacts/error_analysis.json)",
        "",
        "OCR CER/WER, detection-versus-recognition attribution, fully manual causal error categories, alternative OCR benchmarks and multi-seed training were not measured in this build.",
        "",
    ]
    (ROOT / "report/ERROR_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(ledger)} field-error rows and report/ERROR_ANALYSIS.md")


if __name__ == "__main__":
    main()
