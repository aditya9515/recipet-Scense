"""Publish aggregate validation evidence only; never copy private confirmation data."""

import shutil
import xml.etree.ElementTree as ET

from receiptsense.common import ROOT, read_json, write_json
from receiptsense.confirmation import protocol_fingerprint
from receiptsense.novelty_evaluation import policy_fingerprint
from receiptsense.recovery_config import load_config


def pct(value):
    return f"{value:.2%}"


def main():
    run = ROOT / "artifacts/novelty/validation-v1"
    summary, latency = read_json(run / "summary.json"), read_json(run / "latency.json")
    if not summary["complete"] or summary["n_receipts"] != 61 or len(summary["results"]) != 8:
        raise ValueError("Need the full eight-condition validation comparison")
    if (
        summary["configuration_hash"] != load_config().fingerprint
        or summary["policy_fingerprint"] != policy_fingerprint()
        or latency["policy_fingerprint"] != policy_fingerprint()
    ):
        raise ValueError("Policy changed after the saved experiments")
    suites = list(ET.parse(ROOT / "artifacts/novelty/test-results.xml").getroot().iter("testsuite"))
    counts = {
        k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")
    }
    if counts["failures"] or counts["errors"]:
        raise ValueError("Verification contains failed tests")
    invariants = read_json(ROOT / "results/novelty/invariants.json")
    if not invariants.get("passed"):
        raise ValueError("Saved prediction and baseline integrity checks are required")
    destination = ROOT / "results/novelty/validation-v1"
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("summary.json", "latency.json"):
        shutil.copyfile(run / filename, destination / filename)
    shutil.copyfile(ROOT / "artifacts/novelty/test-results.xml", ROOT / "results/novelty/test-results.xml")
    workflow = ROOT / "artifacts/novelty/workflow_check.json"
    if not workflow.exists() or not read_json(workflow).get("passed"):
        raise ValueError("Real enhanced API workflow check is required")
    shutil.copyfile(workflow, ROOT / "results/novelty/workflow_check.json")
    freeze = {
        "configuration_hash": summary["configuration_hash"],
        "policy_fingerprint": summary["policy_fingerprint"],
        "checkpoint": summary["checkpoint"],
        "validation_run": "validation-v1",
        "default_mode": "none",
        "base_ocr_fingerprint": read_json(ROOT / "results/novelty/baseline_manifest.json")["ocr_fingerprint"],
        "evaluation_protocol_fingerprint": protocol_fingerprint(),
        "confirmation_status": "awaiting 50 newly collected, double-annotated receipts; not evaluated",
    }
    write_json(ROOT / "results/novelty/freeze.json", freeze)
    lines = [
        "# ReceiptSense - novelty implementation and validation",
        "",
        "This branch adds a multi-line address graph and selective crop OCR recovery, capped at two additional OCR calls. Both extractors use the same fixed checkpoint/OCR assets. No new backbone fine-tuning or active learning was performed.",
        "",
        "[Roadmap and design](../docs/14_NOVELTY_ROADMAP.md) · [Commands](../RUNNING.md) · [Original measured report](IMPLEMENTATION_REPORT.md)",
        "",
        "## Validation comparison",
        "",
        "These are development results on 61 validation receipts, including the data used to select graph thresholds. They are not independent confirmation or a new held-out claim.",
        "",
        "| Extractor / mode | Strict macro EM | Strict address EM | All-four EM | Address token F1 | Corrected fields | Introduced errors | Retry rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, values in summary["results"].items():
        strict = values["strict-v1"]
        lines.append(
            f"| {condition} | {pct(strict['macro_field_em'])} | {pct(strict['per_field']['address']['exact_match'])} | {pct(strict['all_four_em'])} | {pct(values['address_token_f1'])} | {sum(values['corrected_fields'].values())} | {sum(values['introduced_errors'].values())} | {pct(values['retry_rate'])} |"
        )
    lines += [
        "",
        "Full strict/normalized per-field metrics, correction/error counts, review rates and paired receipt-bootstrap intervals are in [the aggregate summary](../results/novelty/validation-v1/summary.json). Missing predictions remain failures. Merchant-grouped intervals are unavailable for SROIE validation because no verified merchant grouping is provided; the fresh pilot will supply it.",
        "",
        f"Combined default eligibility on validation: {summary['combined_default_gate']}. Baseline remains the default; these development checks do not establish reliable automatic acceptance.",
        "",
        "## Actual uncached CPU timing",
        "",
        f"{latency['n_receipts']} validation receipts, one warmup per condition. {latency['scope']}. Condition order rotates between receipts. Small-sample p95 is unstable.",
        "",
        "| Extractor / mode | Median seconds | p95 seconds | Mean total OCR calls |",
        "|---|---:|---:|---:|",
    ]
    for condition, values in latency["results"].items():
        lines.append(
            f"| {condition} | {values['median_seconds']:.3f} | {values['p95_seconds']:.3f} | {values['mean_ocr_calls']:.2f} |"
        )
    lines += [
        "",
        "[Timing rows](../results/novelty/validation-v1/latency.json) separate actual uncached engine calls from logical retry attempts. Accuracy-run timings permit caches and must not be substituted for this timing study.",
        "",
        "## Verification and limitations",
        "",
        "The frozen checkpoint and all 488 saved predictions were audited for evidence resolution, stable fingerprints and the two-attempt limit. Both baseline modes reproduce all four raw fields on all 61 validation receipts from their original saved runs. [Integrity evidence](../results/novelty/invariants.json).",
        "",
        f"Automated suite: {counts['tests']} tests, {counts['failures']} failures, {counts['errors']} errors, {counts['skipped']} skipped. [Test evidence](../results/novelty/test-results.xml). The [real CPU API workflow](../results/novelty/workflow_check.json) covers enhanced extraction, evidence, correction, immutable originals and JSON/CSV export using explicitly synthetic examples.",
        "",
        "The address graph and retry ranking are heuristics, and OCR geometry may be line-derived. Re-reading can introduce errors; those are counted above. No formal risk calibration, broad multilingual support or new full fine-tuning result is claimed. The contribution is this particular evidence-preserving, budgeted recovery strategy and its measured trade-off, not the invention of OCR or uncertainty-based review.",
        "",
        "## Fresh confirmation remains pending",
        "",
        "The team must collect 50 English/Latin-script Indian receipts across ten merchants (five distinct transactions each), independently double-label the four fields, resolve disagreements and freeze the set before inference. No real fresh confirmation receipts have been supplied, and no confirmation accuracy is reported. The collection/freeze/evaluation tooling is implemented and its integrity checks are tested with synthetic fixtures only.",
        "",
        "Private images, merchant mapping, references, prediction details and locks remain outside Git. Only aggregate confirmation metrics and explicitly approved redacted examples may be published. The original fourteen screenshot areas remain blank.",
        "",
        "[Frozen policy identity](../results/novelty/freeze.json) · [Validation selection](../results/novelty/policy_selection.json) · [Original baseline identity](../results/novelty/baseline_manifest.json)",
        "",
    ]
    (ROOT / "report/NOVELTY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("Published aggregate novelty evidence and report; confirmation is explicitly pending")


if __name__ == "__main__":
    main()
