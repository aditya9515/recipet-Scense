"""Generate the implementation report from saved evidence, without inventing results."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from receiptsense.common import FIELDS, ROOT, read_json


def percent(value):
    return "undefined" if value is None else f"{value * 100:.2f}%"


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        + ["| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in rows]
    )


def main():
    card = read_json(ROOT / "data/dataset_card.json")
    training = read_json(ROOT / "artifacts/models/receiptsense-layoutlmv3/receiptsense_model.json")
    labels = read_json(ROOT / "data/processed/labels_train.summary.json")
    result = read_json(ROOT / "artifacts/runs/held-out-v1/summary.json")
    robust = read_json(ROOT / "artifacts/robustness/robustness-v1/summary.json")
    latency = read_json(ROOT / "artifacts/benchmark.json")
    audit = read_json(ROOT / "data/near_duplicate_audit.json")
    reload_check = read_json(ROOT / "artifacts/checkpoint_reload_check.json")
    environment = read_json(ROOT / "artifacts/environment.json")
    if result["n_receipts"] != card["counts"]["test"] or result["partial_subset"]:
        raise ValueError("The report requires the full held-out comparison")
    if set(robust["results"]) != {"clean", "deskew", "blur", "rotation", "contrast", "perspective"}:
        raise ValueError("The robustness probe has not completed every condition")
    tests = ET.parse(ROOT / "artifacts/test-results.xml").getroot()
    suites = list(tests.iter("testsuite"))
    counts = {
        key: sum(int(s.attrib.get(key, 0)) for s in suites)
        for key in ["tests", "failures", "errors", "skipped"]
    }
    if counts["failures"] or counts["errors"]:
        raise ValueError("Cannot report successful verification with failed tests")
    lines = [
        "# ReceiptSense - implementation and measured results",
        "",
        "Prepared: 23 October 2026. This report is generated from real saved model/run/test artifacts.",
        "",
        "[Run the application](../RUNNING.md) · [Detailed design and screenshot areas](PROJECT_REPORT.md)",
        "",
        "## Implemented system",
        "",
        "ReceiptSense is a working local receipt review desk. It accepts PNG/JPEG receipts, performs OCR, proposes company/address/date/total, highlights source regions, persists corrections separately from immutable predictions, and exports JSON/CSV. Streamlit provides scan, history, experiment and model/data pages; an optional FastAPI service exposes the same pipeline.",
        "",
        "The default OCR runtime is RapidOCR/ONNX Runtime using packaged Paddle-family PP-OCRv4 models. The native PaddleOCR adapter is also implemented. Native Windows oneDNN raised an operator error in this environment; the functional non-oneDNN native path was slower, so ONNX was selected as the explicit default. Outputs/checkpoints identify the actual OCR fingerprint.",
        "",
        f"Verified environment: {environment['platform']}; Python {environment['python'].split()[0]}; {environment['detected_hardware']}. Python execution device: {environment['execution_device']}; CUDA available to PyTorch: {environment['torch_cuda_available']}. Exact installed versions are in [the environment manifest](../artifacts/environment.json) and [dependency lock](../requirements.lock.txt).",
        "",
        "## Dataset and splits",
        "",
        "Source: pinned `jsdnrs/ICDAR2019-SROIE` public mirror. This is a mirror-derived split, not an official leaderboard reproduction. LICENSE, NOTICE, archive hashes and acquisition metadata are retained.",
        "",
        table(["Partition", "Receipts"], [[n, card["counts"][n]] for n in ["train", "validation", "test"]]),
        "",
        f"Before the final split, {card['exclusions']} records were excluded: eight training copies of held-out images, eight within-partition duplicate copies, and two missing-reference records. The seed is {card['seed']}. Byte-identical groups do not cross final partitions.",
        "",
        f"A cross-split perceptual audit checked {audit['receipts_checked']} receipts and flagged {len(audit['candidates'])} candidates. Both were visually reviewed as distinct transactions sharing a merchant template. The audit does not prove that every possible near duplicate is absent.",
        "",
        "Evidence: [dataset card](../data/dataset_card.json), [exclusions](../data/manifests/exclusions.json), [near-duplicate review](../data/near_duplicate_audit.json).",
        "",
        "## Annotation adaptation",
        "",
        "Training uses deployment OCR rather than the mirror's reference transcripts. Exact field-to-unit matches provide weak positive labels. Negative amount signs are preserved and cash/change/subtotal contexts are rejected for total supervision. With incomplete alignment, unverified units are ignored rather than labelled `O`; negative `O` labels are used only on receipts with all four fields located. Conflicting/unusable records are quarantined.",
        "",
        f"{labels['usable_receipts']} of {labels['receipts']} processed training receipts supply usable supervision; {labels['fully_aligned_receipts']} have all four fields aligned. Eight training receipts received a visual spot audit. The entire dataset is **not** claimed to be manually verified.",
        "",
        table(
            ["Field", "Unambiguous matches", "Ambiguous", "Unmatched"],
            [
                [
                    f,
                    labels["per_field"][f].get("exact_match", 0),
                    labels["per_field"][f].get("ambiguous", 0),
                    labels["per_field"][f].get("unmatched", 0),
                ]
                for f in FIELDS
            ],
        ),
        "",
        "Evidence: [alignment coverage](../data/processed/labels_train.summary.json), [visual spot audit](../data/annotation_audit.json).",
        "",
        "## Learned model and training",
        "",
        f"Actual training mode: **{training['adaptation']}**. The classifier is trained on frozen LayoutLMv3 features; this run is not all-parameter fine-tuning. The project separately implements the full fine-tuning command.",
        "",
        table(
            ["Setting", "Measured/configured value"],
            [
                ["Initial checkpoint", training["base_revision"]],
                ["Usable training receipts", training["training_receipts"]],
                ["Validation receipts", training["validation_receipts"]],
                ["Seed", training["seed"]],
                ["Learning rate", training["learning_rate"]],
                ["Selected epoch", training["selected_epoch"]],
                ["Epochs executed", len(training["history"])],
                ["Selection criterion", training["selection_metric"]],
                ["Device", training["device"]],
                ["Elapsed training/features time", f"{training['elapsed_seconds']:.1f} seconds"],
            ],
        ),
        "",
        "The tokenizer uses first-subword supervision, dynamic-length text windows, a fixed 512-token ceiling and center-preference ownership of overlapping units. Bottom-of-receipt totals are retained. Images and spatial boxes share one coordinate frame.",
        "",
        "Frozen feature caches use float16 storage, converted to float32 for classifier optimization and validation. Deployed model inference runs the backbone in float32. Checkpoint reload checks are recorded separately from cached-feature checkpoint selection.",
        "",
        f"The training-only tiny head fit achieved {percent(training['tiny_overfit_check']['supervised_training_accuracy'])} on {training['tiny_overfit_check']['supervised_units']} supervised units from {training['tiny_overfit_check']['receipts']} training receipts. The head was reset before the main run. This is a plumbing diagnostic, not held-out performance.",
        "",
        "Evidence: [model/training manifest](../artifacts/models/receiptsense-layoutlmv3/receiptsense_model.json), [tiny fit](../artifacts/models/receiptsense-layoutlmv3/tiny_overfit_check.json).",
        "",
        f"The deployed checkpoint was reloaded and its weight hash verified. Across {reload_check['n_validation_receipts']} validation receipts, raw field agreement with cached-feature selection outputs was {percent(reload_check['raw_field_agreement'])}; deployed strict macro field EM was {percent(reload_check['reloaded_validation_metrics']['macro_field_em'])}. No reselection used test labels. Evidence: [reload diagnostic](../artifacts/checkpoint_reload_check.json).",
        "",
        "## Full held-out comparison",
        "",
        f"Both methods use the same OCR outputs for all {result['n_receipts']} held-out receipts. Missing predictions remain failures. User corrections are excluded. Strict-v1 preserves case/punctuation/date representation after NFC/whitespace cleanup; normalized-v1 is a separately reported conservative parsing policy.",
        "",
    ]
    for policy in ["strict-v1", "normalized-v1"]:
        lines += [
            f"### {policy}",
            "",
            table(
                ["Method", "Company", "Address", "Date", "Total", "Macro field EM", "All-four EM"],
                [
                    [method]
                    + [percent(values[policy]["per_field"][f]["exact_match"]) for f in FIELDS]
                    + [percent(values[policy]["macro_field_em"]), percent(values[policy]["all_four_em"])]
                    for method, values in result["results"].items()
                ],
            ),
            "",
        ]
    paired = result["paired_strict"]
    lines += [
        f"Learned minus rules strict macro field difference: {paired['difference'] * 100:.2f} percentage points; receipt-bootstrap 95% interval [{paired['ci95'][0] * 100:.2f}, {paired['ci95'][1] * 100:.2f}] using {paired['samples']} resamples. This interval reflects receipt sampling, not variation across independently trained seeds.",
        "",
        "The measured learned run does not establish an overall strict accuracy improvement. It improves total extraction but loses date accuracy. Both methods have zero strict all-four-correct receipts in this held-out run, and addresses are a major weakness. The application is an inspectable academic review prototype; records need human checking.",
        "",
        "Normalized-v1 can score lower than strict-v1: ambiguous or two-digit-year reference dates are intentionally unresolved, and two parse failures never count as an exact match. It is a separately defined parsing policy, not a guaranteed relaxation of strict matching.",
        "",
        table(
            ["Method", "Review rate", "Error recall of review flag", "All-four accuracy among unflagged"],
            [
                [method]
                + [
                    percent(values["strict-v1"][key])
                    for key in ["review_rate", "review_error_recall", "unflagged_accuracy"]
                ]
                for method, values in result["results"].items()
            ],
        ),
        "",
        "Review flags are uncalibrated heuristics. Unflagged records are not established as safe for automatic acceptance by these results.",
        "",
        "Evidence: [full run summary](../artifacts/runs/held-out-v1/summary.json); per-receipt predictions and errors are saved alongside it.",
        "",
        "See [the error analysis](ERROR_ANALYSIS.md) for per-field strict error counts, normalized-policy differences, reference-string availability in OCR, and an auditable error ledger. These diagnostics do not claim causal OCR/classification attribution.",
        "",
        "## Paired correction and degradation probe",
        "",
        f"This is a separate seed-{robust['seed']} subset of {robust['n_parent_receipts']} held-out parent receipts. Every condition uses the same parents and a common max-side-1600 clean reference. Parameters were fixed before final scoring. These results do not replace the full held-out comparison.",
        "",
        f"Policy: {robust['probe_policy']}.",
        "",
        table(
            ["Condition", "Method", "Macro field EM", "All-four EM", "N"],
            [
                [
                    condition,
                    method,
                    percent(metrics["macro_field_em"]),
                    percent(metrics["all_four_em"]),
                    metrics["n_receipts"],
                ]
                for condition, methods in robust["results"].items()
                for method, metrics in methods.items()
            ],
        ),
        "",
        "Evidence: [probe summary](../artifacts/robustness/robustness-v1/summary.json). Parent IDs, condition references and raw predictions are retained.",
        "",
        "## Actual uncached latency",
        "",
        f"Measured on {latency['device']} with {latency['warmup_per_method']} warmup receipt per method and {latency['n_receipts']} timed receipts. Scope: {latency['scope']}. The two methods each recompute OCR for this timing study.",
        "",
        table(
            ["Method", "Median seconds", "p95 seconds", "N"],
            [
                [method, f"{m['median_seconds']:.3f}", f"{m['p95_seconds']:.3f}", m["n"]]
                for method, m in latency["methods"].items()
            ],
        ),
        "",
        f"OCR engine startup: {latency['engine_startup_seconds']:.3f} seconds; learned-model load: {latency['model_load_seconds']:.3f} seconds. Small-sample p95 is unstable. Full-run cached extraction timing is labelled separately in the evaluation summary.",
        "",
        f"Startup measurement scope: {latency.get('startup_scope', 'See the saved timing manifest')}.",
        "",
        "The benchmark ran rules first and the learned method second for each receipt. OCR time, OS caches and background load can vary across these sequential calls; the small-sample median is not evidence that the learned extractor itself is faster. Cached extraction timings show its additional model cost separately.",
        "",
        "Evidence: [timing rows](../artifacts/benchmark.json).",
        "",
        "## Verification and practical limits",
        "",
        f"Automated checks: {counts['tests']} collected, {counts['failures']} failures, {counts['errors']} errors, {counts['skipped']} skipped. Tests cover field selection, ambiguous parsing, geometry, annotation safety, real-tokenizer long windows, missing-output denominators, review immutability, exports, API behavior and Streamlit pages.",
        "",
        "The live browser workflow was exercised with a constructed receipt, evidence inspection and a correction/confirmation. Such demonstration evidence is separate from dataset accuracy. Native OCR was smoke-tested; the full held-out comparison uses the recorded ONNX engine.",
        "",
        "A separate API smoke check exercised real OCR upload/extraction, confirmation, JSON/CSV downloads and original-value preservation with an isolated database. Database connections close after each operation and failed writes roll back. Evidence: [API workflow check](../artifacts/api_smoke_check.json), [browser workflow check](../artifacts/ui_workflow_check.json), [automated test result](../artifacts/test-results.xml).",
        "",
        "Limitations: weak and partial supervision; a frozen backbone in the measured learned run; uncalibrated review signals; coarse line-derived evidence; one training seed; and dataset/domain limits. Two-digit/ambiguous dates intentionally retain unresolved normalized values. Full fine-tuning, alternative paper models and broad phone-photo/multilingual evaluation are not reported as completed experiments.",
        "",
        "## Screenshot areas",
        "",
        "The fourteen blank areas in [the original detailed report](PROJECT_REPORT.md) remain for you to fill. Follow [the capture guide](SCREENSHOT_GUIDE.md). Actual screenshots must be identified by receipt/run context rather than used as evidence for unmeasured performance.",
        "",
        "## Reproduce",
        "",
        "Use [RUNNING.md](../RUNNING.md), the dependency lock, pinned asset downloader, data manifests and saved model bundle. Recreate the held-out summary with the evaluation command; use a fresh run name to retain the existing evidence.",
        "",
    ]
    report_text = "\n".join(lines).replace("../artifacts/", "../results/artifacts/")
    report_text = report_text.replace("../data/processed/", "../results/data/processed/")
    (ROOT / "report/IMPLEMENTATION_REPORT.md").write_text(report_text, encoding="utf-8")
    print("Wrote report/IMPLEMENTATION_REPORT.md from saved evidence")


if __name__ == "__main__":
    main()
