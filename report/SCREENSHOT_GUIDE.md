# Screenshot and report completion guide

[Report](PROJECT_REPORT.md) | [Project plan](../PROJECT_PLAN.md)

The report already contains fourteen **blank image areas**. They are neutral SVG placeholders, not screenshots. No screenshot capture is required at this planning stage.

## How to fill an area later

1. Capture the actual implementation or experiment output.
2. Save the image under `report/assets/` using the PNG filename below.
3. In `PROJECT_REPORT.md`, change that area's image path from `_placeholder.svg` to `.png`.
4. Keep the figure ID and replace its caption with the actual receipt/run/configuration context.
5. Open the Markdown preview and confirm that text is readable and the image matches the claim.

Example: replace `assets/S04_ocr_placeholder.svg` with `assets/S04_ocr.png`. You can then remove that unused placeholder if desired; removal is optional.

## Capture list

| ID | Save actual image as | What it should prove | Useful caption details |
|---|---|---|---|
| S01 | `S01_dataset.png` | Inventory and fixed partition evidence | Release, counts, split hash |
| S02 | `S02_input.png` | A genuine supported receipt input | Receipt ID, image type/size |
| S03 | `S03_preprocessing.png` | Real original-versus-processed comparison | Variant, transform parameters |
| S04 | `S04_ocr.png` | OCR text with aligned regions | Engine/checkpoints, word/line geometry, frame |
| S05 | `S05_labels.png` | Audited field-to-OCR labels | Annotation policy, receipt ID, reviewer check |
| S06 | `S06_training.png` | Training and validation selection evidence | Run ID, seed, epoch, metric, selected checkpoint |
| S07 | `S07_application.png` | Main upload/preview workflow | Application version and actual mode |
| S08 | `S08_fields.png` | Four extracted values and review states | Method, receipt ID, pending flags |
| S09 | `S09_evidence.png` | Selected field's actual source geometry | Field, OCR IDs/frame, coarse geometry if used |
| S10 | `S10_uncertainty.png` | A real ambiguous or missing output | Reason, alternatives, review action |
| S11 | `S11_correction.png` | User edit with preserved original prediction | Before/after value, review state |
| S12 | `S12_export.png` | Actual structured record output | Receipt/review IDs, JSON/CSV format |
| S13 | `S13_results.png` | Measured method comparison | Run IDs, N, strict/normalized metric policy |
| S14 | `S14_failure.png` | A representative error with evidence | Reference, prediction, failure stage |

Report order follows the narrative, so S05 appears in the annotation section before the image/OCR screenshots. IDs are stable and need not be numerically ordered in the report.

## Capture quality

Use enough resolution to read receipt text, labels, values and table headings. Prefer focused application/output views. Include original and transformed image frames when claiming alignment. Keep displayed values consistent with saved exports and run artifacts.

Use authorized dataset/project receipts. If sensitive information appears, redact the presentation copy where appropriate and state that redaction was for display. Do not alter measured result files to make screenshots attractive.

For training/results, a screenshot supports presentation; saved machine-readable logs and predictions remain the primary evidence. A synthetic UI fixture can explain interaction only if labelled synthetic, and cannot prove model performance.

## Tables to complete before submission

- Dataset counts and alignment coverage.
- Actual software/hardware/training settings.
- Strict and secondary normalized field results.
- Paired differences and any confidence/seed analysis.
- Robustness conditions, counts and scores.
- Cold/uncached/cached latency and failures.
- Review behavior and selected error cases.

If an optional experiment was not run, write **not evaluated** with a short reason. Do not enter zero for an unknown metric. Replace the draft conclusion only after checking each claim against its saved evidence.
