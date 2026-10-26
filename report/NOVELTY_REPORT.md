# ReceiptSense - novelty implementation and validation

This branch adds a multi-line address graph and selective crop OCR recovery, capped at two additional OCR calls. Both extractors use the same fixed checkpoint/OCR assets. No new backbone fine-tuning or active learning was performed.

[Roadmap and design](../docs/14_NOVELTY_ROADMAP.md) · [Commands](../RUNNING.md) · [Original measured report](IMPLEMENTATION_REPORT.md)

## Validation comparison

These are development results on 61 validation receipts, including the data used to select graph thresholds. They are not independent confirmation or a new held-out claim.

| Extractor / mode | Strict macro EM | Strict address EM | All-four EM | Address token F1 | Corrected fields | Introduced errors | Retry rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| rules/none | 32.79% | 4.92% | 0.00% | 50.48% | 0 | 0 | 0.00% |
| rules/address | 32.79% | 4.92% | 0.00% | 59.36% | 0 | 0 | 0.00% |
| rules/retry | 34.02% | 4.92% | 0.00% | 51.65% | 3 | 0 | 93.44% |
| rules/combined | 34.43% | 4.92% | 0.00% | 62.32% | 4 | 0 | 91.80% |
| layoutlmv3/none | 35.66% | 4.92% | 1.64% | 69.55% | 0 | 0 | 0.00% |
| layoutlmv3/address | 35.25% | 3.28% | 0.00% | 62.62% | 0 | 1 | 0.00% |
| layoutlmv3/retry | 37.30% | 4.92% | 1.64% | 69.55% | 4 | 0 | 100.00% |
| layoutlmv3/combined | 36.89% | 3.28% | 0.00% | 62.95% | 4 | 1 | 100.00% |

Full strict/normalized per-field metrics, correction/error counts, review rates and paired receipt-bootstrap intervals are in [the aggregate summary](../results/novelty/validation-v1/summary.json). Missing predictions remain failures. Merchant-grouped intervals are unavailable for SROIE validation because no verified merchant grouping is provided; the fresh pilot will supply it.

Combined default eligibility on validation: {'rules': False, 'layoutlmv3': False}. Baseline remains the default; these development checks do not establish reliable automatic acceptance.

## Actual uncached CPU timing

10 validation receipts, one warmup per condition. warm uncached image preparation + base OCR + enhancement + extraction + validation; excludes startup and persistence. Condition order rotates between receipts. Small-sample p95 is unstable.

| Extractor / mode | Median seconds | p95 seconds | Mean total OCR calls |
|---|---:|---:|---:|
| rules/none | 1.636 | 2.515 | 1.00 |
| rules/address | 1.615 | 3.175 | 1.00 |
| rules/retry | 4.427 | 6.152 | 2.90 |
| rules/combined | 3.682 | 5.705 | 2.90 |
| layoutlmv3/none | 2.258 | 3.330 | 1.00 |
| layoutlmv3/address | 2.316 | 3.435 | 1.00 |
| layoutlmv3/retry | 5.412 | 8.741 | 3.00 |
| layoutlmv3/combined | 5.104 | 7.438 | 3.00 |

[Timing rows](../results/novelty/validation-v1/latency.json) separate actual uncached engine calls from logical retry attempts. Accuracy-run timings permit caches and must not be substituted for this timing study.

## Verification and limitations

The frozen checkpoint and all 488 saved predictions were audited for evidence resolution, stable fingerprints and the two-attempt limit. Both baseline modes reproduce all four raw fields on all 61 validation receipts from their original saved runs. [Integrity evidence](../results/novelty/invariants.json).

Automated suite: 51 tests, 0 failures, 0 errors, 0 skipped. [Test evidence](../results/novelty/test-results.xml). The [real CPU API workflow](../results/novelty/workflow_check.json) covers enhanced extraction, evidence, correction, immutable originals and JSON/CSV export using explicitly synthetic examples.

The address graph and retry ranking are heuristics, and OCR geometry may be line-derived. Re-reading can introduce errors; those are counted above. No formal risk calibration, broad multilingual support or new full fine-tuning result is claimed. The contribution is this particular evidence-preserving, budgeted recovery strategy and its measured trade-off, not the invention of OCR or uncertainty-based review.

## Fresh confirmation remains pending

The team must collect 50 English/Latin-script Indian receipts across ten merchants (five distinct transactions each), independently double-label the four fields, resolve disagreements and freeze the set before inference. No real fresh confirmation receipts have been supplied, and no confirmation accuracy is reported. The collection/freeze/evaluation tooling is implemented and its integrity checks are tested with synthetic fixtures only.

Private images, merchant mapping, references, prediction details and locks remain outside Git. Only aggregate confirmation metrics and explicitly approved redacted examples may be published. The original fourteen screenshot areas remain blank.

[Frozen policy identity](../results/novelty/freeze.json) · [Validation selection](../results/novelty/policy_selection.json) · [Original baseline identity](../results/novelty/baseline_manifest.json)
