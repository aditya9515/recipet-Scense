# ReceiptSense - implementation and measured results

Prepared: 23 October 2026. This report is generated from real saved model/run/test artifacts.

[Run the application](../RUNNING.md) · [Detailed design and screenshot areas](PROJECT_REPORT.md)

## Implemented system

ReceiptSense is a working local receipt review desk. It accepts PNG/JPEG receipts, performs OCR, proposes company/address/date/total, highlights source regions, persists corrections separately from immutable predictions, and exports JSON/CSV. Streamlit provides scan, history, experiment and model/data pages; an optional FastAPI service exposes the same pipeline.

The default OCR runtime is RapidOCR/ONNX Runtime using packaged Paddle-family PP-OCRv4 models. The native PaddleOCR adapter is also implemented. Native Windows oneDNN raised an operator error in this environment; the functional non-oneDNN native path was slower, so ONNX was selected as the explicit default. Outputs/checkpoints identify the actual OCR fingerprint.

Verified environment: Windows-11-10.0.26300-SP0; Python 3.12.14; Intel i5-11400H; 16 GB RAM; NVIDIA RTX 3050 Laptop 4 GB. Python execution device: CPU; CUDA available to PyTorch: False. Exact installed versions are in [the environment manifest](../results/artifacts/environment.json) and [dependency lock](../requirements.lock.txt).

## Dataset and splits

Source: pinned `jsdnrs/ICDAR2019-SROIE` public mirror. This is a mirror-derived split, not an official leaderboard reproduction. LICENSE, NOTICE, archive hashes and acquisition metadata are retained.

| Partition | Receipts |
|---|---|
| train | 550 |
| validation | 61 |
| test | 358 |

Before the final split, 18 records were excluded: eight training copies of held-out images, eight within-partition duplicate copies, and two missing-reference records. The seed is 42. Byte-identical groups do not cross final partitions.

A cross-split perceptual audit checked 969 receipts and flagged 2 candidates. Both were visually reviewed as distinct transactions sharing a merchant template. The audit does not prove that every possible near duplicate is absent.

Evidence: [dataset card](../data/dataset_card.json), [exclusions](../data/manifests/exclusions.json), [near-duplicate review](../data/near_duplicate_audit.json).

## Annotation adaptation

Training uses deployment OCR rather than the mirror's reference transcripts. Exact field-to-unit matches provide weak positive labels. Negative amount signs are preserved and cash/change/subtotal contexts are rejected for total supervision. With incomplete alignment, unverified units are ignored rather than labelled `O`; negative `O` labels are used only on receipts with all four fields located. Conflicting/unusable records are quarantined.

526 of 550 processed training receipts supply usable supervision; 37 have all four fields aligned. Eight training receipts received a visual spot audit. The entire dataset is **not** claimed to be manually verified.

| Field | Unambiguous matches | Ambiguous | Unmatched |
|---|---|---|---|
| company | 388 | 7 | 155 |
| address | 347 | 0 | 203 |
| date | 200 | 28 | 322 |
| total | 208 | 301 | 41 |

Evidence: [alignment coverage](../results/data/processed/labels_train.summary.json), [visual spot audit](../data/annotation_audit.json).

## Learned model and training

Actual training mode: **frozen-backbone classifier-head adaptation**. The classifier is trained on frozen LayoutLMv3 features; this run is not all-parameter fine-tuning. The project separately implements the full fine-tuning command.

| Setting | Measured/configured value |
|---|---|
| Initial checkpoint | cfbbbff0762e6aab37086fdd4739ad14fe7d5db4 |
| Usable training receipts | 526 |
| Validation receipts | 61 |
| Seed | 42 |
| Learning rate | 0.001 |
| Selected epoch | 7 |
| Epochs executed | 15 |
| Selection criterion | validation strict-v1 macro field EM |
| Device | cpu |
| Elapsed training/features time | 366.9 seconds |

The tokenizer uses first-subword supervision, dynamic-length text windows, a fixed 512-token ceiling and center-preference ownership of overlapping units. Bottom-of-receipt totals are retained. Images and spatial boxes share one coordinate frame.

Frozen feature caches use float16 storage, converted to float32 for classifier optimization and validation. Deployed model inference runs the backbone in float32. Checkpoint reload checks are recorded separately from cached-feature checkpoint selection.

The training-only tiny head fit achieved 100.00% on 126 supervised units from 8 training receipts. The head was reset before the main run. This is a plumbing diagnostic, not held-out performance.

Evidence: [model/training manifest](../results/artifacts/models/receiptsense-layoutlmv3/receiptsense_model.json), [tiny fit](../results/artifacts/models/receiptsense-layoutlmv3/tiny_overfit_check.json).

The deployed checkpoint was reloaded and its weight hash verified. Across 61 validation receipts, raw field agreement with cached-feature selection outputs was 100.00%; deployed strict macro field EM was 35.66%. No reselection used test labels. Evidence: [reload diagnostic](../results/artifacts/checkpoint_reload_check.json).

## Full held-out comparison

Both methods use the same OCR outputs for all 358 held-out receipts. Missing predictions remain failures. User corrections are excluded. Strict-v1 preserves case/punctuation/date representation after NFC/whitespace cleanup; normalized-v1 is a separately reported conservative parsing policy.

### strict-v1

| Method | Company | Address | Date | Total | Macro field EM | All-four EM |
|---|---|---|---|---|---|---|
| rules | 26.54% | 1.12% | 57.82% | 49.44% | 33.73% | 0.00% |
| layoutlmv3 | 27.93% | 1.68% | 30.73% | 70.11% | 32.61% | 0.00% |

### normalized-v1

| Method | Company | Address | Date | Total | Macro field EM | All-four EM |
|---|---|---|---|---|---|---|
| rules | 31.56% | 1.40% | 33.52% | 55.59% | 30.52% | 0.00% |
| layoutlmv3 | 33.24% | 6.42% | 16.20% | 75.70% | 32.89% | 2.23% |

Learned minus rules strict macro field difference: -1.12 percentage points; receipt-bootstrap 95% interval [-3.42, 0.98] using 1000 resamples. This interval reflects receipt sampling, not variation across independently trained seeds.

The measured learned run does not establish an overall strict accuracy improvement. It improves total extraction but loses date accuracy. Both methods have zero strict all-four-correct receipts in this held-out run, and addresses are a major weakness. The application is an inspectable academic review prototype; records need human checking.

Normalized-v1 can score lower than strict-v1: ambiguous or two-digit-year reference dates are intentionally unresolved, and two parse failures never count as an exact match. It is a separately defined parsing policy, not a guaranteed relaxation of strict matching.

| Method | Review rate | Error recall of review flag | All-four accuracy among unflagged |
|---|---|---|---|
| rules | 92.46% | 92.46% | 0.00% |
| layoutlmv3 | 99.72% | 99.72% | 0.00% |

Review flags are uncalibrated heuristics. Unflagged records are not established as safe for automatic acceptance by these results.

Evidence: [full run summary](../results/artifacts/runs/held-out-v1/summary.json); per-receipt predictions and errors are saved alongside it.

See [the error analysis](ERROR_ANALYSIS.md) for per-field strict error counts, normalized-policy differences, reference-string availability in OCR, and an auditable error ledger. These diagnostics do not claim causal OCR/classification attribution.

## Paired correction and degradation probe

This is a separate seed-42 subset of 12 held-out parent receipts. Every condition uses the same parents and a common max-side-1600 clean reference. Parameters were fixed before final scoring. These results do not replace the full held-out comparison.

Policy: common max-side 1600; automatic deskew; blur 5x5 sigma1.2; rotation7deg padded; contrast0.55; perspective0.04.

| Condition | Method | Macro field EM | All-four EM | N |
|---|---|---|---|---|
| clean | rules | 41.67% | 0.00% | 12 |
| clean | layoutlmv3 | 29.17% | 0.00% | 12 |
| deskew | rules | 41.67% | 0.00% | 12 |
| deskew | layoutlmv3 | 31.25% | 0.00% | 12 |
| blur | rules | 33.33% | 0.00% | 12 |
| blur | layoutlmv3 | 25.00% | 0.00% | 12 |
| rotation | rules | 22.92% | 0.00% | 12 |
| rotation | layoutlmv3 | 27.08% | 0.00% | 12 |
| contrast | rules | 37.50% | 0.00% | 12 |
| contrast | layoutlmv3 | 33.33% | 0.00% | 12 |
| perspective | rules | 31.25% | 0.00% | 12 |
| perspective | layoutlmv3 | 25.00% | 0.00% | 12 |

Evidence: [probe summary](../results/artifacts/robustness/robustness-v1/summary.json). Parent IDs, condition references and raw predictions are retained.

## Actual uncached latency

Measured on CPU with 1 warmup receipt per method and 10 timed receipts. Scope: warm uncached image preparation+OCR+extraction+validation; startup measured separately. The two methods each recompute OCR for this timing study.

| Method | Median seconds | p95 seconds | N |
|---|---|---|---|
| rules | 2.936 | 3.701 | 10 |
| layoutlmv3 | 2.377 | 3.185 | 10 |

OCR engine startup: 0.571 seconds; learned-model load: 0.280 seconds. Small-sample p95 is unstable. Full-run cached extraction timing is labelled separately in the evaluation summary.

Startup measurement scope: fresh engine/model instances in this process; excludes Python/dependency imports and OS cold-cache guarantees.

The benchmark ran rules first and the learned method second for each receipt. OCR time, OS caches and background load can vary across these sequential calls; the small-sample median is not evidence that the learned extractor itself is faster. Cached extraction timings show its additional model cost separately.

Evidence: [timing rows](../results/artifacts/benchmark.json).

## Verification and practical limits

Automated checks: 29 collected, 0 failures, 0 errors, 0 skipped. Tests cover field selection, ambiguous parsing, geometry, annotation safety, real-tokenizer long windows, missing-output denominators, review immutability, exports, API behavior and Streamlit pages.

The live browser workflow was exercised with a constructed receipt, evidence inspection and a correction/confirmation. Such demonstration evidence is separate from dataset accuracy. Native OCR was smoke-tested; the full held-out comparison uses the recorded ONNX engine.

A separate API smoke check exercised real OCR upload/extraction, confirmation, JSON/CSV downloads and original-value preservation with an isolated database. Database connections close after each operation and failed writes roll back. Evidence: [API workflow check](../results/artifacts/api_smoke_check.json), [browser workflow check](../results/artifacts/ui_workflow_check.json), [automated test result](../results/artifacts/test-results.xml).

Limitations: weak and partial supervision; a frozen backbone in the measured learned run; uncalibrated review signals; coarse line-derived evidence; one training seed; and dataset/domain limits. Two-digit/ambiguous dates intentionally retain unresolved normalized values. Full fine-tuning, alternative paper models and broad phone-photo/multilingual evaluation are not reported as completed experiments.

## Screenshot areas

The fourteen blank areas in [the original detailed report](PROJECT_REPORT.md) remain for you to fill. Follow [the capture guide](SCREENSHOT_GUIDE.md). Actual screenshots must be identified by receipt/run context rather than used as evidence for unmeasured performance.

## Reproduce

Use [RUNNING.md](../RUNNING.md), the dependency lock, pinned asset downloader, data manifests and saved model bundle. Recreate the held-out summary with the evaluation command; use a fresh run name to retain the existing evidence.
