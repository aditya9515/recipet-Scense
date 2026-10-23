# 10 - Evaluation, controlled experiments, and error analysis

[Index](../README.md) | Previous: [application](09_APPLICATION_AND_EXPORT.md) | Next: [engineering](11_ENGINEERING_AND_REPRODUCIBILITY.md)

## Main question and unit of analysis

Does the learned extractor improve receipt fields relative to transparent rules under the same OCR inputs? The independent unit is the original receipt. Four fields and multiple degraded versions from one receipt are not independent samples.

The main metric is project-defined field exact match. Published benchmark numbers can use different OCR, splits, normalization and entity metrics. Describe the actual scoring protocol; do not imply official SROIE leaderboard equivalence without the official evaluator and matching conditions.

## 1. Freeze scoring policies before test access

**Primary strict policy:** compare raw extracted field strings after Unicode NFC normalization, trimming, and replacing whitespace runs with a single space. Preserve case, punctuation, digits and date representation. Record this policy as `strict-v1`; include worked edge cases in the metric configuration.

**Secondary normalized policy:** case-fold company/address with the declared whitespace policy; compare dates only when reference and prediction have unambiguous parses under the same declared formats; compare amounts using conservative decimal parsing. Parse failure is a mismatch. Do not treat two parse failures or two ambiguous nulls as a correct field. Retain raw strict scores alongside normalized scores.

Reference field absence needs a frozen policy. Primary four-field evaluation should use receipts whose reference labels are verified present for those fields before inference. Pre-existing annotation defects/exclusions need a ledger. Prediction absence on a labelled field is an error. Never exclude receipts because a method fails.

## 2. Required metrics

For field `f`, with `N_f` predeclared evaluable receipts:

```text
EM_f = correct complete field predictions / N_f
MacroFieldEM = (EM_company + EM_address + EM_date + EM_total) / 4
AllFourEM = receipts with all four fields correct / receipts with four reference fields
```

On a common fully labelled set, `MicroFieldEM = correct fields / (4*N)` equals macro field exact match; explain different denominators if they do not coincide. Missing outputs, inference exceptions and abstentions count as wrong for the affected evaluable fields.

Optional diagnostics:

- Entity/span precision, recall and F1 on independently audited span references.
- Token F1 with its label/ignore policy, never substituted for full field correctness.
- OCR CER/WER with specified transcript alignment and normalization.
- Review rate, error recall of review flags, and accuracy among unflagged outputs.
- Median and p95 stage/end-to-end latency, failure rate and peak memory.

Define review metrics explicitly: `review_rate = flagged_receipts/N`; `error_recall = flagged_wrong_receipts/all_wrong_receipts`; `unflagged_accuracy = correct_unflagged_receipts/all_unflagged_receipts`. State whether “wrong” means any field wrong. A zero denominator yields undefined, not a fabricated zero or one.

## 3. Minimum experiment matrix

| ID | Image condition | OCR source | Extractor | Purpose |
|---|---|---|---|---|
| E01 | Identity preparation | Frozen engine | Rules | Required baseline |
| E02 | Identity preparation | Same cached OCR as E01 | Trained LayoutLMv3 | Required learned comparison |
| E03 | Selected correction variant | Frozen engine, recomputed for variant | Rules | Preprocessing effect for baseline |
| E04 | Same corrected images as E03 | Same cached OCR as E03 | LayoutLMv3 | Corrected pipeline comparison |
| E05, optional | Identity | Provided reference transcripts/boxes | Learned extractor | OCR error diagnostic, explicitly oracle inputs |
| E06, optional | Identity | Same as E02 | Learned + explicit overrides | Isolate override effect |

For E02/E04, state whether the checkpoint is fixed or retrained on corresponding training images/OCR. Fixed-checkpoint comparison isolates inference preprocessing under that checkpoint. Retraining makes it a full pipeline comparison and must be logged as such.

Choose any production configuration with validation only. Keep all planned E01-E04 results even if one is weaker. Any unused optional experiment is “not evaluated”.

## 4. Robustness experiment

Define a small fixed perturbation grid on held-out receipt images: for example, mild/moderate blur, padded rotations of 5 and 10 degrees, reduced contrast, and controlled perspective shifts. These are proposed conditions, not validated realism claims. Store seed, parameters, interpolation, output size and parent ID.

Keep field labels attached to the parent when content remains intact; report if a transformation makes labels unreadable or crops them. Use the identical degraded images for both extraction methods. Recompute OCR and evidence for each image. Do not tune on the robustness test results.

Report per-condition counts, macro/all-four EM, and change from the same receipts' clean scores. If a robustness condition is used for development, reserve a new confirmation set or disclose that it became development data.

## 5. Latency protocol

Record hardware, device, image resolution distribution, batch size, warmup count and timed sample count.

Separate:

1. Cold application/model startup.
2. Uncached image preparation + OCR + extraction + validation.
3. Cached-OCR extraction + validation.
4. UI/review time, if measured, as a user interaction quantity.

Use a monotonic timer; synchronize GPU work before timing boundaries. Warm the model before steady-state timing. Do not present cached OCR timing as full receipt processing. Report median and p95 with sample counts; p95 from a tiny set should be identified as unstable.

## 6. Compare with uncertainty

Save paired per-receipt correctness for each method. Report absolute percentage-point differences. For confidence intervals, bootstrap original receipt IDs (for example 1,000 resamples, seed 42); keep all fields/versions of each sampled receipt together. Repeated training seeds measure another source of variation: report their scores separately or mean/range as appropriate, without choosing a winner by test score.

If resources do not permit repeated seeds or intervals, state that limitation. A small score difference from one run is not proof of a general improvement.

## 7. Error taxonomy

| Stage | Example | Evidence needed |
|---|---|---|
| Image preparation | Deskew clips bottom total | Original/processed comparison |
| Detection | Amount region absent | Image and raw detector output |
| Recognition | `8` read as `B` | Correct region crop and OCR text |
| Reading order | Label separated from amount | Ordered OCR list and geometry |
| Annotation | Reference total aligned to subtotal | Audited training labels |
| Classification | Correct OCR assigned to cash | Unit labels and competing spans |
| Decoding | Multiline address truncated | Token/window/global-unit mapping |
| Normalization | Ambiguous date forced | Raw string and parse alternatives |
| Reference issue | Entity string disagrees with image | Independent annotation review |

Record receipt ID, method/run ID, reference, prediction, stage, explanation, and whether it affected the final field. Inspect both successes and failures; screenshots chosen only for attractive results are insufficient evidence.

## 8. Result integrity checklist

- Predictions are saved before scoring and before manual review.
- Every method has an identified OCR/cache fingerprint and split hash.
- Rules, decoders, normalization and thresholds were frozen before test scoring.
- Failed receipts remain in denominators according to the declared scope.
- Empty placeholders never become numeric “0%” results.
- Learned, oracle, combined and human-corrected outputs have separate identities.
- Test-driven revisions are disclosed and use fresh confirmation data where possible.

Use the empty tables in [the report](../report/PROJECT_REPORT.md) once actual runs exist. None of these metrics has been measured in this workspace yet.
