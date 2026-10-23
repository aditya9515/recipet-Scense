# 07 - LayoutLMv3 preparation, training, and field decoding

[Index](../README.md) | Previous: [rules](06_RULES_BASELINE.md) | Next: [validation](08_VALIDATION_AND_HUMAN_REVIEW.md)

## Purpose

Adapt a pretrained document model to assign OCR text units to company/address/date/total. Inputs: selected image, OCR text, and spatial boxes. Output: token predictions reconstructed into the common field schema with evidence IDs.

Read [P01](../research/notes/P01_LayoutLMv3.md) first, with [P05](../research/notes/P05_LayoutLM.md) as background. The project fine-tunes downstream classification; it does not reproduce large-scale pretraining. The model's published results are not ReceiptSense results.

## 1. Freeze the model bundle

Proposed checkpoint: [microsoft/layoutlmv3-base](https://huggingface.co/microsoft/layoutlmv3-base), adapted using a token-classification head. Record the exact checkpoint revision and paired processor. A newly initialized receipt head requires training; loading a base model is not completed extraction.

The official [Transformers LayoutLMv3 guide](https://huggingface.co/docs/transformers/model_doc/layoutlmv3) describes RGB image processing, text tokenization, spatial boxes, and token classification. Use its APIs for the pinned installed version. Retain external OCR, disable processor-internal OCR, and keep image/boxes in one frame. The common model box convention is 0-1000. Verify tokenizer label handling and the configured text-length limit instead of guessing tensor shapes.

The checkpoint card identifies a CC BY-NC-SA 4.0 license. Record the actual checkpoint terms with the project artifacts; commercial deployment would require checking those terms. This pack proposes an academic implementation, not a deployment authorization.

## 2. Define labels and supervision

Freeze this proposed mapping:

| ID | Label |
|---|---|
| 0 | O |
| 1 | B-COMPANY |
| 2 | I-COMPANY |
| 3 | B-ADDRESS |
| 4 | I-ADDRESS |
| 5 | B-DATE |
| 6 | I-DATE |
| 7 | B-TOTAL |
| 8 | I-TOTAL |

One OCR word may split into several subwords. Proposed policy: supervise the first subword with the unit's BIO tag; set continuation pieces, special tokens, padding, and unresolved labels to the cross-entropy ignore index `-100`. During decoding, use the first-subword output per unit. Freeze this policy and verify it against the processor's actual label alignment.

If using line units, a single unit may contain more than one semantic field. Resolve this in the annotation/granularity design, not by assigning contradictory labels to one unit. Replicated line boxes with word text are an explicitly approximate option.

## 3. Normalize geometry

For processed image dimensions `W,H`, derive `[xmin,ymin,xmax,ymax]` from each polygon. Proposed normalization:

```text
x' = round(1000 * x / W)
y' = round(1000 * y / H)
box' = [xmin', ymin', xmax', ymax']
```

Validate `0 <= xmin' <= xmax' <= 1000`, similarly for y. Reject invalid original geometry; clamp tiny numerical boundary errors only with a logged policy. Tiny positive regions can quantize to zero width/height: record and apply a consistent bounded expansion if needed. Keep the original polygon for UI evidence.

The processor may resize the image to its configured pixel input shape. Normalized boxes still describe relative positions in that same selected image. Avoid passing boxes from the uncorrected original alongside a perspective-corrected image.

## 4. Handle long receipts without losing the total

A common configuration permits 512 text tokens including special tokens; inspect the chosen checkpoint configuration. Long receipts can exceed this limit, especially after subword tokenization.

Proposed strategy:

1. Tokenize with unit-to-piece mappings and measure token lengths before truncation.
2. Create overlapping windows fitting the configured text budget, reserving special-token capacity.
3. Prefer complete OCR-unit boundaries; log the rare unit that itself exceeds the budget.
4. Reuse the full selected image for each text window in the initial design.
5. Preserve global unit IDs and window offsets.
6. During training, assign a deterministic owner window per unit for token loss, masking duplicated overlap labels; log remaining entity boundary splits.
7. At inference, combine repeated-unit scores deterministically, initially using a fixed center-preference owner rule. Freeze any alternative averaging scheme on validation.
8. Reconstruct receipt-level spans using global IDs and BIO continuity rules.

Do not simply truncate to the first window. Save a long-receipt fixture whose total is near the bottom and verify it is represented and decoded.

## 5. Verify before training at scale

Inspect encoded examples with OCR text, normalized boxes, subword IDs, word mappings, labels, ignore positions and image shape. Assert finite inputs/loss, valid label IDs, and correct masks. Then run the tiny overfit check on 8-16 manually audited training receipts.

If loss falls but every output is `O`, inspect class imbalance, missing labels, ignored targets, span alignment and decoder mapping. Do not spend a full training budget on a broken label pipeline.

## 6. Proposed starting training configuration

These values are a modest starting point, not paper reproduction settings or guaranteed hardware requirements.

| Setting | Initial proposal |
|---|---|
| Initialization | `microsoft/layoutlmv3-base`, pinned revision |
| Objective | Masked token classification cross-entropy |
| Optimizer | AdamW |
| Learning rate | 2e-5; at most a small validation-only comparison with 5e-5 |
| Weight decay | 0.01 |
| Microbatch | 1-2 windows, determined by measured memory |
| Gradient accumulation | Set for effective batch around 8 windows; log actual value |
| Epoch budget | Up to 10, early stop after 2 non-improving validation evaluations |
| Warmup | About 10% of optimization steps |
| Gradient clipping | 1.0 |
| Primary seed | 42; optional chosen-config repeats 43 and 44 |
| Precision | FP32 first; mixed precision only after device compatibility/finite-loss check |
| Selection | Validation macro average of the four field exact-match scores |

Window counts vary by receipt. Record both windows and unique receipt counts per epoch. Keep the same search budget explicit; optional seeds describe repeated final configurations, not a search for the luckiest test seed.

If memory is insufficient, reduce microbatch and use accumulation, then consider supported gradient checkpointing. CPU development can validate contracts and perform tiny checks, but full training time must be measured. No GPU availability was established in this documentation task.

## 7. Decode units into fields

1. Map text logits back to original OCR units using the frozen subword/window policy.
2. Ignore special, padding and visual patch positions when constructing textual fields.
3. Convert invalid isolated `I-X` tags into `B-X` under a documented repair rule.
4. Build candidate spans; allow multiline addresses only under defined ordering/grouping rules.
5. If several spans compete for one field, rank with a frozen model-score/consistency policy. Store alternatives and mark close/conflicting cases ambiguous.
6. Join source text while retaining unit IDs and line breaks separately.
7. Pass selected raw values to shared validation.

Never concatenate every predicted `TOTAL` span into one amount. A candidate score aggregates uncalibrated token evidence; it is not a probability that the complete field is correct. Evidence identifies selected OCR regions, not a causal explanation of every transformer decision.

## 8. Save enough to reproduce

Save model weights, processor/tokenizer files, label maps, checkpoint revision, training/decoder config, dependency lock, source revision, split and OCR fingerprints, annotation policy/coverage, random seed, logs, validation predictions and selected-epoch reasoning.

A final reload check in a new process must recover the same label map and validation outputs within the declared numerical tolerance. Keep the raw predicted strings for strict field metrics even when parsed normalization succeeds.

## Completion gate

Pass only when the label audit and tiny overfit check succeed, long receipts are retained, evidence IDs survive decoding, the trained checkpoint reloads, and checkpoint selection uses validation rather than test results.
