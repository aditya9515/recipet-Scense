# 06 - Transparent rules baseline

[Index](../README.md) | Previous: [OCR](05_OCR_AND_READING_ORDER.md) | Next: [LayoutLMv3](07_LAYOUTLMV3_TRAINING.md)

## Purpose and contract

Build a simple, explainable comparator before training the transformer. Input: canonical OCR with text and geometry. Output: the same four-field prediction schema as the learned model, including evidence IDs, ambiguity and rule reasons.

Rules establish whether the added model complexity helps. [P09](../research/notes/P09_Joint_Receipt_Model.md) is related hybrid work; our planned rules are a separate, smaller design, not a reproduction of that paper.

## 1. Create candidates before choosing winners

Preserve raw text. Derive normalized text only for pattern matching. Reconstruct lines from OCR IDs and keep their geometry. For each field, create candidate spans with value, unit IDs, line IDs, supporting keywords, and competing interpretations.

Each score is a heuristic rank, not a probability. Prefer an explicit ambiguous result when candidates tie or conflict. Keep the runner-up and score margin for error analysis.

## 2. Company candidates

Proposed signals: near the top of the receipt, alphabetic content, a plausible business suffix if present, and a visually/textually coherent line. Penalize obvious headings (`TAX INVOICE`, `RECEIPT`), telephone lines, dates, item lines and address-like strings.

Do not assume the topmost text is the company. Logos, slogans and receipt headings can precede it. If the company name spans lines, group only adjacent compatible lines and retain both evidence sets. Avoid a lookup dictionary containing held-out merchant labels.

## 3. Address candidates

Start from consecutive lines below or near a company candidate. Possible clues include street/place words, building numbers and postal patterns appropriate to the audited domain. Penalize telephone-only, tax-registration-only, item and payment lines.

Stop grouping at a clear field boundary, a large spatial gap, or a competing section. Do not absorb the entire receipt header because it is near the shop name. Location-specific cues should be described as domain assumptions and checked against validation examples.

## 4. Date candidates

Find plausible numeric or textual date patterns, especially near `DATE` or transaction context. Recognize competing date-like identifiers and time values. Keep the receipt's original date string, parsed alternatives and evidence.

For `03/04/2025`, both 3 April and 4 March are plausible without a declared locale/context. Return ambiguity rather than force a normalized ISO date. The display/export can preserve the raw string while awaiting confirmation.

## 5. Total candidates

1. Identify numeric amount spans using a conservative parser.
2. Link them to labels on the same line or nearby lines using geometric distance.
3. Reward explicit final-total labels such as `TOTAL` or `AMOUNT DUE`, accounting for the fact that `SUBTOTAL` contains the substring `TOTAL`.
4. Penalize `SUBTOTAL`, `CASH`, `TENDERED`, `CHANGE`, unit-price and tax-only contexts.
5. Use relative position as a weak cue, not a requirement that total is always last.
6. Check simple arithmetic only when all supporting operands have clear meanings; use inconsistency as a warning, never proof of a replacement value.
7. Preserve multiple plausible totals as an ambiguity condition.

Constructed test: subtotal 470.00, tax 23.50, total 493.50, cash 500.00, change 6.50. Expected total is 493.50. A “largest amount” rule fails here.

## Suggested scoring representation

```text
score(candidate) = sum(weight[name] * signal[name])
```

Signals might include exact final-total keyword, same-line amount, distance, competing cash label and parse validity. Choose a small fixed set of weights on training/validation only. Save their values, revision, and selection rationale. Do not present proposed weights as empirically optimal before experimentation.

## Relationship with shared validation

Rules choose evidence-bearing raw values. [Module 08](08_VALIDATION_AND_HUMAN_REVIEW.md) applies the same format/parser checks used after learned extraction. A parser can reject an invalid total but should not select a different OCR number without a separately named extraction/override policy.

Keep three possible experiment identities distinct: rules only, learned extraction with shared format validation, and learned extraction plus explicit rule overrides. The required first comparison is rules versus learned extraction using shared OCR.

## Development and failure analysis

Use train/validation examples to inspect wrong candidates. Record the reason: heading chosen as company, telephone included in address, ambiguous date forced, cash mistaken for total, OCR missing the amount, or reading-order failure.

Useful behavior checks cover repeated total values, missing fields, multiple dates, multiline addresses and no OCR. Deterministic examples should verify outcomes, evidence IDs and ambiguity, rather than merely assert that a scoring function runs.

## Completion evidence

- Versioned candidate signals/weights and field reconstruction policy.
- Validation predictions in the common schema.
- Explanations for several correct and incorrect receipts.
- Same OCR fingerprint as the paired learned run.
- No test-dependent merchant lists or rules.

The baseline may outperform a poorly trained model. That is an experimental finding to investigate and report, not a reason to remove it.
