# 08 - Validation, uncertainty, evidence, and correction

[Index](../README.md) | Previous: [LayoutLMv3](07_LAYOUTLMV3_TRAINING.md) | Next: [application](09_APPLICATION_AND_EXPORT.md)

## Responsibility

Convert extracted raw strings into usable, inspectable fields without hiding ambiguity. Shared validation takes either extractor's prediction. The UI then allows a person to confirm or correct it. Machine predictions and human-reviewed records remain separate.

## 1. Normalize conservatively

| Field | Allowed initial handling | Avoid |
|---|---|---|
| Company | Unicode/whitespace cleanup; preserve original | Replacing with a guessed merchant dictionary name |
| Address | Preserve line order; normalize whitespace for export | Removing numbers, punctuation, or unrelated-looking lines without evidence |
| Date | Parse supported formats; retain raw text and alternatives | Choosing day/month order for ambiguous dates silently |
| Total | Decimal arithmetic and declared separators | Binary floating-point money, inferred zeros, or selecting largest amount |

Date example: `14/02/2025` can have an unambiguous day-first parse if that pattern is supported. `03/04/2025` requires context or confirmation. A two-digit year needs a documented policy; uncertain interpretation stays unresolved.

Amount example: `1,234.50` and `1.234,50` may represent the same value in different locales, while `1,234` is ambiguous without context. Never remove all punctuation indiscriminately. Do not replace `O` with `0` in raw predictions; any repair proposal is explicit, evidence-bearing, and scored separately if it changes the field.

Currency stays null when unsupported. Dataset geography is not proof that a particular receipt explicitly shows a currency.

## 2. Generate review reasons

Proposed stable codes:

```text
missing_company, missing_address, missing_date, missing_total,
ambiguous_date, amount_parse_failed, multiple_total_candidates,
low_ocr_evidence, low_model_margin, unsupported_box_precision,
inconsistent_amounts, empty_ocr, extraction_failed
```

Separate syntax validity from semantic correctness. A plausible number can still be cash. A syntactically valid address can still be incomplete.

## 3. Treat scores honestly

OCR recognition scores, model token softmax scores and rule scores measure different things. Do not average them and label the result “confidence: 95%”. Initially show qualitative review flags plus explicit score names in a diagnostics panel.

If a review threshold is needed, choose it on validation and measure how many errors it catches and how much correct output it also flags. A useful evaluation plots coverage versus error among unflagged predictions. If calibration is attempted, fit it using validation-only data and assess calibration separately; no calibration has been performed now.

Missing fields, model exceptions and empty OCR always require review. A high token score never overrides a parser ambiguity automatically.

## 4. Show field evidence

Selecting a field should highlight its selected unit polygons on the receipt. Name the displayed coordinate frame. Use the transform chain when drawing processed OCR evidence on the orientation-corrected original.

For multiline fields, highlight each supporting line/region instead of a giant rectangle that implies unrelated text was selected. If geometry is line-level or replicated from lines, label the highlight as coarse. A typed correction may have no engine region; keep that fact visible.

Evidence tells the user what text was selected. It does not prove the model's full reasoning or guarantee the field label is correct.

## 5. Review workflow

1. Show machine values, missing/ambiguous states and evidence.
2. User selects a field and inspects its highlighted source.
3. User edits raw/final text or confirms the suggestion.
4. Revalidate edited values; unresolved ambiguities remain visible.
5. Record a correction event and the new review state.
6. Confirm the record explicitly, then export the reviewed version.

Keep the original prediction immutable. Persist `prediction_id`, reviewer/session identifier if used, timestamp, previous/new value, correction source and optional evidence selection. Do not claim that a correction was model-generated.

Suggested states: `unreviewed -> edited -> confirmed`, with editing a confirmed record returning it to `edited`. Invalid date/amount parsing can coexist with an explicitly retained raw string; export indicates unresolved status.

## 6. Rule overrides and feedback boundaries

Initial policy: validation flags problems but does not silently replace learned spans with rule candidates. If the team later adds overrides, keep the original prediction, override reason, and separate combined-method evaluation.

Reviewed examples can become future training data only through a new versioned annotation/split process. Do not let corrections to evaluation receipts flow into the active checkpoint while continuing to report the same held-out score.

## Completion evidence

- Invalid and ambiguous dates remain understandable and editable.
- Total normalization preserves exact decimal value and source text.
- Every automated non-null field has valid OCR evidence IDs.
- Highlighted evidence aligns in the displayed frame.
- A correction can be audited without changing automated results.
- Review rates are measured separately from automated extraction accuracy.
