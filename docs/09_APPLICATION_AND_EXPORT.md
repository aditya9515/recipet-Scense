# 09 - Local application, storage, and export

[Index](../README.md) | Previous: [review](08_VALIDATION_AND_HUMAN_REVIEW.md) | Next: [evaluation](10_EVALUATION_AND_EXPERIMENTS.md)

## Product goal

Demonstrate the complete receipt journey: upload -> extraction -> evidence -> correction -> export. A simple local Streamlit interface is the proposed first implementation. Keep the pipeline callable outside the UI so evaluation does not depend on browser interactions.

## 1. Main screen

| Area | User-facing content |
|---|---|
| Upload | PNG/JPEG chooser, supported limits, process button |
| Receipt pane | Original/processed toggle, zoom, selected-field highlights |
| Field pane | Four editable values, status, review reasons |
| Evidence action | Click field to show selected regions and source text |
| Review actions | Save edits and confirm reviewed record |
| Export | Download reviewed JSON/CSV with review state |
| Diagnostics, optional | Engine/model revision, stage times, raw OCR, error detail |

Keep model internals in optional diagnostics. The main user flow should explain field problems in plain language. Always identify the active extraction method if a rules fallback is used.

## 2. Interaction sequence

1. Decode and preview the uploaded receipt.
2. Show a processing state; disable duplicate submissions while one job is active.
3. Run the shared pipeline and save its immutable prediction.
4. Populate fields, distinguishing null from empty text and failed processing.
5. Let the user inspect evidence and edit values.
6. Save a reviewed record linked to the prediction.
7. Export the chosen version with its status clearly labelled.

If unreviewed export is allowed, label it as unreviewed; do not mark it confirmed automatically. A failed model load offers a visible retry or an explicitly named rules mode.

## 3. State and caching

Load the chosen model once per application process when supported. Keep uploaded receipt state separate from the model cache. A rerender must not accidentally re-run inference or lose edits. Changing receipt resets its field/evidence selections and unsaved-edit state.

Use the OCR cache from module 05. Record whether displayed timings include a cache hit. Do not use cached full outputs when claiming uncached end-to-end latency.

## 4. Proposed local persistence

Use JSONL for immutable experiment outputs and SQLite for review records if persistent review history is needed.

Proposed SQLite entities:

| Table | Main contents |
|---|---|
| `receipts` | Receipt ID, image hash/reference, created time, metadata |
| `predictions` | Prediction ID, receipt ID, method/config revisions, immutable result JSON |
| `reviews` | Review ID, prediction ID, final values, state, timestamps |
| `correction_events` | Review ID, field, old/new values, source, time |

Use parameterized queries, stable generated IDs and atomic transactions. Do not overwrite images based on user filenames. Keep user receipts local in this proposed app; external OCR services are outside the initial design.

## 5. Optional API boundary

Add FastAPI only if a second client or separate worker needs it. Suggested route contract:

| Route | Purpose |
|---|---|
| `GET /health` | Process/model readiness without exposing receipt contents |
| `POST /receipts` | Validate upload and create receipt record |
| `POST /receipts/{id}/extract` | Run a named method/configuration |
| `GET /predictions/{id}` | Read immutable automated output |
| `POST /predictions/{id}/reviews` | Save edited/confirmed review |
| `GET /reviews/{id}/export?format=json` | Export the explicit reviewed version |

These routes are now implemented in `api/main.py`; see [RUNNING.md](../RUNNING.md) for the local server command and interactive API documentation. The Streamlit application still uses the shared functions directly.

## 6. JSON export

Use UTF-8, schema version, IDs, four final fields, review state, and machine provenance. Money is a decimal string, dates retain raw text where unresolved, and missing values are JSON null. Include raw predictions and evidence references in a nested object or clearly linked export if detailed audit is required.

Do not invent a normalized date/currency just to avoid nulls. A machine export and reviewed export must remain distinguishable by both contents and metadata.

## 7. CSV export

Proposed flat columns:

```text
receipt_id,prediction_id,review_id,company,address,date_raw,date_normalized,
total_raw,total_normalized,currency,review_status,extraction_method
```

Use a real CSV library to quote commas, quotes, and line breaks. Document the chosen encoding and null representation. If intended for spreadsheet opening, neutralize formula-leading textual cells such as `=`, `+`, `-`, or `@` under an explicit CSV presentation policy, while retaining untouched values in JSON. Do not alter valid numeric money semantics silently.

## 8. Error states and practical checks

Test unreadable upload, oversize image, empty OCR, unavailable checkpoint, ambiguous date, no total, a long address, rerender after editing, and repeated export. Validate downloaded JSON/CSV round trips and ensure the record belongs to the visible receipt.

Completion means a teammate can upload a new supported receipt, inspect four fields, see evidence, make a correction, confirm and open both exports. Screenshot proof comes later through [the capture guide](../report/SCREENSHOT_GUIDE.md).
