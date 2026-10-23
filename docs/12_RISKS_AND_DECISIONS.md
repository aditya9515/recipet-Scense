# 12 - Technical analysis, risks, and decision log

[Index](../README.md) | Previous: [engineering](11_ENGINEERING_AND_REPRODUCIBILITY.md) | Next: [team and viva](13_TEAM_SCHEDULE_AND_VIVA.md)

## Why this document exists

This is the durable analysis record for the team and future implementation work. It records why choices were proposed, what evidence is missing, and what should cause a choice to change. It contains engineering rationale, not hidden reasoning or unmeasured conclusions.

## Principal technical assessment

ReceiptSense is feasible as an academic adaptation of existing OCR and document models. The largest uncertainty is trustworthy downstream supervision, followed by OCR granularity/quality and access to a suitable training environment. The UI is relatively straightforward once the shared schema is stable.

The likely bottleneck is not merely getting a transformer to run. It is ensuring that field strings are aligned to the correct OCR units, boxes refer to the correct image, long receipts retain their totals, and experiments compare the same inputs. Address extraction often spans lines; totals often compete with semantically different amounts. Treat these as designed cases from the start.

## Risk register

| Risk | Consequence | Early evidence/check | Mitigation and fallback |
|---|---|---|---|
| Dataset labels/release differ from assumptions | Wrong training size or unusable test | Inventory and manual file audit | Register exact release; use documented local holdout if needed |
| Incorrect field-to-OCR alignment | Model learns wrong entities | Audited spans and per-field coverage | Manual review, masked/quarantined loss, versioned annotation policy |
| OCR line/word granularity misunderstood | Invalid geometry/supervision | Inspect native results and overlays | Freeze line-unit or documented replicated-box policy |
| OCR misses faded totals | Extraction cannot recover source text | Field coverage audit and image inspection | Review flag, quality retry, optional recognizer study later |
| Incorrect geometry transforms | Misleading evidence | Landmark round trips and overlays | Track frames; use processed image if inverse unavailable |
| Long receipt truncation | Bottom totals systematically lost | Token-length histogram and long fixture | Overlapping windows with global unit mappings |
| GPU/dependency constraints | Training delayed or impossible locally | Tiny forward/overfit and memory measurement | Separate OCR/train environments, modest batches, notebook GPU if available |
| Rules tuned to test examples | Inflated comparison | Run/configuration history | Freeze methods before test access |
| Review edits enter automatic scores | Inflated accuracy | Immutable prediction/review separation | Separate stores and evaluator inputs |
| Normalization hides mistakes | False exact matches | Strict edge cases and paired raw scores | Conservative frozen parsing, strict primary metric |
| Image correction removes text | Worse accuracy despite cleaner appearance | Before/after OCR and field results | Identity baseline; retain only validated variants |
| Phone photos differ from benchmark | Overstated practical performance | Separate real-photo set | State domain limit; collect authorized extra holdout if feasible |
| Too many alternative models | Core work unfinished | Milestone review | Defer BROS/TrOCR/Donut until minimum pipeline works |
| Overconfident review scores | Users accept wrong totals | Validation review-error analysis | Qualitative flags, named uncalibrated scores |

## Proposed decisions and revision triggers

| ID | Proposed decision | Rationale | Revisit when |
|---|---|---|---|
| D01 | Keep four required fields | Matches source and bounded evaluation | Supervisor explicitly expands scope |
| D02 | Build rules before transformer | Establishes schema and comparator quickly | Remains required even if model strong |
| D03 | Use canonical OCR cache for both methods | Separates extraction differences from OCR changes | Engine/processing version changes, requiring new experiments |
| D04 | Train main model with deployment OCR | Reduces gold-OCR versus real-OCR mismatch | Coverage is too poor; report oracle only as diagnostic |
| D05 | Use nine BIO tags | Simple token-classification formulation | Data needs a different verified span representation |
| D06 | First-subword supervision | Clear unit-to-label mapping | A controlled validation study supports another policy |
| D07 | Streamlit first, optional FastAPI | Provides a local end-to-end demo with less integration | A separate client/worker requires an API |
| D08 | Validation flags before rule overrides | Keeps learned comparison interpretable | Explicit override experiment is implemented and evaluated |
| D09 | Strict field EM primary, normalized EM secondary | Preserves visible string errors | A required official protocol supersedes it, with both documented |
| D10 | Screenshot blanks now, actual evidence later | Requested by user and avoids fabricated screenshots | Real build/experiment outputs exist |

These are recommendations recorded on 23 October 2026. They are not claims that the team already approved or implemented each choice.

## Open questions and safe working assumptions

| Question | Current working assumption | How to resolve |
|---|---|---|
| Which SROIE release is obtainable? | Unknown | Acquire and audit before fixing counts |
| Are test entity labels accessible? | Unknown | Inspect authorized release; define local holdout if necessary |
| What hardware can train the model? | Windows 11, 16 GB RAM, RTX 3050 4 GB detected; installed PyTorch uses CPU | Saved environment manifest and actual training mode |
| Is a submission format prescribed? | Markdown report is editable source | Check course template before final conversion |
| What is the final deadline? | No deadline supplied; six-week sample schedule | Team/supervisor milestone agreement |
| Is multilingual/phone-photo evaluation required? | Initial audited printed-receipt scope only | Confirm course expectations and collect suitable held-out data |
| Who owns each module? | Suggested role areas, not assigned individuals | Team agreement |

No open question prevents preparing this documentation pack. Dataset access and usable training resources will become concrete implementation gates.

## Change log template

```text
Decision ID:
Date and author:
Prior decision:
New decision:
Evidence/run IDs:
Affected schema/config/cache versions:
Does this invalidate an existing test result?:
Required reruns or confirmation data:
```

## Evidence policy

Distinguish source requirements, research findings, proposed designs, measured project results and user-corrected outputs. Cite original papers for borrowed methods; cite run IDs for project performance. A diagram or screenshot alone does not establish accuracy. A failed hypothesis remains useful when the experimental conditions and failure are recorded honestly.

## Decisions made during implementation, 23 October 2026

| ID | Implemented decision | Evidence and consequence |
|---|---|---|
| D11 | Default to Paddle-family PP-OCRv4 models through RapidOCR/ONNX Runtime | Native Paddle 3.3.1 oneDNN raised a PIR operator error on this Windows setup. Native non-oneDNN inference remains an explicit option; engine/model hashes distinguish outputs. |
| D12 | Provide CPU-ready frozen-backbone classifier-head adaptation and a separate full fine-tuning mode | CUDA is unavailable in the installed CPU PyTorch build despite the laptop's NVIDIA card. Saved training manifests identify the actual mode. No full fine-tuning result is inferred from head training. |
| D13 | Use partial exact-span weak supervision | A repeated amount can be cash. Unresolved units remain ignored; `O` labels require all four fields located. Negative signs and cash/subtotal contexts are protected by regression tests. |
| D14 | Accept up to 40 megapixels, retaining the 10 MB upload limit | The audited mirror includes legitimate scans above the proposed 20 MP default. Source bytes and evidence geometry are preserved. |
| D15 | Use variable-length text windows within the 512-token ceiling | Padding each short receipt to 512 wastes CPU attention work. Every real unit retains deterministic window ownership and long-receipt coverage. |
| D16 | Keep user corrections as immutable review versions and restore the latest review | UI/SQLite/export tests cover confirmation, subsequent edits, reversion and unchanged machine predictions. |
| D17 | Audit perceptual duplicate candidates without auto-removing matching templates | Both flagged cross-split pairs were distinct transactions; the audit records its heuristic scope and visual decisions. |

The exact executed model, evaluation, degradation and timing results belong in the generated implementation report and their saved JSON evidence. The screenshot companion remains editable for the team.
