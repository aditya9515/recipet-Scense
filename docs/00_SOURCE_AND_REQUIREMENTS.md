# 00 - Source interpretation and requirements

[Index](../README.md) | [Plan](../PROJECT_PLAN.md)

## Source identity

The supplied [downloadfile.pdf](../downloadfile.pdf) is a five-page **ReceiptSense: Brief Literature Review**, revised 10 October 2026, for Document Image Analysis at Manipal Academy of Higher Education. It explicitly describes proposed work and says results have not yet been measured. It is a project proposal/literature review, not a dataset, runnable repository, trained checkpoint, or completed experimental report.

| Page | Main information |
|---|---|
| 1 | Four fields, receipt image input, supporting regions, correction/export, team |
| 2 | LayoutLMv3 as base model, SROIE benchmark, word/box/image inputs |
| 3 | DBNet/CRNN foundations, LayoutLM/BROS spatial motivation |
| 4 | Optional TrOCR/Donut, related hybrid work, proposed local architecture |
| 5 | Controlled comparison, image degradations, literature references |

## Source requirements and traceability

| ID | Requirement | Source | Design document | Planned proof |
|---|---|---|---|---|
| R01 | Accept scanned/photographed printed receipts | p.1 | [04](04_IMAGE_PREPROCESSING.md), [09](09_APPLICATION_AND_EXPORT.md) | Upload and validation demonstration |
| R02 | Extract company, address, date, total | pp.1-2 | [02](02_ARCHITECTURE_AND_DATA_CONTRACTS.md), [06](06_RULES_BASELINE.md), [07](07_LAYOUTLMV3_TRAINING.md) | Per-field exact-match table |
| R03 | Show supporting image regions | p.1 | [05](05_OCR_AND_READING_ORDER.md), [08](08_VALIDATION_AND_HUMAN_REVIEW.md) | Field-to-region screenshots and coordinate checks |
| R04 | Flag uncertainty and accept corrections | p.1 | [08](08_VALIDATION_AND_HUMAN_REVIEW.md), [09](09_APPLICATION_AND_EXPORT.md) | Before/after correction audit |
| R05 | Export a structured record | pp.1,4 | [09](09_APPLICATION_AND_EXPORT.md) | Valid JSON and CSV samples |
| R06 | Adapt pretrained LayoutLMv3, retaining OCR/layout/image | p.2 | [07](07_LAYOUTLMV3_TRAINING.md) | Training config and reloadable checkpoint |
| R07 | Audit SROIE and fix train/validation/test partitions | p.2 | [03](03_DATASET_AND_ANNOTATION.md) | Dataset card, ID manifests and leakage checks |
| R08 | Reuse pretrained OCR; record actual detector/recognizer | p.3 | [05](05_OCR_AND_READING_ORDER.md) | OCR engine/checkpoint manifest |
| R09 | Compare rules and learned extraction with shared OCR | p.4 | [06](06_RULES_BASELINE.md), [10](10_EVALUATION_AND_EXPERIMENTS.md) | Paired experiment records |
| R10 | Evaluate correction and selected image degradations | p.5 | [04](04_IMAGE_PREPROCESSING.md), [10](10_EVALUATION_AND_EXPERIMENTS.md) | Ablation/robustness tables |
| R11 | Report only measured improvement, inspect OCR failures | pp.1,5 | [10](10_EVALUATION_AND_EXPERIMENTS.md), [report](../report/PROJECT_REPORT.md) | Saved outputs and error analysis |

## Proposed engineering decisions

The source does not prescribe annotation matching rules, BIO labels, split percentages, training settings, exact-match normalization, API routes, database schema, review thresholds, screenshot count, or schedule. Those details in this pack are **proposed decisions**, not quotations from the PDF.

The plan recommends Streamlit first, an optional FastAPI boundary, SQLite for local review records, JSONL for experiment outputs, a nine-label BIO classifier, and field-level exact match as the main project metric. Dataset counts and hardware assumptions remain pending audit. The six-week schedule is an adjustable planning example.

The PDF mentions OpenCV, PaddleOCR, PyTorch/Transformers, FastAPI/Streamlit, SQLite, JSON/CSV, and tests in its architecture illustration. Their precise versions and integration still need to be established.

## Required versus optional

Required: four-field extraction, OCR evidence, a rules comparator, LayoutLMv3 adaptation, fixed held-out evaluation, review/export, and reproducibility.

Optional: BROS implementation, TrOCR replacement recognizer, Donut comparison, line items, multilingual support, public hosting, mobile capture, external accounting integrations, large-scale pretraining. Read these papers without turning each into a mandatory implementation.

## Source limitations to resolve

1. SROIE is named, but the actual downloaded release and accessible label counts are unknown.
2. Field strings and OCR boxes do not automatically provide token-level BIO labels.
3. A pretrained base checkpoint needs a trained classification head and downstream adaptation.
4. OCR pipeline defaults may change; citing DBNet/CRNN does not identify installed models.
5. Confidence-based review needs a documented policy and validation, not an arbitrary accuracy claim.
6. Scanned-receipt benchmark performance does not establish performance on arbitrary phone photographs.

These gaps are addressed by explicit audit and experiment gates in the plan.
