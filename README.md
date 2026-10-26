# ReceiptSense - receipt extraction and review

Prepared: 23 October 2026. Source: [downloaded literature review](downloadfile.pdf), revised 10 October 2026.

ReceiptSense reads a printed receipt image and proposes **company, address, date, and total**, with image evidence, uncertainty warnings, user correction, and JSON/CSV export.

**Novelty branch:** selective cropped OCR recovery (maximum two extra calls) and multi-line address reconstruction are implemented as optional processing modes. See [the roadmap](docs/14_NOVELTY_ROADMAP.md) and [the measured novelty report](report/NOVELTY_REPORT.md). The existing trained head stays fixed. Fresh confirmation on 50 new Indian receipts remains pending team collection; development results are labelled separately.

## Fresh checkout

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[ml,dev,api]"
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Try the included synthetic examples with **Rules baseline**. Dataset images, runtime review records, and large model weights are excluded from Git. Follow [RUNNING.md](RUNNING.md) to reproduce training. The [saved results snapshot](results/README.md) supplies the measured evidence without installing a model. The trained run uses a frozen-backbone classifier head and does not improve overall strict accuracy over rules.

**Implementation:** the local application, shared OCR/extraction pipeline, review/export, optional API, dataset audit, training commands, and evaluation tools are now implemented. See [RUNNING.md](RUNNING.md) for startup and reproduction. Training/evaluation artifacts record their actual status; the earlier design guides remain below. Report screenshot areas are intentionally left for you to fill.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Open [ReceiptSense on this computer](http://127.0.0.1:8501). The default OCR runtime uses packaged Paddle-family models through ONNX; native Paddle remains an explicit option. The learned training mode and measured results are recorded with each model/run rather than inferred from the design.

## Start here

1. Read [the project explanation](docs/01_PROJECT_UNDERSTANDING.md) to understand the problem and deliverables.
2. Follow [PROJECT_PLAN.md](PROJECT_PLAN.md) for the implementation order and completion checks.
3. Read the individual modules below while implementing their corresponding steps.
4. Use [the literature guide](research/README.md) and [paper notes](research/LITERATURE_REVIEW.md) to connect each design choice to research.
5. Read [the measured implementation report](report/IMPLEMENTATION_REPORT.md), then add your screenshots to [the detailed report companion](report/PROJECT_REPORT.md).

## The whole project, divided into parts

| Document | What it explains |
|---|---|
| [00 - Source and requirements](docs/00_SOURCE_AND_REQUIREMENTS.md) | What the supplied PDF actually requests, additions made in this plan, traceability |
| [01 - Project understanding](docs/01_PROJECT_UNDERSTANDING.md) | Plain-language explanation, example, objectives, scope, vocabulary |
| [02 - Architecture and contracts](docs/02_ARCHITECTURE_AND_DATA_CONTRACTS.md) | How modules connect, coordinate frames, proposed schemas and interfaces |
| [03 - Dataset and annotation](docs/03_DATASET_AND_ANNOTATION.md) | SROIE audit, fixed splits, OCR-to-field labels, leakage prevention |
| [04 - Image preprocessing](docs/04_IMAGE_PREPROCESSING.md) | Safe image handling, deskew, perspective, preserving evidence coordinates |
| [05 - OCR and reading order](docs/05_OCR_AND_READING_ORDER.md) | Detection versus recognition, OCR granularity, boxes, caching, ordering |
| [06 - Rules baseline](docs/06_RULES_BASELINE.md) | Transparent candidate extraction, total versus cash, deterministic comparison |
| [07 - LayoutLMv3](docs/07_LAYOUTLMV3_TRAINING.md) | Processor inputs, BIO labels, subwords, long receipts, training and decoding |
| [08 - Validation and review](docs/08_VALIDATION_AND_HUMAN_REVIEW.md) | Parsing, abstention, confidence, evidence, correction audit trail |
| [09 - Application and export](docs/09_APPLICATION_AND_EXPORT.md) | Local UI, optional API, records, errors, JSON and CSV |
| [10 - Evaluation and experiments](docs/10_EVALUATION_AND_EXPERIMENTS.md) | Exact match, OCR diagnostics, fair ablations, robustness, timing |
| [11 - Engineering and reproducibility](docs/11_ENGINEERING_AND_REPRODUCIBILITY.md) | Proposed repository, setup gates, run artifacts, useful tests |
| [12 - Risks and decisions](docs/12_RISKS_AND_DECISIONS.md) | Open questions, technical analysis, fallback choices, decision log |
| [13 - Team, schedule and viva](docs/13_TEAM_SCHEDULE_AND_VIVA.md) | Six-week example schedule, suggested roles, explanations for review |
| [14 - Recovery and address novelty](docs/14_NOVELTY_ROADMAP.md) | Address graph, selective retries, evidence provenance, validation and fresh confirmation |

The documents are one connected specification: the schema in module 02 is shared by data, OCR, both extractors, the interface, and evaluation. Do not invent a separate format inside each module.

## What has been provided

- A dependency-ordered project plan with tasks, outputs, and acceptance gates.
- A runnable receipt review application, shared extraction pipeline, optional API, and automated checks.
- Pinned dataset/model downloads, audited splits, learned extractor training, and reproducible evaluation commands.
- Fourteen module documents, including the source-to-requirement map.
- Nine original paper PDFs, nine individual reading notes, a literature matrix, and a verified import manifest.
- An editable Markdown report with fourteen blank screenshot areas and empty experiment tables.
- A screenshot capture guide and a small utility for reproducing the paper import.

## Understand the implementation sequence

Read plan steps 1-4 to understand why the shared schema, dataset audit, fixed splits and a small OCR check come before training. The executed setup and evidence paths are documented in [RUNNING.md](RUNNING.md).

The working folder originally contained only the supplied PDF. The implementation is now in `src/`, `app/`, `api/`, `tests/`, and `tools/`. Earlier module documents preserve the design rationale; [RUNNING.md](RUNNING.md) and the measured report describe the executed system.

## Document conventions

- **Source requirement:** stated in the supplied PDF, with a page reference.
- **Proposed decision:** a concrete recommendation for implementation; revise with recorded evidence.
- **Pending:** must be established through an audit, experiment, or team decision.
- **Illustrative:** a constructed example, never a measured result.
- Checkbox completion in the plan means code and evidence exist; documentation alone does not complete implementation.

Use relative links within this pack so it stays portable. Keep the original PDF and research PDFs unchanged.
