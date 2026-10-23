# 11 - Engineering setup, artifacts, and reproducibility

[Index](../README.md) | Previous: [evaluation](10_EVALUATION_AND_EXPERIMENTS.md) | Next: [risks](12_RISKS_AND_DECISIONS.md)

## Current workspace versus proposed implementation

The initial version of this document specified the implementation before code existed. The application and offline commands are now implemented; [RUNNING.md](../RUNNING.md) records their actual use. The layout below remains the design reference; the trained model/run manifests establish what was executed.

## Proposed repository layout

```text
ReceiptSense/
  README.md, PROJECT_PLAN.md
  docs/                         # This knowledge pack
  research/papers/, research/notes/, paper_manifest.json
  report/PROJECT_REPORT.md, assets/
  src/receiptsense/
    contracts.py                # Shared records/validation
    imaging.py                  # Image handling and transform chains
    ocr_adapter.py              # Native output -> canonical OCR
    reading_order.py
    dataset.py, alignment.py    # Inventory/splits/training labels
    rules.py
    encoding.py, training.py    # LayoutLMv3 preparation and training
    decoding.py, validation.py
    pipeline.py, review_store.py, export.py
    evaluation.py
  app/streamlit_app.py
  api/main.py                   # Optional
  configs/                      # Versioned OCR/train/metric configs
  data/raw/, manifests/, processed/   # Respect distribution terms
  artifacts/ocr/, checkpoints/, runs/
  tests/fixtures/                # Tiny synthetic/allowed fixtures
  tools/download_papers.py       # Exists now
  requirements.lock.txt         # Create after environment validation
```

Do not create empty implementation files simply to make the repository appear complete. Add modules as functionality is built.

## 1. Establish a working environment

1. Record OS, Python availability, CPU/RAM and any usable GPU/VRAM.
2. Select a Python version supported by the chosen Paddle and PyTorch wheels on the actual OS.
3. Use an isolated environment rather than modifying the system interpreter.
4. Consult official [PyTorch setup](https://pytorch.org/get-started/locally/), [PaddleOCR installation](https://www.paddleocr.ai/main/en/version3.x/installation.html), and [Transformers installation](https://huggingface.co/docs/transformers/installation) for the installed platform/version. Those pages are setup entry points; a compatible stack has not been validated here.
5. Prove image decode, one OCR result, one model forward pass and one saved/reloaded tiny checkpoint before freezing dependencies.
6. Record exact versions and model revisions in a lock/manifest.

Proposed libraries: Pillow/NumPy/OpenCV for images, PaddleOCR with its selected inference engine for OCR, PyTorch/Transformers for the extractor, pandas or standard libraries for tables, Streamlit for UI, SQLite for reviews, and pytest for meaningful invariants. FastAPI is optional. Select only what the implemented workflow needs.

Paddle and PyTorch can contend for GPU memory or have incompatible dependencies. If needed, run OCR as a separate process/environment producing the same canonical cache. A cloud notebook can train the extractor from that cache, with paths/configuration portable and artifacts exported back locally.

Do not copy the original research repository's historical CUDA/Python installation commands blindly into this Windows setup. Use a supported tested stack for the chosen implementation.

## 2. Proposed run entry points

Implement a small set of explicit commands/functions for these operations:

| Operation | Required inputs | Required saved output |
|---|---|---|
| Inventory | Dataset root, source metadata | Inventory, exclusion ledger |
| Split | Inventory, split policy/seed | ID manifests and hashes |
| OCR | Manifest, image/OCR config | Versioned canonical OCR cache |
| Align | Train/validation OCR + reference entities | Labels, coverage, review queue |
| Baseline | OCR manifest + rules config | Field prediction JSONL |
| Train | Labelled data + model config | Checkpoint bundle, logs |
| Predict | Checkpoint + OCR/image manifests | Immutable predictions |
| Evaluate | Predictions + references + metric config | Summary and error rows |
| Demo | Model bundle + UI config | Local interactive workflow |

These are proposed command responsibilities, not commands already implemented. Document exact invocation once real entry points exist.

## 3. Run manifest

Every experiment should include a run ID and manifest with:

```text
created_at, source_revision, environment_lock_hash,
dataset_release, split_manifest_hash, annotation_policy_hash,
preprocessing_config_hash, OCR_fingerprint,
model_checkpoint_revision, trained_checkpoint_hash,
rules_revision, decoding_revision, validation_revision,
metric_policy_revision, random_seed, device,
parent_run_id, status, failure_reason
```

Store configuration, logs, per-receipt predictions, metrics and failures together. Never identify an experiment only as “final model”. A file named `best.pt` needs its selection policy and validation evidence.

## 4. Tests worth writing

Focus on high-impact invariants and integration behavior:

- Annotation parsing preserves commas in receipt text.
- Split/duplicate groups never cross partitions.
- Known geometry transforms map evidence to the correct image.
- OCR records validate IDs, frames and box/text lengths.
- Annotation matching does not choose subtotal when the same amount repeats.
- Word/subword/window mappings retain a bottom-of-receipt total.
- Field metrics have known strict/normalized edge-case outcomes and correct denominators.
- User edits never mutate predictions; exports retain correct receipt/review IDs.
- A tiny end-to-end fixture reaches export with evidence IDs intact.

Do not add hundreds of checks that merely repeat a helper's implementation. Inspect real receipt overlays and errors in addition to tests. Actual application verification is recorded by the pytest JUnit artifact and implementation report, rather than inferred from this test plan.

## 5. Data and artifact management

Version source/configuration/documents in Git if the team adopts it. Keep large datasets, raw user receipts, caches and checkpoints out of routine commits; document their acquisition/storage paths and hashes. Keep allowed small synthetic fixtures in the repository.

Use relative project paths in configs, support explicit dataset/artifact roots, and avoid hardcoded personal directories in the future app. Record UTF-8 encoding; Windows paths and console encoding need particular care.

Downloads in this pack are unmodified public paper PDFs for reading. Keep publisher attribution and notices. Their presence does not authorize arbitrary redistribution of models or datasets.

## 6. Clean reproduction gate

In a new environment/session: recreate dependencies, obtain the documented data, verify manifests, load the trained bundle, process a receipt, open evidence, edit/export, and reproduce a saved evaluation summary. Record any nondeterminism and reasonable numerical tolerance.

Supply final setup instructions with actual commands and measured constraints only after this gate is run.
