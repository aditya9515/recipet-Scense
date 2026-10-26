# Run ReceiptSense

The project now contains a local Streamlit application, shared OCR/extraction modules, SQLite review history, JSON/CSV export, an optional FastAPI service, and dataset/training/evaluation commands. This guide distinguishes the working implementation from the earlier design documents.

## Start the application

From the project folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Open [ReceiptSense](http://127.0.0.1:8501). Alternatively run `./run.ps1` when your PowerShell policy permits local scripts. The direct Python command does not require changing execution policy.

1. Choose **Try an example**, or upload one PNG/JPEG receipt.
2. Select Rules baseline or the trained LayoutLMv3 bundle when available.
3. Click **Read receipt**.
4. Select a field to highlight its source. Line-derived geometry is labelled coarse.
5. Edit values, then **Save changes** or **Confirm record**.
6. Download JSON/CSV. Unsaved changes disable export to prevent downloading stale values.

Example receipts are explicitly synthetic and are not benchmark evidence. The original machine prediction remains unchanged when you save a review. History reopens stored predictions; each saved review is a separate immutable version.

## Install on another machine

Use Python 3.10-3.13 with compatible wheels; the verified environment uses Python 3.12 on Windows 11.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[ml,dev,api]"
.\.venv\Scripts\python.exe -m receiptsense.cli samples
.\.venv\Scripts\python.exe -m receiptsense.cli doctor
```

`requirements.lock.txt`, when present, records the tested full environment. Install the package after the pinned dependencies. Native Paddle is optional and adds substantial dependencies; the default CPU OCR path uses packaged Paddle-family PP-OCRv4 ONNX models through RapidOCR. It does not upload receipts to an OCR service.

For native Paddle:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ocr]"
$env:RECEIPTSENSE_OCR_BACKEND = "paddle"
```

Use the ONNX default by clearing that environment variable or setting it to `onnx`. OCR backend changes create different cache fingerprints and require rebuilding training/evaluation records. Rules and learned methods must use the same fingerprint in each comparison.

## Why the implementation has two OCR paths

Native PaddleOCR is implemented and was tested. Its oneDNN path raised a Windows PIR operator error in the tested Paddle 3.3.1 environment; disabling oneDNN made it functional but slow. The default ONNX deployment uses Paddle-family models with an independently recorded recognizer and weight hashes. This is an explicit engineering adaptation of the design, not an assertion that both engines produce identical text.

The optional native adapter pins the detector/recognizer and a maximum detection side of 960 pixels rather than relying on changing library defaults. The ONNX engine records its packaged model hashes and package versions. Both use the same canonical contract and preserve approximate line-derived boxes honestly.

## Acquire the pinned research assets

```powershell
.\.venv\Scripts\python.exe tools/download_assets.py --kind all
.\.venv\Scripts\python.exe -m receiptsense.cli prepare
```

The downloader verifies hashes of both public SROIE parquet archives and the base model. It uses bounded range requests and rejects incomplete files. The data comes from the pinned `jsdnrs/ICDAR2019-SROIE` mirror; its LICENSE, NOTICE and source card are retained in `data/raw/`. Source revision, observed counts, exclusions and fixed split hashes are written to `data/dataset_card.json`.

`prepare` removes byte-identical training copies of held-out receipts, deduplicates within supplied partitions and excludes missing reference entities before model development. Therefore the resulting counts can differ from counts advertised for other SROIE releases. This is a documented mirror-derived evaluation split, not official leaderboard reproduction.

## Prepare real OCR and training labels

```powershell
.\.venv\Scripts\python.exe -m receiptsense.cli ocr data/manifests/train.jsonl
.\.venv\Scripts\python.exe -m receiptsense.cli ocr data/manifests/validation.jsonl
.\.venv\Scripts\python.exe -m receiptsense.cli ocr data/manifests/test.jsonl
.\.venv\Scripts\python.exe -m receiptsense.cli align data/manifests/train.jsonl
.\.venv\Scripts\python.exe -m receiptsense.cli align data/manifests/validation.jsonl
```

OCR is cached by image/processing/engine fingerprint. Reference OCR from the mirror is retained for provenance, but these commands use the deployed OCR engine. Alignment uses normalized exact spans. On partially aligned receipts it supervises only unambiguous positive spans and ignores every other unit; it labels unrelated units `O` only when all four fields are located unambiguously. Conflicting/unusable records are quarantined. Labels remain weak supervision until audited. Coverage summaries record usable and fully aligned receipts; they must not be described as manually verified labels.

The training command rejects test alignment and overlapping training/validation IDs. Final field evaluation includes unalignable receipts and failed predictions rather than dropping them.

## Train LayoutLMv3

The memory-conscious default adapts a nine-label classifier head on cached, frozen LayoutLMv3 features. It is a real learned extractor, but it is **classifier-head adaptation**, not all-parameter fine-tuning. The saved manifest identifies which was performed.

```powershell
.\.venv\Scripts\python.exe -m receiptsense.cli train --mode head --epochs 40
```

Only usable training alignments enter the optimizer. Checkpoint selection uses complete-field exact match across the full validation manifest. The bundle includes weights, processor, label mapping, training/validation fingerprints, weak-label policy, history and selection evidence.

To perform all-parameter fine-tuning later in a suitable environment:

```powershell
.\.venv\Scripts\python.exe -m receiptsense.cli train --mode full --epochs 10 --learning-rate 0.00002 --output artifacts/models/receiptsense-full
```

Existing trained bundles are not implicitly overwritten. Use a fresh output directory for a new experiment. Install a compatible CUDA PyTorch build from the [official PyTorch installer](https://pytorch.org/get-started/locally/) if you want GPU training; the local tested PyTorch build is CPU-only. A detected NVIDIA card alone does not mean CUDA is available to Python.

## Evaluate and inspect

```powershell
.\.venv\Scripts\python.exe -m receiptsense.cli evaluate data/manifests/test.jsonl --name held-out-v1
```

Each run saves per-receipt predictions, strict/normalized results, missing/failure counts, paired receipt-bootstrap differences and cached-extraction timing. Test labels are used for scoring after extraction. This timing excludes OCR and startup and is labelled accordingly. The app's **Experiments** page reads these saved results.

For the fixed paired correction/degradation probe and actual uncached timing:

```powershell
.\.venv\Scripts\python.exe -m receiptsense.cli robustness data/manifests/test.jsonl --limit 12 --name robustness-v1
.\.venv\Scripts\python.exe -m receiptsense.cli benchmark data/manifests/test.jsonl --limit 10
```

The probe samples parent receipts with seed 42. All conditions share a maximum-side-1600 clean reference; it separately tests automatic deskew, blur, padded rotation, contrast and perspective. These small-subset results must be reported separately from the full held-out comparison. Parameters are fixed before final evaluation and must not be tuned on the probe's results.

For a quick development run, use the validation manifest and a new run name. `--limit` selects an explicitly marked subset; never report it as the full split. Evaluate a different bundle with `--checkpoint artifacts/models/receiptsense-full`.

## Single-image CLI

```powershell
.\.venv\Scripts\python.exe -m receiptsense.cli predict examples/receipt_clear.png --method rules
.\.venv\Scripts\python.exe -m receiptsense.cli predict examples/receipt_clear.png --method layoutlmv3
```

Add `--no-cache` for real OCR timing or `--variant deskew` / `--variant contrast` for explicit preprocessing variants. A base checkpoint with an untrained classification head is rejected by the application inference loader.

## Optional local API

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open [interactive API documentation](http://127.0.0.1:8000/docs). Upload at `POST /receipts`, extract at `POST /receipts/{id}/extract`, retrieve predictions, save reviews and export the chosen reviewed version. The service is bound to localhost. It has no public-hosting authentication layer and is intended for the local project demonstration.

## Checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src app api tests tools/download_assets.py
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m tools.smoke_api
```

The tests exercise total-versus-cash selection, geometry transforms, ambiguous dates, decimal parsing, repeated-label quarantine, long-receipt ownership, missing-output denominators, immutable predictions, SQLite cleanup/rollback, review/CSV/JSON behavior, API routes and Streamlit pages. The additional API smoke command uses actual OCR on a constructed receipt with an isolated temporary review database.

## Main implementation files

| Part | Location |
|---|---|
| Local application | `app/streamlit_app.py` |
| Shared contracts and pipeline | `src/receiptsense/contracts.py`, `pipeline.py` |
| Image/coordinates and OCR | `imaging.py`, `ocr_adapter.py`, `reading_order.py` |
| Extraction and parsing | `rules.py`, `encoding.py`, `model.py`, `decoding.py`, `validation.py` |
| Review and export | `review_store.py`, `export.py` |
| Dataset and training | `dataset.py`, `alignment.py`, `training.py` |
| Experiments | `evaluation.py` |
| API | `api/main.py` |
| Design/report/papers | `docs/`, `report/`, `research/` |

Runtime receipts, review database, weights, caches and experiment results are under `artifacts/` and excluded from ordinary Git commits. Screenshot blanks in the original report remain available for you to fill.

## Address reconstruction and selective recovery

Choose **Processing mode** in the application. Baseline (`none`) remains the default. The other modes are `address`, `retry`, and `combined`; the selected extractor remains Rules or the existing frozen-head LayoutLMv3 bundle. No full-model training occurs during recovery.

```powershell
.\.venv\Scripts\python.exe -m receiptsense.cli predict examples/receipt_ambiguous.png --enhancement combined
.\.venv\Scripts\python.exe -m receiptsense.cli predict examples/receipt_clear.png --method layoutlmv3 --enhancement address
.\.venv\Scripts\python.exe -m tools.smoke_novelty
```

The API extraction request accepts `"enhancement": "combined"` alongside `method` and `variant`. JSON exports retain the baseline fields and recovery trace. CSV adds `processing_mode`, `logical_retry_attempts`, and `uncached_retry_ocr_calls`. Old records still load and reopen. A fresh checkout with only model metadata does not advertise an installed learned bundle.

Recovery permits at most two logical attempts. Cache hits are reported separately from actual OCR calls. Failed or conflicting retries preserve the original suggestion and review reasons. The UI shows before/after values, crop regions, and accept/reject decisions. Scores are heuristics, not calibrated probabilities.

## Reproduce the novelty experiments

```powershell
.\.venv\Scripts\python.exe -m receiptsense.cli novelty-select data/manifests/validation.jsonl
.\.venv\Scripts\python.exe -m receiptsense.cli novelty-evaluate data/manifests/validation.jsonl --name validation-new-run
.\.venv\Scripts\python.exe -m receiptsense.cli novelty-benchmark data/manifests/validation.jsonl --name validation-new-run --limit 10
.\.venv\Scripts\python.exe -m pytest -q --junitxml=artifacts/novelty/test-results.xml
```

Selection searches only the predeclared graph grid on validation. Each accuracy run compares both extractors across all four modes. Use a fresh run name; completed evidence is not implicitly overwritten. Uncached latency is a separate experiment. The report generator `tools/build_novelty_report.py` reads the delivered `validation-v1` run and verifies its frozen policy before publishing an explicit aggregate allowlist. Run `python tools/check_novelty_invariants.py` first to verify all 488 saved predictions, the retry budget, evidence resolution and unchanged baseline fields; these audit scripts require the actual runtime artifacts.

## Collect and freeze fresh confirmation receipts

```powershell
.\.venv\Scripts\python.exe -m receiptsense.cli confirmation-init
```

This creates ignored `data/confirmation/`, containing instructions, `images/`, `collection.csv`, two annotation CSVs, adjudications and exclusions. Collect 50 English/Latin-script Indian printed receipts from ten merchants (five distinct transactions each), with all four fields visible. Select before inspecting predictions. Two people independently fill the four reference strings using distinct reviewer IDs. A third reviewer must resolve disagreements with a reason. Keep verbatim receipt text and do not silently choose an interpretation for an ambiguous date.

After the validation policy is fixed and both annotation versions are ready:

```powershell
.\.venv\Scripts\python.exe -m receiptsense.cli confirmation-freeze --name confirmation-v1
.\.venv\Scripts\python.exe -m receiptsense.cli confirmation-evaluate data/confirmation/frozen/confirmation-v1/manifest.jsonl --lock data/confirmation/frozen/confirmation-v1/lock.json --name confirmation-v1
```

Freeze rejects missing labels, annotation disagreement without adjudication, wrong merchant counts, duplicate images, overlap with development images, unsupported files, or missing validation evidence. Evaluation verifies hashes of receipts, annotations, manifest, policy/configuration, OCR backend, evaluation protocol and checkpoint before inference. The tool verifies supplied annotations; it cannot prove reviewer independence or collection timing.

Receipt images, transaction-level labels, merchant mappings, predictions and locks stay local. Only aggregate metrics and explicitly approved redacted examples should be published. No fresh confirmation result is supplied with this branch; synthetic test fixtures are not confirmation data. Do not tune the method after inspecting confirmation outcomes and still call the same set untouched.
