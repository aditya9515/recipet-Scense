# ReceiptSense research library

[Project index](../README.md) | [Literature synthesis](LITERATURE_REVIEW.md) | [Verification manifest](paper_manifest.json)

Imported: 23 October 2026. All nine papers referenced by the supplied literature review have been downloaded as original PDFs. Each file was checked for PDF signature, readable pages, expected title on its first page, byte length and SHA-256. These checks establish file identity/integrity, not scientific reproducibility of the authors' experiments.

## Papers and individual reading notes

| ID | Paper / year | Local original | Individual note | Role |
|---|---|---|---|---|
| P01 | LayoutLMv3, ACM Multimedia 2022 | [PDF](papers/P01_LayoutLMv3.pdf) | [Note](notes/P01_LayoutLMv3.md) | Main base paper |
| P02 | SROIE, ICDAR 2019; preprint 2021 | [PDF](papers/P02_SROIE.pdf) | [Note](notes/P02_SROIE.md) | Benchmark/task |
| P03 | DBNet, AAAI 2020; preprint 2019 | [PDF](papers/P03_DBNet.pdf) | [Note](notes/P03_DBNet.md) | Text detection foundation |
| P04 | CRNN, TPAMI 2017; preprint 2015 | [PDF](papers/P04_CRNN.pdf) | [Note](notes/P04_CRNN.md) | Text recognition foundation |
| P05 | LayoutLM, KDD 2020 | [PDF](papers/P05_LayoutLM.pdf) | [Note](notes/P05_LayoutLM.md) | Text-layout background |
| P06 | BROS, AAAI 2022; preprint 2021 | [PDF](papers/P06_BROS.pdf) | [Note](notes/P06_BROS.md) | Spatial/order analysis; optional model |
| P07 | TrOCR, AAAI 2023; preprint 2021 | [PDF](papers/P07_TrOCR.pdf) | [Note](notes/P07_TrOCR.md) | Optional recognition alternative |
| P08 | Donut, ECCV 2022 | [PDF](papers/P08_Donut.pdf) | [Note](notes/P08_Donut.md) | Optional OCR-free alternative |
| P09 | Joint receipt transformer/rules, MAKE 2025 | [PDF](papers/P09_Joint_Receipt_Model.pdf) | [Note](notes/P09_Joint_Receipt_Model.md) | Closely related hybrid work |

The numbers correspond to references [1]-[9] in the source PDF. The notes are original concise explanations for this project; the PDFs retain their original authors' content and notices.

## Reading sequence

1. P02: understand what is annotated and evaluated.
2. P01: understand the main learned extractor.
3. P09: understand related receipt integration and boundaries on originality claims.
4. P03/P04: understand OCR's separate stages.
5. P05/P06: understand layout and reading-order decisions.
6. P07/P08: consider alternatives only after the required pipeline works.

For each paper, write down its actual inputs, outputs, training data, metric definition, and which part of our proposed system borrows its idea. A published receipt score is not automatically comparable to our end-to-end field metric.

## Imported versions and provenance

Eight downloads use explicitly versioned arXiv PDFs: P01 v3, P02 v1, P03 v2, P04 v1, P05 v5, P06 v5, P07 v5, P08 v5. P09 is the publisher PDF from MDPI's content host. The manifest records exact source URLs, versions, hashes, sizes, page counts and retrieval timestamps.

The publisher landing page for P09 rate-limited/blocked direct access during research; the publisher's PDF content host supplied the validated original successfully. During subsequent implementation, a pinned SROIE mirror and base model were imported separately. Their provenance and checksums are in the runtime asset manifests; see [RUNNING.md](../RUNNING.md). The nine papers remain unchanged.

## Reproduce the import

With Python and `pypdf` installed, run from the project root:

```powershell
python tools/download_papers.py
```

The utility reuses valid existing PDFs, validates them, and refreshes the manifest. Failed requests are recorded and produce a failing exit status. It does not download training data or models.

## Additional official implementation references

- [Transformers LayoutLMv3 documentation](https://huggingface.co/docs/transformers/model_doc/layoutlmv3) - processor/model interfaces, verified during research.
- [Microsoft base checkpoint card](https://huggingface.co/microsoft/layoutlmv3-base) - checkpoint identity and listed terms, verified during research.
- [Original LayoutLMv3 repository](https://github.com/microsoft/unilm/tree/master/layoutlmv3) - research code entry point; historical installation examples require adaptation.
- [PaddleOCR pipeline documentation](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html) - actual configurable OCR pipeline, verified during research.
- [SROIE competition portal](https://rrc.cvc.uab.es/?ch=13) - dataset/task entry point; access was unavailable to the research fetcher, so no release contents are asserted.
