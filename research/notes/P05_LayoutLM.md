# P05 - LayoutLM

[Library](../README.md) | [Original paper](https://arxiv.org/abs/1912.13318v5) | [Local PDF](../papers/P05_LayoutLM.pdf)

**Citation:** Yiheng Xu et al., KDD, 2020. Imported version: v5.

## Central idea

LayoutLM jointly models text and document layout and incorporates visual word information for document understanding.

## What to read

Study the representation diagram and two-dimensional position embeddings, then compare text-only and layout-aware task formulations.

## ReceiptSense use

Keep boxes and geometry instead of flattening a receipt into a single text string. This provides background for the selected LayoutLMv3 family. See [contracts](../../docs/02_ARCHITECTURE_AND_DATA_CONTRACTS.md).

## Boundaries and questions

LayoutLM is not an additional mandatory baseline. A fair comparison would need a separately implemented, trained and evaluated model under matching OCR/splits. Ask what useful information disappears when OCR regions are flattened or reordered.

**Status:** architecture background; LayoutLMv3 remains the required learned extractor.
