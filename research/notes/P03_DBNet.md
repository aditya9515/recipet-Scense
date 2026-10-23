# P03 - DBNet

[Library](../README.md) | [Original paper](https://arxiv.org/abs/1911.08947v2) | [Local PDF](../papers/P03_DBNet.pdf)

**Citation:** Minghui Liao et al., AAAI, 2020; preprint 2019. Imported version: v2.

## Central idea

Differentiable binarization makes a segmentation-based text detector learn an adaptive threshold along with text-region predictions.

## What to read

Inspect the detector diagram, threshold/binarization explanation and detection evaluation. Separate text localization outputs from recognized characters.

## ReceiptSense use

Understand why region polygons must be retained and why detection misses limit extraction. Use a pretrained OCR detector initially; record the actual installed checkpoint. See [OCR](../../docs/05_OCR_AND_READING_ORDER.md).

## Boundaries and questions

This citation does not mean every PaddleOCR detector is the original DBNet. We do not train DBNet from scratch. Ask whether a missed total is absent from detection or present but misrecognized; the remedies differ.

**Status:** conceptual detection foundation, not a separate required implementation.
