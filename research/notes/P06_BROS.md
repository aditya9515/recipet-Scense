# P06 - BROS

[Library](../README.md) | [Original paper](https://arxiv.org/abs/2108.04539v5) | [Local PDF](../papers/P06_BROS.pdf)

**Citation:** Teakgyu Hong et al., AAAI, 2022. Imported version: v5.

## Central idea

BROS emphasizes text and relative two-dimensional layout without relying on image features. It examines incorrect text ordering and limited downstream examples.

## What to read

Read the spatial representation and area-masking explanation, then the reading-order analysis. Check the exact dataset variant used in comparisons.

## ReceiptSense use

Motivate careful line ordering and spatial diagnostics. A future text-plus-layout comparison could help study the role of image information. See [OCR/order](../../docs/05_OCR_AND_READING_ORDER.md).

## Boundaries and questions

No BROS baseline is claimed unless it is actually adapted and evaluated. Keep OCR, labels, splits and training budget comparable. Ask whether a failure persists when region order changes but geometry stays fixed.

**Status:** supporting literature; implementation optional after core milestones.
