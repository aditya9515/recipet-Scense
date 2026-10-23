# P08 - Donut

[Library](../README.md) | [Original paper](https://arxiv.org/abs/2111.15664v5) | [Local PDF](../papers/P08_Donut.pdf)

**Citation:** Geewook Kim et al., ECCV, 2022. Imported version: v5.

## Central idea

Donut maps document images to structured text using an OCR-free transformer, addressing dependencies and error propagation from an external OCR stage.

## What to read

Inspect the architecture, structured output formulation and task adaptations. Examine how its data and evaluation differ from OCR-based pipelines.

## ReceiptSense use

Understand an alternative system design and why OCR quality can constrain extraction. It is a possible later comparison, not part of the required pipeline.

## Boundaries and questions

ReceiptSense prioritizes explicit OCR regions for inspection. An OCR-free alternative needs its own trained output schema and a considered evidence strategy. Do not assume it eliminates recognition errors or automatically supplies word-box highlights.

**Status:** optional future direction after the core system and evaluation are complete.
