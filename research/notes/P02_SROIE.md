# P02 - SROIE benchmark

[Library](../README.md) | [Original paper](https://arxiv.org/abs/2103.10213v1) | [Local PDF](../papers/P02_SROIE.pdf)

**Citation:** Zheng Huang et al., ICDAR, 2019; public preprint 2021. Imported version: v1.

## Central idea

The competition separates receipt text localization, recognition and key information extraction. Its receipt task motivates the company/address/date/total targets.

## What to read

Read dataset/annotation descriptions and the task-specific evaluation protocols before reading rankings. Inspect example annotation figures.

## ReceiptSense use

Audit an authorized dataset release, preserve its partition provenance, and distinguish OCR failures from extraction failures. See [data preparation](../../docs/03_DATASET_AND_ANNOTATION.md).

## Boundaries and questions

A pinned public SROIE mirror has now been imported; its observed counts, exclusions and hashes are in the dataset card. Complete field strings are not automatically token-level BIO labels. Project exact match must not be called official competition performance without matching the official evaluator.

**Status:** benchmark foundation; the implemented dataset/annotation pipeline records the actual mirror audit and supervision coverage.
