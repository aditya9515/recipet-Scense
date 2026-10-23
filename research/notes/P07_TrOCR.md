# P07 - TrOCR

[Library](../README.md) | [Original paper](https://arxiv.org/abs/2109.10282v5) | [Local PDF](../papers/P07_TrOCR.pdf)

**Citation:** Minghao Li et al., AAAI, 2023; preprint 2021. Imported version: v5.

## Central idea

TrOCR uses an image-transformer encoder and text-transformer decoder for text recognition, with pretraining and task adaptation.

## What to read

Study the encoder-decoder input/output and the recognition evaluation. Check whether the selected checkpoint expects printed or handwritten text and the appropriate crop type.

## ReceiptSense use

Consider it later as a replacement recognizer on the same detected crops if recognition errors dominate. Compare transcription and final field extraction separately.

## Boundaries and questions

TrOCR does not by itself locate all receipt text or assign company/address/date/total. Changing recognizers changes OCR text, caches and possibly supervision, so it is not a free drop-in claim of better extraction.

**Status:** optional future recognition experiment; not required for the initial build.
