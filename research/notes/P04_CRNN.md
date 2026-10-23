# P04 - CRNN sequence recognition

[Library](../README.md) | [Original paper](https://arxiv.org/abs/1507.05717v1) | [Local PDF](../papers/P04_CRNN.pdf)

**Citation:** Baoguang Shi, Xiang Bai, Cong Yao; IEEE TPAMI, 2017; preprint 2015. Imported version: v1.

## Central idea

The model combines convolutional features, recurrent sequence modeling and CTC transcription to recognize variable-length image text without individual character segmentation.

## What to read

Follow the architecture from text image to sequence, then study why CTC handles alignment. Identify assumptions about input crops.

## ReceiptSense use

Understand recognition as a distinct stage after text detection and diagnose character errors independently of field meaning. See [OCR](../../docs/05_OCR_AND_READING_ORDER.md).

## Boundaries and questions

We initially reuse a pretrained recognizer, not a newly trained CRNN. The selected PaddleOCR recognizer may use a different architecture. Even perfect transcription does not distinguish total from cash. Check whether numerical mistakes arise from crop quality or recognition.

**Status:** foundational reading; no separate CRNN training planned.
