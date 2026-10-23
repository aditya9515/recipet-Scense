# 05 - OCR, region geometry, and reading order

[Index](../README.md) | Previous: [preprocessing](04_IMAGE_PREPROCESSING.md) | Next: [rules](06_RULES_BASELINE.md)

## What this module does

Detection locates text. Recognition reads it. The OCR adapter preserves both in a stable receipt record. Input: selected image in a known frame. Output: text units, polygons/boxes, engine scores, line/order metadata, and an engine fingerprint.

Read [DBNet P03](../research/notes/P03_DBNet.md) for detection and [CRNN P04](../research/notes/P04_CRNN.md) for sequence recognition. These are foundations, not a claim about the internal architecture of an installed PaddleOCR release.

## Freeze actual models, not a library name

Select and test a pretrained PaddleOCR configuration. Save package versions, inference backend/device, language, detector name/revision, recognizer name/revision, orientation/unwarping switches, score thresholds and adapter revision. Cache or locally reference model artifacts for reproducibility.

The [official PaddleOCR usage guide](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html) documents a configurable detection/recognition pipeline with optional orientation and unwarping stages. Choose explicit settings and inspect actual output fields; defaults and APIs differ between versions. Avoid copying a legacy call signature into a newer environment without a smoke test.

## Adapt and validate the output

1. Preserve the native text region and engine index.
2. Determine whether it is a word or line. Verify on several examples rather than inferring granularity from a sample string.
3. Retain the polygon, raw text, recognition score, and chosen image frame.
4. Derive an axis-aligned box from polygon minima/maxima for the transformer.
5. Check valid bounds, positive area, Unicode, duplicate IDs, and finite scores.
6. Create stable receipt-scoped unit IDs and save native-to-canonical mapping.
7. Group units into lines and assign deterministic reading order.

Do not discard all low-score regions before examining their effect; faint totals can be important. A configurable threshold, if used, becomes part of the experiment. Keep a rejected-region record to diagnose missing evidence.

## Reading-order reconstruction

Proposed initial algorithm:

1. Work in the processed image where any deskew has been applied.
2. Estimate region/line heights from geometry.
3. Group nearby regions by vertical overlap or a center-distance tolerance scaled by typical text height.
4. Sort lines top to bottom, then units left to right within each line.
5. Preserve a deterministic tie-breaker using native index.
6. Inspect columns, date/time rows, and label-amount rows manually on training examples.

Do not sort globally by x then y. For a row `TOTAL ... 493.50`, the left label and right amount should remain neighbors. Complex columns may need additional grouping; keep that logic versioned and evaluated. [BROS P06](../research/notes/P06_BROS.md) motivates examining spatial relations and order sensitivity.

## Granularity policy

Use real word boxes when the engine supports and returns them reliably. If output is line-based, adopt a documented line-unit model or split text using replicated line boxes as described in [03](03_DATASET_AND_ANNOTATION.md). Retain line geometry for evidence even when text is split.

A replicated line box does not identify the exact characters selected within that line. Show coarse evidence honestly in the UI. Geometry changes invalidate field alignment and model inputs, so freeze this policy before generating the labelled training corpus.

## Cache design

```text
cache_key = hash(
  source_image_hash,
  selected_processed_image_hash,
  preprocessing_fingerprint,
  OCR_package_and_checkpoint_revisions,
  OCR_parameters,
  adapter_and_ordering_revision
)
```

Store one canonical result per key. Cache hits should not conceal engine failures or substitute a different image. Use atomic writes or a completed-file marker so interrupted OCR cannot leave a valid-looking partial record.

OCR is shared between extraction methods only within the same condition. If preprocessing changes, both methods use the corresponding new OCR output and its distinct fingerprint.

## OCR diagnostics

Where reference transcripts exist, examine character error rate (CER) and word error rate (WER) with a declared alignment procedure. Line splits/merges can distort these metrics; report whether references are matched by boxes, region sequence, or whole-receipt transcript.

Track empty OCR rate, unit count distribution, low-score count, runtime, and target-field coverage under manual/reference alignment. Detector metrics require a compatible geometric reference/evaluator. End-to-end field accuracy remains the required product metric.

## Failure handling and acceptance

For no detected text, return `empty_ocr`, preserve the image, leave fields missing, and offer retry/review. An exception is `ocr_failed`, not an empty successful receipt.

Accept this module when overlays align, native output is recoverable, order is deterministic, granularity is verified, version fingerprints are complete, and the same cache can feed both extractors. Save an easy receipt and a difficult receipt for later screenshots; capture is not required now.
