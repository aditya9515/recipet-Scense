# 04 - Image validation and conservative preprocessing

[Index](../README.md) | Previous: [dataset](03_DATASET_AND_ANNOTATION.md) | Next: [OCR](05_OCR_AND_READING_ORDER.md)

## Role in the pipeline

Prepare a readable receipt image while preserving its content and evidence coordinates. Input: uploaded/stored image and explicit configuration. Output: selected RGB image, original image reference, quality diagnostics, status and transform metadata from [module 02](02_ARCHITECTURE_AND_DATA_CONTRACTS.md).

The source proposes image correction; it does not establish that correction improves results. Start with identity processing and evaluate every added operation.

## 1. Validate and decode

1. Accept PNG/JPEG initially. Check decoder success rather than trusting filename extensions.
2. Apply configurable upload and decoded-pixel limits. The implementation accepts 10 MB files and up to 40 megapixels; the audited dataset includes legitimate scans above the originally proposed 20 megapixels. Perspective and deskew outputs are checked against the same pixel ceiling.
3. Convert grayscale/RGBA/CMYK inputs to a documented RGB representation; composite alpha consistently.
4. Apply EXIF orientation once and record the raw-to-display transform.
5. Retain the orientation-corrected original for comparison; never overwrite the upload.
6. Return a structured failure for undecodable/empty inputs. Large or unsupported files get an actionable UI message.

Limit values are engineering defaults, not accuracy claims or dataset properties.

## 2. Measure quality before modifying pixels

Potential signals: image dimensions, estimated text height, global contrast, saturated regions, skew angle estimate, and a blur measure. Such signals are diagnostics. For example, a Laplacian-based blur score depends on resolution/content and needs validation before driving a reject/retry threshold.

Record signals and the chosen action. A low-quality warning may request a clearer image while allowing inspection. Do not silently reject difficult in-scope test receipts from the scoring denominator.

## 3. Add transformations incrementally

| Variant | Proposed operation | Main failure risk | Check |
|---|---|---|---|
| P0 | EXIF handling and RGB conversion only | Poor original readability | Baseline condition |
| P1 | Small-angle deskew with padding | Cropped corners or wrong angle | Text remains visible; transform saved |
| P2 | Perspective correction when corners are reliable | Incorrect boundary/warping | Corner plausibility and aligned overlays |
| P3 | Mild contrast/illumination correction | Faint strokes removed or amplified noise | OCR/field comparison on validation |

Use OpenCV for the proposed geometric/image operations. Keep version, interpolation, padding color, crop bounds, and parameters in the preprocessing fingerprint. Avoid automatically stacking all variants: combinations create a different experiment.

Do not force every receipt into a page-sized rectangle before OCR. Preserve aspect ratio and adequate text resolution. Model-specific image resizing happens later through its processor and must not be confused with OCR resolution.

## 4. Coordinate bookkeeping

If deskew rotates then translates into a padded canvas, save both operations in the proper multiplication order. A crop has an offset; resizing has scale factors; perspective correction has a homography. Apply the same chain to every point requiring conversion.

Example, illustrative only: a display point at `(100, 200)` resized by 0.5 becomes `(50, 100)` before further transforms. A processed OCR polygon cannot be drawn unchanged over the full-resolution display image.

Use known corner/landmark fixtures for rotation, scaling, crop, and perspective mapping. For invertible transforms, verify a round trip within a small numerical tolerance in pixels. Visually inspect the overlaid regions on actual training receipts.

## 5. Choosing a variant

Use a fixed validation set to choose preprocessing. For a rules-versus-model pair, pass both the same OCR produced by the selected processed image. To measure preprocessing itself, regenerate OCR and labels/caches as required and hold other decisions fixed.

For a strict inference preprocessing ablation, use one fixed checkpoint on both variants and state the possible training-distribution mismatch. For separately retrained pipelines, name the experiment as a pipeline comparison; do not attribute the whole change to one image operation.

## 6. What to store

Store original/processed hashes, dimensions, frame names, transform chain, quality signals, preprocessing revision, parameter values, chosen variant, failure flags, and processing time. Cache identity processing too so it is as reproducible as enhancement.

## Useful verification and completion

- Known rotations and EXIF examples retain correct orientation and boxes.
- Transparent/grayscale images yield expected RGB inputs.
- Empty or malformed uploads fail visibly.
- Long receipts keep bottom text after resizing/rotation.
- A perspective-corrected receipt has correctly mapped evidence.
- Validation tables include cases where enhancement hurts.

Do not add arbitrary aggressive binarization just to make receipts look cleaner. Legible appearance and downstream extraction quality must both be checked.
