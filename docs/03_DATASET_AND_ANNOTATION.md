# 03 - Dataset preparation, splits, and annotation alignment

[Index](../README.md) | Previous: [contracts](02_ARCHITECTURE_AND_DATA_CONTRACTS.md) | Next: [preprocessing](04_IMAGE_PREPROCESSING.md)

## Purpose

Turn the selected receipt release into trustworthy, split-safe training and evaluation data. This is the main scientific dependency of the learned extractor.

Primary reading: [P02 SROIE](../research/notes/P02_SROIE.md). Its competition report distinguishes localization, recognition, and key information extraction. Use its four target fields while auditing the release actually obtained. [Original paper](https://arxiv.org/abs/2103.10213).

## 1. Acquire and inventory

Start with the [SROIE competition entry point](https://rrc.cvc.uab.es/?ch=13). Access may require registration; the entry point was not retrievable during this documentation session. A mirror is acceptable only with recorded provenance, version and usage terms. Do not assume a mirror is an official benchmark release.

The competition paper and redistributed datasets may describe different counts/splits. **Do not hardcode a remembered count.** Record observed images, paired OCR annotations, usable entity labels, and exclusions for each supplied partition.

Suggested inventory columns:

```text
receipt_id, source_partition, image_path, image_sha256, width, height,
ocr_annotation_path, entity_annotation_path, image_ok, boxes_ok,
entities_ok, duplicate_group, exclusion_reason
```

Dataset card fields: acquisition date, provider URL, archive filename/hash, release/version, terms location, field definitions, language/domain observations, annotation granularity, missing labels, ID derivation, and final counts.

## 2. Parse annotations without losing text

Inspect actual files before implementing a parser. A common receipt-box format begins with eight coordinates followed by text. Text itself may contain commas: split only the first eight commas where that format applies. Handle BOM, Unicode, line endings, escaped text and malformed lines explicitly.

For each polygon, check four valid corners, nonzero area, bounds, orientation convention, and image pairing. Do not silently drop an annotation because the detector would find a different region.

Entity files may give only four complete strings. Those strings are sufficient to score final fields but **do not directly assign BIO labels to OCR words**. Some OCR releases group words into lines; determine this from the actual data.

## 3. Freeze splits before fitting anything

Proposed policy:

1. Keep the release's official held-out test partition if usable labels are available for local evaluation.
2. Split the remaining supplied training IDs into about 90% train and 10% validation with seed 42.
3. Detect byte duplicates and likely visual duplicates before splitting. Review near-duplicate groups and keep them together.
4. If the original partition itself contains cross-split duplicates, document the issue and choose a predeclared policy. Report any filtered evaluation as a modified subset, alongside original-split results where possible.
5. If no trustworthy official partition exists, define and document a receipt-group split before model/rule development.

Save explicit manifests; a random seed alone does not identify a split if input ordering changes. Hash the manifests. All crops, enhanced versions, and degraded copies inherit the parent's split. Template/shop-disjoint evaluation is optional and must be distinguished from a standard receipt split.

If test field labels are inaccessible, keep that test untouched for any later official submission and use a labelled holdout carved from training. Name it a **local holdout**, not official test performance.

## 4. Build OCR-to-field supervision

The deployed pipeline uses engine OCR; train the main experiment on the same engine and granularity to reduce input mismatch.

1. Run the frozen OCR adapter on training receipts and retain original region IDs.
2. Normalize reference and OCR text conservatively for matching only: Unicode normalization, case folding, and whitespace handling. Save original strings and character-to-unit offsets.
3. Search contiguous unit spans, including consecutive lines for addresses, whose normalized text equals a reference field.
4. Preserve all exact candidates. Use field context to identify repeated values; route unresolved ties to a reviewer.
5. For a total, label the amount span, not the literal word `TOTAL`, unless the chosen annotation policy explicitly includes labels for every field consistently.
6. Apply bounded fuzzy matching only as a documented proposal for silver labels. Manually verify ambiguous/approximate positives before treating them as trustworthy supervision.
7. Assign `B-FIELD`, `I-FIELD` to approved contiguous spans and `O` only to confidently unrelated units.
8. Use ignore labels on unresolved units. If the missing field location cannot be localized, either manually resolve the receipt or quarantine that training receipt from token loss. Do not train all its unknown positive units as negative examples.
9. Save aligned units, policy version, match method, ambiguity reasons and review decisions.

For multi-span/discontinuous fields, do not fake a contiguous entity across unrelated text. Define and audit a reconstruction rule or mark the example unresolved for training. Final receipt-level evaluation remains possible.

## 5. Word geometry versus line geometry

Preferred: genuine word-level coordinates from the frozen OCR pipeline, if verified available. If the engine returns lines, either retain each line as a model text unit or split into words while replicating its line rectangle. Replicated geometry must be labelled as approximate. Equal-width word boxes invented from string length are not observed word boxes.

Replicated line rectangles may be useful for a prototype, but a line containing both label and amount can make field boundaries/evidence coarse. Record this limitation and keep the same policy at training and inference. Changing granularity requires new caches, annotations and runs.

## 6. Annotation audit

Inspect at least 20-30 training receipts initially, with company/address/date/total examples, multiline addresses, repeated amounts, OCR misses, and long receipts. Sample randomly plus known difficult categories; report that sampling scheme. Increase the audit until systematic errors are corrected.

Track per field:

| Audit quantity | Why it matters |
|---|---|
| Reference field count | Real denominator |
| Unambiguous exact alignments | High-confidence supervision coverage |
| Ambiguous repeated matches | Risk of training on the wrong region |
| Approximate matches and manual fixes | Weak-label dependence |
| No OCR match | OCR ceiling/coverage limitation |
| Masked or quarantined examples | Effective training loss coverage |
| Audited positive-label errors | Direct label quality estimate |

Independent review of a subset is useful. Fix annotation policy from train/validation only. Keep reference annotation corrections separately versioned and justified from the image, never tailored to a system's output.

## 7. Evaluation boundaries

No training or matching procedure may read test entity labels. The evaluator may compare final fields with those labels after predictions have been saved and frozen. Token-F1 evaluation needs a separate audited span reference; field exact match does not.

Never remove test receipts merely because OCR misses their target fields or token alignment fails. Include them in end-to-end field scores. An oracle experiment using provided transcripts/boxes must be named separately and is not deployable OCR performance.

## Required outputs and acceptance

Produce the dataset card, inventory, exclusion ledger, three split manifests/hashes, OCR annotation schema, alignment coverage report, and reviewed examples. Accept the data stage only when IDs are disjoint, annotation provenance is inspectable, and the model's intended unit granularity is known.

Implementation update: a pinned SROIE mirror has now been imported and audited. Actual counts/split hashes are in `data/dataset_card.json`, visual duplicate review in `data/near_duplicate_audit.json`, and alignment coverage in `data/processed/labels_train.summary.json` after preparation completes. These facts were established by the data pipeline, separately from the earlier paper import. See [RUNNING.md](../RUNNING.md).
