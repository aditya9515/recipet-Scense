# 01 - Understand ReceiptSense

[Index](../README.md) | Previous: [source](00_SOURCE_AND_REQUIREMENTS.md) | Next: [architecture](02_ARCHITECTURE_AND_DATA_CONTRACTS.md)

## The problem in ordinary language

A receipt is a picture of semi-structured information. People can usually recognize the shop name, address, purchase date, and final total quickly. Software first needs to locate printed text, read it, then determine what each piece means.

ReceiptSense converts that picture into four useful fields and lets a person inspect the evidence before accepting them. The purpose is assisted receipt indexing and expense entry. The computer vision work is image preparation, text localization, recognition, and use of two-dimensional layout and visual information.

## A complete example

This is a constructed example for understanding, not project data:

```text
EXAMPLE STORE
12 MAIN ROAD
DATE 2025-02-14
SUBTOTAL 470.00
TAX       23.50
TOTAL    493.50
CASH     500.00
CHANGE     6.50
```

Desired values:

```json
{
  "company": "EXAMPLE STORE",
  "address": "12 MAIN ROAD",
  "date": "2025-02-14",
  "total": "493.50"
}
```

OCR might recognize all amounts perfectly, yet an extractor that chooses the largest number returns cash, 500.00. Field extraction must interpret labels and neighboring positions. A model may also fail if OCR reads `493.50` as `493.SO`. The interface should highlight the source amount and permit correction, preserving the original prediction.

## The system as seven responsibilities

| Responsibility | Question answered | Result |
|---|---|---|
| Image preparation | Can the receipt be decoded and read safely? | Selected image and transform metadata |
| Text detection | Where is the printed text? | Polygons/regions |
| Text recognition | What characters are in each region? | OCR units and scores |
| Field extraction | Which units describe each target? | Four predicted raw fields and evidence |
| Validation | Are values parseable or ambiguous? | Normalized values and review reasons |
| Human review | What does the user accept or correct? | Separate reviewed record and edit history |
| Export/evaluation | Can the record be used, and how well did automation perform? | Structured export and reproducible metrics |

## What the research contributes

[LayoutLMv3 (P01)](../research/notes/P01_LayoutLMv3.md) supplies the main pretrained model; [SROIE (P02)](../research/notes/P02_SROIE.md) supplies the task and benchmark context. Detection and recognition papers explain the OCR stages. LayoutLM/BROS explain why spatial information matters. TrOCR/Donut are optional alternative directions. The related joint receipt model motivates examining rules alongside learned extraction.

The proposed student contribution is a working, inspectable adaptation and controlled evaluation. Combining existing OCR, a transformer, and rules is not itself a new model invention.

## Scope and deliverables

Initial engineering scope: one printed receipt per PNG/JPEG image, using an audited English/Latin-script dataset. The exact supported language range must be verified from the selected OCR/model setup. The original source requests printed receipts but does not define all languages or file formats.

The required output fields are company, address, date, and total. Optional metadata includes evidence IDs, currency if explicitly supported by the receipt, validation warnings, and processing time. Itemized products and tax extraction are outside the required four-field schema.

The project is complete when it has a reproducible dataset/split record, working OCR, a rules baseline, a trained learned extractor, shared validation, an evidence-and-correction UI, exports, held-out comparison, and a report supported by real evidence.

## Questions the experiments should answer

1. Does the trained model outperform rules on the same OCR and receipts?
2. Which fields are hardest, and how often are all four correct together?
3. Does conservative image correction help enough to justify its cost?
4. Are errors caused primarily by OCR, extraction, or normalization?
5. How often does the review policy flag errors, and how much correct output does it also flag?
6. How does latency change between cold startup, uncached processing, and cached extraction?

## Vocabulary

| Term | Meaning here |
|---|---|
| OCR unit | A text region from the selected engine; may be a word or line |
| Polygon | Corners enclosing an OCR region in an identified image frame |
| Axis-aligned box | Minimum/maximum x/y rectangle used as the model's spatial input |
| KIE | Key information extraction: assigning text to meaningful fields |
| BIO | Begin/inside/outside tags describing entity spans |
| Subword | A tokenizer piece; one OCR word can produce several |
| Fine-tuning | Adapting pretrained weights to labelled receipt examples |
| Weak/silver label | A label inferred by an alignment heuristic, not directly verified |
| Exact match | A complete field agrees with its reference under a frozen policy |
| Ablation | A comparison that changes one component while holding others fixed |
| Abstention | Returning missing/uncertain rather than a fabricated value |
| Leakage | Development gains information from data intended to stay held out |

## Your first understanding check

Explain a receipt's journey from image pixels to exported JSON. Then explain why a high OCR score, a plausible amount, and a high token probability do not independently establish that the total is correct. If this is clear, continue to the shared architecture.
