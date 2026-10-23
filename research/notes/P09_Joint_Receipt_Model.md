# P09 - Joint receipt transformer and rules

[Library](../README.md) | [DOI](https://doi.org/10.3390/make7040167) | [Local PDF](../papers/P09_Joint_Receipt_Model.pdf)

**Citation:** Xandru Mifsud et al., Machine Learning and Knowledge Extraction, 7(4), 167, 2025. Imported publisher version of record.

## Central idea

The study combines OCR, a LayoutLMv3-based receipt classifier and designed rules, using its own annotated real/synthetic receipts and a broader label set than our four fields.

## What to read

Read annotation creation/curation, OCR selection, learned model, rules and ablations. Inspect class-level outcomes rather than assuming every rule improves every class.

## ReceiptSense use

Frame rules-versus-model and possible combined experiments, with transparent supervision and evaluation. See [baseline](../../docs/06_RULES_BASELINE.md) and [experiments](../../docs/10_EVALUATION_AND_EXPERIMENTS.md).

## Boundaries and questions

We do not reproduce the full method or borrow its scores as targets/results. Its dataset, class definitions and metric differ. Combining transformers and rules is prior work, so our contribution must focus on the actual adaptation, inspection and controlled findings.

**Status:** closely related work, not an additional required architecture.
