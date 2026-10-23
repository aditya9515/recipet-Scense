# P01 - LayoutLMv3

[Library](../README.md) | [Original paper](https://arxiv.org/abs/2204.08387v3) | [Local PDF](../papers/P01_LayoutLMv3.pdf)

**Citation:** Yupan Huang et al., ACM Multimedia, 2022. Imported arXiv version: v3.

## Central idea

LayoutLMv3 learns document representations using text, spatial positions and image patches. Its pretraining combines text/image masking with word-patch alignment.

## What to read

Read the introduction, architecture and pretraining objectives, then the downstream extraction setup. Distinguish pretraining from fine-tuning and identify the OCR assumptions in any benchmark result.

## ReceiptSense use

Adapt a pretrained checkpoint for four-field token classification. Preserve OCR unit IDs through encoding/decoding so predictions have image evidence. See [training](../../docs/07_LAYOUTLMV3_TRAINING.md).

## Boundaries and questions

We do not recreate pretraining or claim the paper's reported performance. The project must establish annotation quality, handle long receipts and evaluate actual OCR inputs. Ask whether failures come from missing OCR text, wrong geometry or classification.

**Implementation status:** the required adaptation is implemented in the training module. Saved model manifests distinguish the measured frozen-backbone head adaptation from the separately supported full fine-tuning mode.
