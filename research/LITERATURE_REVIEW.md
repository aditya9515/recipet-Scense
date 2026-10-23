# ReceiptSense literature synthesis and project mapping

[Research library](README.md) | [Source proposal](../downloadfile.pdf) | [Project plan](../PROJECT_PLAN.md)

## How the papers form a system

The papers address different levels. Detection and recognition recover text from pixels. Layout-aware models assign semantic roles to that text. Benchmark design determines what counts as success. OCR-free alternatives change the pipeline itself. Related hybrid work helps frame integration and comparison claims.

Our required implementation uses a pretrained OCR pipeline plus a rules baseline and a fine-tuned LayoutLMv3 extractor. The other papers provide foundations, related work, or optional future experiments; importing them does not mean their architectures are implemented.

## Comparison matrix

| Paper | Main input -> output | ReceiptSense use | Required implementation? | Key caution |
|---|---|---|---|---|
| P01 LayoutLMv3 | Image + text + boxes -> task predictions | Main learned extractor | Yes, downstream adaptation | Needs correct supervised labels and decoding |
| P02 SROIE | Receipt benchmark -> OCR/KIE task definitions | Four-field data/evaluation foundation | Yes, release audit | Verify actual downloaded split/labels |
| P03 DBNet | Image -> detected text regions | Detection concepts and region preservation | Pretrained OCR only | Installed detector may differ |
| P04 CRNN | Text crop -> character sequence | Recognition concepts | Pretrained OCR only | Recognition alone does not assign fields |
| P05 LayoutLM | Text/layout and visual features -> document understanding | Motivation for keeping geometry | Background | Not a second mandatory extractor |
| P06 BROS | Text + relative layout -> KIE | Reading-order/spatial analysis | Optional model | Requires its own fair supervised setup |
| P07 TrOCR | Text crop -> recognized text | Possible recognizer comparison | Optional | Needs detected crops and downstream extraction |
| P08 Donut | Document image -> generated structured output | Alternative OCR-free direction | Optional | Direct OCR-region evidence is less immediate |
| P09 Joint receipt model | OCR/image/layout + rules -> receipt labels | Related integration/comparator design | Related work only | Different dataset/classes/metrics |

## Stage-to-reading map

| Project task | Reading | Where it affects the design |
|---|---|---|
| Define field labels and benchmark | P02 | [Dataset](../docs/03_DATASET_AND_ANNOTATION.md) |
| Preserve regions and recognized text | P03, P04 | [OCR](../docs/05_OCR_AND_READING_ORDER.md) |
| Keep two-dimensional geometry | P05, P06 | [Contracts](../docs/02_ARCHITECTURE_AND_DATA_CONTRACTS.md) |
| Fine-tune the learned extractor | P01 | [Training](../docs/07_LAYOUTLMV3_TRAINING.md) |
| Compare rules and learned methods | P09 | [Rules](../docs/06_RULES_BASELINE.md), [evaluation](../docs/10_EVALUATION_AND_EXPERIMENTS.md) |
| Diagnose recognition limits | P07 | Future OCR comparison |
| Explore removing external OCR | P08 | Future alternative pipeline |

## Research position and contribution

The proposed contribution is a practical receipt system with inspectable evidence, explicit annotation adaptation, controlled rules/model comparison, and measured preprocessing/robustness analysis. The research literature supplies model ideas and task context. No claim of inventing OCR, multimodal document transformers, or transformer/rule combinations is justified.

The useful engineering question is whether those ingredients work reliably under the selected dataset, OCR, annotation policy and hardware. The report should support that answer with project-specific runs, including negative findings.

## Comparison discipline

Keep these boundaries visible when reading result tables:

- Gold transcripts/boxes versus real OCR inputs.
- Field exact match versus token/entity F1.
- Different receipt datasets and label sets.
- Different split definitions or filtered subsets.
- Different training corpora, augmentation and checkpoints.
- Automated output versus corrected records.
- Cold/uncached pipeline time versus cached/model-only time.

Do not rank these papers using incompatible published headline numbers. Record published results only when needed, with their exact task and protocol; this pack avoids transplanting those numbers into ReceiptSense targets.

## Full references and originals

1. Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei. **LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking.** ACM Multimedia, 2022. [Original](https://arxiv.org/abs/2204.08387) · [Local PDF](papers/P01_LayoutLMv3.pdf).
2. Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shjian Lu, C. V. Jawahar. **ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction.** ICDAR, 2019; public preprint 2021. [Original](https://arxiv.org/abs/2103.10213) · [Local PDF](papers/P02_SROIE.pdf).
3. Minghui Liao, Zhaoyi Wan, Cong Yao, Kai Chen, Xiang Bai. **Real-time Scene Text Detection with Differentiable Binarization.** AAAI, 2020. [Original](https://arxiv.org/abs/1911.08947) · [Local PDF](papers/P03_DBNet.pdf).
4. Baoguang Shi, Xiang Bai, Cong Yao. **An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition.** IEEE TPAMI, 2017; preprint 2015. [Original](https://arxiv.org/abs/1507.05717) · [Local PDF](papers/P04_CRNN.pdf).
5. Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou. **LayoutLM: Pre-training of Text and Layout for Document Image Understanding.** KDD, 2020. [Original](https://arxiv.org/abs/1912.13318) · [Local PDF](papers/P05_LayoutLM.pdf).
6. Teakgyu Hong, Donghyun Kim, Mingi Ji, Wonseok Hwang, Daehyun Nam, Sungrae Park. **BROS: A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents.** AAAI, 2022. [Original](https://arxiv.org/abs/2108.04539) · [Local PDF](papers/P06_BROS.pdf).
7. Minghao Li, Tengchao Lv, Jingye Chen, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei. **TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models.** AAAI, 2023; preprint 2021. [Original](https://arxiv.org/abs/2109.10282) · [Local PDF](papers/P07_TrOCR.pdf).
8. Geewook Kim, Teakgyu Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, Seunghyun Park. **OCR-free Document Understanding Transformer.** ECCV, 2022. [Original](https://arxiv.org/abs/2111.15664) · [Local PDF](papers/P08_Donut.pdf).
9. Xandru Mifsud, Leander Grech, Adriana Baldacchino, Léa Keller, Gianluca Valentino, Adrian Muscat. **Receipt Information Extraction with Joint Multi-Modal Transformer and Rule-Based Model.** Machine Learning and Knowledge Extraction, 7(4), article 167, 2025. [DOI](https://doi.org/10.3390/make7040167) · [Publisher](https://www.mdpi.com/2504-4990/7/4/167) · [Local PDF](papers/P09_Joint_Receipt_Model.pdf).
