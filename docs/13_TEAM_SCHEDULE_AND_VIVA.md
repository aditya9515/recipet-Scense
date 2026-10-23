# 13 - Team workflow, example schedule, and viva preparation

[Index](../README.md) | Previous: [risks](12_RISKS_AND_DECISIONS.md) | [Report](../report/PROJECT_REPORT.md)

## Team listed in the source

| Registration number | Team member |
|---|---|
| 230968014 | PANDIRI JASWANTH KUMAR |
| 230968316 | KOSURU VENKATA SAI ADITYA |
| 230968304 | PERIVELA VENKATA SRI SATYA PHANI PRITHVIRAM |
| 230968140 | GORASA SAKETH RAM |

These names are transcribed from the supplied PDF. The responsibilities below are suggestions for the team to assign, not existing commitments by individuals.

## Suggested four responsibility areas

| Area | Primary work | Required cross-review |
|---|---|---|
| Data and image pipeline | Dataset audit, splits, preprocessing, OCR | Model contributor reviews labels/geometry |
| Learned extraction | Alignment, processor inputs, training, decoding | Data contributor audits label examples |
| Application and integration | Common contracts, review storage, UI, export | Evaluation contributor checks prediction immutability |
| Evaluation and documentation | Rules comparison, metrics, experiments, report | Entire team checks results and evidence |

Keep interfaces shared. Each member should understand the full pipeline and be able to explain an end-to-end receipt. Do not let one person be the only holder of the training setup or split files.

## Six-week example schedule

No deadline was supplied. This is an adjustable relative schedule with deliverables, not a prediction of training duration.

| Week | Focus | Deliverable at review |
|---|---|---|
| 1 | Scope, schema, paper reading, environment and data audit | Requirements, inventory, split manifests, OCR smoke test |
| 2 | OCR adapter, image transforms, rules, evaluation skeleton | Shared OCR cache, baseline validation outputs, label audit |
| 3 | Model encoding, tiny overfit, initial training | Verified token/window mappings and reloadable initial checkpoint |
| 4 | Validation selection and end-to-end UI | Chosen configuration, evidence, correction and export demonstration |
| 5 | Freeze methods; held-out/robustness/timing runs | Saved predictions, result tables, error ledger |
| 6 | Report, screenshots, reproduction, rehearsal | Final evidence-backed report and runnable submission |

UI fixtures and evaluator development can overlap model work after contracts stabilize. Do not overlap “freeze methods” and “tune on final test” as if they were independent activities.

## Weekly review agenda

1. Show working artifacts and current gate status, not only slides.
2. Inspect one successful and one failed receipt end to end.
3. Review schema/configuration changes and any invalidated caches/results.
4. Confirm the next dependency-blocking task and its owner.
5. Update risks and the report's evidence ledger.

## Paper reading order

For the full team: P02 -> P01 -> P09. For OCR contributors: P03 -> P04. For spatial understanding: P05 -> P06. Read P07/P08 after the core design is understood; implementing them is optional.

Each individual [paper note](../research/LITERATURE_REVIEW.md) lists what to read, what the project uses, and what it does not reproduce.

## Demo sequence, about five minutes

1. State the four-field task and show an unseen supported receipt.
2. Run extraction, naming the method.
3. Select company/address/date/total and show corresponding evidence.
4. Explain a difficult amount or ambiguous date; correct one value.
5. Export JSON/CSV and show the prediction-versus-review distinction.
6. Show held-out rules/model comparison, counts, and one failure.
7. State the practical limits and what would be studied next.

Use genuine run outputs. A rehearsed receipt can demonstrate interaction, but it is not itself an evaluation set.

## Viva questions and concise answer points

| Question | Answer points |
|---|---|
| Why is this a computer vision project? | Images, geometry correction, text localization/recognition, and document visual-layout evidence drive the system |
| Why not use OCR alone? | OCR transcribes; field extraction assigns semantic roles such as total versus cash |
| Why LayoutLMv3? | It provides a pretrained basis for combining text, spatial positions and image information; actual benefit must be measured |
| What do you train? | The downstream token-classification task, with a new label map/head and chosen fine-tuning policy; not pretraining from scratch |
| How are training labels obtained? | Align reference fields to deployment OCR units, audit ambiguity and track weak/manual label provenance |
| How do you avoid data leakage? | Fixed receipt IDs/groups, validation-only selection, frozen test protocol and separate review outputs |
| What happens on long receipts? | Overlapping text windows preserve global unit IDs and bottom fields |
| What is your baseline? | Transparent field rules consuming the identical OCR cache |
| What proves improvement? | Paired held-out field metrics with shared conditions, failure counts and uncertainty where possible |
| Is model confidence trustworthy? | Raw OCR/token/rule scores are uncalibrated signals; review behavior needs separate validation |
| What is your original contribution? | Practical inspectable integration, annotation adaptation and controlled evaluation, without claiming invention of existing models |
| What if the model loses to rules? | Report it, diagnose supervision/OCR/model errors, and explain limits under the tested conditions |

## Final handover checklist

- The README leads to plan, modules, papers and report.
- Actual setup commands and hardware observations replace proposed setup notes.
- Data source and model terms/revisions are recorded.
- Result tables agree with saved runs; all planned-but-unrun work is labelled.
- Screenshots show real outputs, with receipt/run IDs where relevant.
- A new session loads the checkpoint and reproduces the demo/evaluation.
- Each teammate can explain one failure case and the common data contract.
