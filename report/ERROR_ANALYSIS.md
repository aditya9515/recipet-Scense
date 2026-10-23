# ReceiptSense - frozen held-out error analysis

Run `held-out-v1`: 358 receipts, both methods using the same OCR. This analysis is post-hoc; it does not change the evaluated checkpoint, rules, normalization or thresholds.

## Reference text availability in OCR

The diagnostic searches contiguous whole OCR units after NFKC/case/punctuation cleanup (amount signs retained). It allows up to 80 units. A found span establishes string availability, not that the region is semantically the desired field. Multiple equal amounts can be total, cash or change.

| Field | No matching span | One span | Multiple spans |
|---|---:|---:|---:|
| company | 111 | 247 | 0 |
| address | 140 | 216 | 2 |
| date | 205 | 133 | 20 |
| total | 9 | 116 | 233 |

Missing spans can result from recognition errors, transcript formatting, word boundaries, reading order or reference differences. This table is not an OCR character/word error rate and cannot separate detection from recognition failures.

## Strict field errors

Each strict error enters one category in this order: missing complete prediction; resolved by the declared normalized policy; missing field; reference span available; reference span not found. The categories are diagnostics rather than proven causes.

| Method | Field | Missing prediction | Normalization resolves | Missing field | Reference available | Reference not found | Total strict errors |
|---|---|---:|---:|---:|---:|---:|---:|
| rules | company | 0 | 18 | 0 | 134 | 111 | 263 |
| rules | address | 0 | 1 | 50 | 186 | 117 | 354 |
| rules | date | 0 | 20 | 95 | 7 | 29 | 151 |
| rules | total | 0 | 22 | 0 | 152 | 7 | 181 |
| layoutlmv3 | company | 0 | 19 | 0 | 128 | 111 | 258 |
| layoutlmv3 | address | 0 | 17 | 0 | 195 | 140 | 352 |
| layoutlmv3 | date | 0 | 16 | 104 | 25 | 103 | 248 |
| layoutlmv3 | total | 0 | 20 | 7 | 75 | 5 | 107 |

## How to use these findings

Use train/validation receipts to investigate label coverage, OCR geometry, competing spans and multi-line address boundaries. Richer verified labels and full-backbone adaptation are possible future experiments. Keep the held-out results above frozen; any changes motivated by these test errors need a new untouched confirmation set or a disclosure that the old set became development data.

[Full metrics and report](IMPLEMENTATION_REPORT.md) · [Complete error ledger](../results/artifacts/error_analysis.json)

OCR CER/WER, detection-versus-recognition attribution, fully manual causal error categories, alternative OCR benchmarks and multi-seed training were not measured in this build.
