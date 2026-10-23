"""One-shot dependent experiment stages, preserving completed run artifacts."""

import time

from receiptsense.common import ROOT, read_json


def main():
    deadline = time.monotonic() + 7200
    model_marker = ROOT / "artifacts/models/receiptsense-layoutlmv3/receiptsense_model.json"
    ocr_marker = ROOT / "data/processed/records/ocr_summary_test_identity.json"
    print("Awaiting the completed trained bundle and held-out OCR manifest.", flush=True)
    while not (model_marker.exists() and ocr_marker.exists()):
        if time.monotonic() > deadline:
            raise TimeoutError("Training or dataset preparation did not complete")
        time.sleep(5)
    if not read_json(model_marker).get("trained"):
        raise ValueError("Model marker does not identify a trained bundle")
    from receiptsense.benchmark import benchmark
    from receiptsense.evaluation import evaluate_manifest
    from receiptsense.robustness import run_robustness

    manifest = ROOT / "data/manifests/test.jsonl"
    if not (ROOT / "artifacts/runs/held-out-v1/summary.json").exists():
        result = evaluate_manifest(manifest, run_name="held-out-v1")
        print(
            "HELD_OUT_COMPLETE",
            {m: v["strict-v1"]["macro_field_em"] for m, v in result["results"].items()},
            flush=True,
        )
    if not (ROOT / "artifacts/robustness/robustness-v1/summary.json").exists():
        run_robustness(manifest, name="robustness-v1")
        print("ROBUSTNESS_COMPLETE", flush=True)
    if not (ROOT / "artifacts/benchmark.json").exists():
        benchmark(manifest)
        print("LATENCY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
