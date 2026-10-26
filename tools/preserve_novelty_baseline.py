"""Preserve the published baseline identity before running the new experiments."""

import importlib.metadata
import subprocess

from receiptsense.common import ROOT, digest, read_json, utc_now, write_json
from receiptsense.novelty_evaluation import checkpoint_identity


def main():
    baseline = "12dc9749ee77df55c04e7443ecc404b37009e280"
    output = ROOT / "results/novelty/baseline_manifest.json"
    if output.exists():
        print("Baseline record already preserved")
        return
    subprocess.run(["git", "cat-file", "-e", f"{baseline}^{{commit}}"], check=True, cwd=ROOT)
    engine = read_json(ROOT / "data/processed/records/ocr_summary_validation_identity.json")
    files = [
        "results/artifacts/runs/held-out-v1/summary.json",
        "results/artifacts/benchmark.json",
        "results/artifacts/test-results.xml",
        "results/artifacts/checkpoint_reload_check.json",
    ]
    record = {
        "preserved_at": utc_now(),
        "baseline_commit": baseline,
        "repository": "https://github.com/aditya9515/recipet-Scense",
        "checkpoint": checkpoint_identity(verify_weights=True),
        "ocr_fingerprint": engine["ocr_fingerprint"],
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in [
                "torch",
                "transformers",
                "rapidocr-onnxruntime",
                "onnxruntime",
                "streamlit",
                "pydantic",
            ]
        },
        "published_evidence": {name: digest((ROOT / name).read_bytes()) for name in files},
        "training_change": "none; existing frozen-backbone head held fixed for every ablation",
    }
    write_json(output, record)
    print("Preserved baseline commit, model weights, OCR, environment and original results")


if __name__ == "__main__":
    main()
