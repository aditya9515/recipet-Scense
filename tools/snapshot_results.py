"""Copy selected reproducibility evidence without runtime images, DBs or weights."""

import hashlib
import json
import shutil

from receiptsense.common import ROOT


def main():
    relative_paths = [
        "artifacts/environment.json",
        "artifacts/benchmark.json",
        "artifacts/test-results.xml",
        "artifacts/checkpoint_reload_check.json",
        "artifacts/api_smoke_check.json",
        "artifacts/ui_workflow_check.json",
        "artifacts/error_analysis.json",
        "data/processed/labels_train.summary.json",
        "data/processed/labels_validation.summary.json",
    ]
    for folder in ["artifacts/runs/held-out-v1", "artifacts/robustness/robustness-v1"]:
        relative_paths.extend(
            str(path.relative_to(ROOT)).replace("\\", "/")
            for path in sorted((ROOT / folder).iterdir())
            if path.suffix in {".json", ".jsonl"}
        )
    for name in [
        "receiptsense_model.json",
        "tiny_overfit_check.json",
        "training_log.json",
        "validation_metrics.json",
        "validation_predictions.jsonl",
    ]:
        relative_paths.append(f"artifacts/models/receiptsense-layoutlmv3/{name}")
    manifest = []
    for relative in relative_paths:
        source, target = ROOT / relative, ROOT / "results" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest.append(
            {
                "path": relative,
                "bytes": target.stat().st_size,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    (ROOT / "results/manifest.json").write_text(
        json.dumps(
            {
                "scope": "selected frozen experiment evidence; no model weights or runtime databases",
                "files": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Snapshotted {len(manifest)} evidence files")


if __name__ == "__main__":
    main()
