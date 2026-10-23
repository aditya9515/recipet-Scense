"""Warm, uncached pipeline latency distinct from startup and cached extraction."""

from __future__ import annotations

import gc
from pathlib import Path
from time import perf_counter

import numpy as np

from .common import ROOT, read_jsonl, utc_now, write_json
from .model import get_extractor
from .ocr_adapter import get_engine
from .pipeline import process_receipt


def benchmark(manifest: str | Path, *, limit: int = 10, checkpoint: str | Path | None = None) -> dict:
    entries = read_jsonl(manifest)[:limit]
    if not entries:
        raise ValueError("Empty benchmark manifest")
    get_engine.cache_clear()
    get_extractor.cache_clear()
    gc.collect()
    start = perf_counter()
    get_engine()
    engine_startup = perf_counter() - start
    start = perf_counter()
    checkpoint = checkpoint or ROOT / "artifacts/models/receiptsense-layoutlmv3"
    get_extractor(str(checkpoint))
    model_startup = perf_counter() - start
    first = Path(entries[0]["image_path"])
    for method in ["rules", "layoutlmv3"]:
        process_receipt(first.read_bytes(), first.name, method=method, checkpoint=checkpoint, use_cache=False)
    rows = []
    for entry in entries:
        path = Path(entry["image_path"])
        for method in ["rules", "layoutlmv3"]:
            result = process_receipt(
                path.read_bytes(), path.name, method=method, checkpoint=checkpoint, use_cache=False
            )
            rows.append({"receipt_id": entry["receipt_id"], "method": method, **result.prediction.timings})
        print(f"Latency {entry['source_key']}", flush=True)
    summary = {
        "created_at": utc_now(),
        "n_receipts": len(entries),
        "warmup_per_method": 1,
        "device": "CPU" if get_extractor(str(checkpoint)).device.type == "cpu" else "CUDA",
        "engine_startup_seconds": engine_startup,
        "model_load_seconds": model_startup,
        "scope": "warm uncached image preparation+OCR+extraction+validation; startup measured separately",
        "startup_scope": "fresh engine/model instances in this process; excludes Python/dependency imports and OS cold-cache guarantees",
        "methods": {},
        "rows": rows,
    }
    for method in ["rules", "layoutlmv3"]:
        times = [r["total_seconds"] for r in rows if r["method"] == method]
        summary["methods"][method] = {
            "median_seconds": float(np.median(times)),
            "p95_seconds": float(np.quantile(times, 0.95)),
            "n": len(times),
        }
    write_json(ROOT / "artifacts/benchmark.json", summary)
    return summary
