"""Fixed, paired receipt degradation probes with explicit parent identifiers."""

from __future__ import annotations

import io
import random
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from .common import ROOT, digest, read_jsonl, utc_now, write_json, write_jsonl
from .evaluation import score_predictions
from .pipeline import process_receipt


def perturb(data: bytes, condition: str) -> bytes:
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    # Every condition, including clean, shares this declared probe resolution.
    image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    if condition == "blur":
        array = cv2.GaussianBlur(np.asarray(image), (5, 5), 1.2)
        image = Image.fromarray(array)
    elif condition == "rotation":
        image = image.rotate(7, Image.Resampling.BICUBIC, expand=True, fillcolor="white")
    elif condition == "contrast":
        image = ImageEnhance.Contrast(image).enhance(0.55)
    elif condition == "perspective":
        w, h = image.size
        source = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        target = np.array(
            [[0.04 * w, 0.02 * h], [0.97 * w, 0], [w - 1, 0.96 * h], [0, h - 1]], dtype=np.float32
        )
        matrix = cv2.getPerspectiveTransform(source, target)
        image = Image.fromarray(
            cv2.warpPerspective(np.asarray(image), matrix, (w, h), borderValue=(255, 255, 255))
        )
    elif condition != "clean":
        raise ValueError(f"Unknown condition: {condition}")
    stream = io.BytesIO()
    image.save(stream, format="PNG", compress_level=1)
    return stream.getvalue()


def run_robustness(
    manifest: str | Path,
    *,
    limit: int = 12,
    seed: int = 42,
    checkpoint: str | Path | None = None,
    methods: tuple = ("rules", "layoutlmv3"),
    conditions: tuple = ("clean", "deskew", "blur", "rotation", "contrast", "perspective"),
    name: str | None = None,
) -> dict:
    entries = read_jsonl(manifest)
    if not entries:
        raise ValueError("Empty probe manifest")
    selected = random.Random(seed).sample(entries, min(limit, len(entries)))
    folder = ROOT / "artifacts/robustness" / (name or f"probe-{uuid4().hex[:10]}")
    folder.mkdir(parents=True, exist_ok=False)
    summary = {
        "created_at": utc_now(),
        "seed": seed,
        "n_parent_receipts": len(selected),
        "parent_ids": [e["receipt_id"] for e in selected],
        "manifest": str(manifest),
        "probe_policy": "common max-side 1600; automatic deskew; blur 5x5 sigma1.2; rotation7deg padded; contrast0.55; perspective0.04",
        "partial_subset": len(selected) < len(entries),
        "results": {},
        "failures": [],
    }
    for condition in conditions:
        references, outputs = [], {method: [] for method in methods}
        for entry in selected:
            data = perturb(
                Path(entry["image_path"]).read_bytes(), "clean" if condition == "deskew" else condition
            )
            receipt_id = digest(data)[:20]
            reference = {
                **entry,
                "receipt_id": receipt_id,
                "parent_receipt_id": entry["receipt_id"],
                "condition": condition,
            }
            references.append(reference)
            for method in methods:
                try:
                    result = process_receipt(
                        data,
                        f"{entry['source_key']}_{condition}.png",
                        method=method,
                        variant="deskew" if condition == "deskew" else "identity",
                        checkpoint=checkpoint,
                    )
                    outputs[method].append(result.prediction.model_dump(mode="json"))
                except Exception as exc:
                    summary["failures"].append(
                        {
                            "parent_id": entry["receipt_id"],
                            "condition": condition,
                            "method": method,
                            "error": str(exc),
                        }
                    )
            print(f"Robustness {condition} {entry['source_key']}", flush=True)
        write_jsonl(folder / f"{condition}_references.jsonl", references)
        summary["results"][condition] = {}
        for method in methods:
            write_jsonl(folder / f"{condition}_{method}_predictions.jsonl", outputs[method])
            metrics = score_predictions(references, outputs[method])
            write_json(folder / f"{condition}_{method}_metrics.json", metrics)
            summary["results"][condition][method] = {k: v for k, v in metrics.items() if k != "rows"}
        write_json(folder / "summary.json", summary)
    return summary
