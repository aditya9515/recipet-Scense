"""Audit the pinned SROIE mirror, freeze splits, and create shared OCR records."""

from __future__ import annotations

import csv
import io
import random
from pathlib import Path

from PIL import Image

from .common import FIELDS, ROOT, digest, read_json, read_jsonl, utc_now, write_json, write_jsonl
from .imaging import prepare_image
from .ocr_adapter import get_engine, run_ocr


def prepare_dataset(
    raw_root: str | Path = ROOT / "data/raw", output: str | Path = ROOT / "data", seed: int = 42
) -> dict:
    import pyarrow.parquet as pq

    raw_root, output = Path(raw_root), Path(output)
    images = output / "processed/images"
    images.mkdir(parents=True, exist_ok=True)
    records, excluded = [], []
    for split in ["train", "test"]:
        parquet = pq.ParquetFile(raw_root / f"{split}.parquet")
        for batch in parquet.iter_batches(batch_size=1):
            row = batch.to_pylist()[0]
            data = row["image"]["bytes"]
            image_hash = digest(data)
            receipt_id = image_hash[:20]
            try:
                with Image.open(io.BytesIO(data)) as image:
                    width, height = image.size
                    image.verify()
                entities = row["entities"]
                if any(not isinstance(entities.get(f), str) or not entities[f].strip() for f in FIELDS):
                    raise ValueError("Missing reference entity")
                if len(row["words"]) != len(row["bboxes"]):
                    raise ValueError("Reference text/box count mismatch")
                suffix = ".jpg" if data.startswith(b"\xff\xd8") else ".png"
                path = images / f"{receipt_id}{suffix}"
                if not path.exists():
                    path.write_bytes(data)
                records.append(
                    {
                        "receipt_id": receipt_id,
                        "source_key": row["key"],
                        "source_partition": split,
                        "image_path": str(path.resolve()),
                        "image_sha256": image_hash,
                        "width": width,
                        "height": height,
                        "reference": entities,
                        "reference_words": row["words"],
                        "reference_boxes": row["bboxes"],
                    }
                )
            except Exception as exc:
                excluded.append({"source_key": row["key"], "source_partition": split, "reason": str(exc)})
    test_hashes = {r["image_sha256"] for r in records if r["source_partition"] == "test"}
    seen = set()
    kept = []
    for record in records:
        key = (record["source_partition"], record["image_sha256"])
        reason = None
        if record["source_partition"] == "train" and record["image_sha256"] in test_hashes:
            reason = "training duplicate of held-out receipt"
        elif key in seen:
            reason = "byte-identical duplicate within supplied partition"
        seen.add(key)
        if reason:
            excluded.append({"source_key": record["source_key"], "reason": reason})
        else:
            kept.append(record)
    training = sorted([r for r in kept if r["source_partition"] == "train"], key=lambda r: r["receipt_id"])
    random.Random(seed).shuffle(training)
    n_val = max(1, round(len(training) * 0.1))
    splits = {
        "validation": training[:n_val],
        "train": training[n_val:],
        "test": sorted([r for r in kept if r["source_partition"] == "test"], key=lambda r: r["receipt_id"]),
    }
    hashes = {}
    for name, subset in splits.items():
        for record in subset:
            record["split"] = name
        write_jsonl(output / "manifests" / f"{name}.jsonl", subset)
        hashes[name] = digest([r["receipt_id"] for r in subset])
    write_json(output / "manifests/exclusions.json", excluded)
    card = {
        "created_at": utc_now(),
        "provider": "jsdnrs/ICDAR2019-SROIE",
        "source_url": "https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE",
        "revision": "bffe40c26759f3376ec2b3ae9031dbba54cd587c",
        "mirror_license": "CC BY 4.0",
        "terms_files": [str((raw_root / n).resolve()) for n in ["LICENSE", "NOTICE", "README.md"]],
        "seed": seed,
        "counts": {n: len(v) for n, v in splits.items()},
        "split_hashes": hashes,
        "exclusions": len(excluded),
        "duplicate_policy": "remove train duplicates of test; deduplicate within partition",
        "reference_geometry": "mirror OCR regions; inspect word/line granularity; not deployment OCR",
        "archive_sha256": {s: digest((raw_root / f"{s}.parquet").read_bytes()) for s in ["train", "test"]},
    }
    write_json(output / "dataset_card.json", card)
    with (output / "inventory.csv").open("w", encoding="utf-8", newline="") as stream:
        columns = ["receipt_id", "source_key", "split", "width", "height", "image_sha256"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(r for subset in splits.values() for r in subset)
    return card


def build_ocr_records(
    manifest: str | Path,
    output: str | Path = ROOT / "data/processed/records",
    *,
    variant: str = "identity",
    limit: int | None = None,
) -> dict:
    engine = get_engine()
    entries = read_jsonl(manifest)
    entries = entries[:limit] if limit else entries
    output = Path(output)
    failures = []
    success = 0
    for index, entry in enumerate(entries):
        try:
            record_path = output / variant / f"{entry['receipt_id']}.json"
            if record_path.exists():
                cached = read_json(record_path)
                prior_image, prior_ocr = cached["image"], cached["ocr"]
                expected_image_config = digest(
                    {
                        "variant": variant,
                        "operations": prior_image["operations"],
                        "matrix": prior_image["display_to_processed"],
                        "version": "image-v2",
                    }
                )
                if (
                    prior_image["source_sha256"] == entry["image_sha256"]
                    and prior_image["preprocessing_fingerprint"] == expected_image_config
                    and prior_ocr["ocr_fingerprint"] == engine.fingerprint
                    and Path(prior_image["processed_path"]).exists()
                ):
                    success += 1
                    print(f"OCR {index + 1}/{len(entries)} {entry['source_key']} record reused", flush=True)
                    continue
            path = Path(entry["image_path"])
            image = prepare_image(path.read_bytes(), path.name, variant=variant)
            ocr, hit = run_ocr(image, engine=engine)
            write_json(
                record_path, {"image": image.model_dump(mode="json"), "ocr": ocr.model_dump(mode="json")}
            )
            success += 1
            print(
                f"OCR {index + 1}/{len(entries)} {entry['source_key']} units={len(ocr.units)} cache={hit}",
                flush=True,
            )
        except Exception as exc:
            failures.append({"receipt_id": entry["receipt_id"], "error": str(exc)})
            print(f"OCR failure {entry['receipt_id']}: {exc}", flush=True)
    summary = {
        "manifest": str(manifest),
        "ocr_fingerprint": engine.fingerprint,
        "variant": variant,
        "success": success,
        "failures": failures,
    }
    write_json(output / f"ocr_summary_{Path(manifest).stem}_{variant}.json", summary)
    return summary


def dataset_status() -> dict:
    card = ROOT / "data/dataset_card.json"
    return read_json(card) if card.exists() else {"status": "not_prepared"}
