"""Local-only, double-annotation confirmation collection and immutable freeze."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from .common import FIELDS, ROOT, digest, read_json, read_jsonl, utc_now, write_json, write_jsonl
from .ocr_adapter import get_engine
from .recovery_config import load_config
from .validation import clean_text

PRIVATE_ROOT = ROOT / "data/confirmation"
COLLECTION_HEADERS = ["filename", "merchant_id", "selected_at"]
LABEL_HEADERS = ["filename", "reviewer", *FIELDS]


def protocol_fingerprint():
    return digest(
        {
            name: digest((Path(__file__).parent / name).read_bytes())
            for name in ("confirmation.py", "novelty_evaluation.py", "evaluation.py")
        }
    )


def initialize_confirmation(folder=PRIVATE_ROOT):
    folder = Path(folder)
    (folder / "images").mkdir(parents=True, exist_ok=True)
    files = {
        "collection.csv": COLLECTION_HEADERS,
        "labels_a.csv": LABEL_HEADERS,
        "labels_b.csv": LABEL_HEADERS,
        "adjudications.csv": [*LABEL_HEADERS, "reason"],
        "exclusions.csv": ["filename", "reason", "recorded_at"],
    }
    for name, headers in files.items():
        path = folder / name
        if not path.exists():
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                csv.writer(stream).writerow(headers)
    readme = folder / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Private confirmation collection\n\n"
            "Collect 50 distinct English/Latin-script Indian printed receipts: 10 merchants, five transactions each. "
            "All four fields must be visibly present. Select receipts before inspecting model outputs. Put PNG/JPEG files in images/.\n\n"
            "1. Record filename, pseudonymous merchant ID (M01-M10), and ISO selection time in collection.csv.\n"
            "2. Two people independently fill labels_a.csv and labels_b.csv, using distinct reviewer IDs and verbatim company/address/date/total. CSV supports quoted multi-line addresses.\n"
            "3. A third reviewer resolves disagreements in adjudications.csv, providing all four final strings and a reason.\n"
            "4. Log collection exclusions in exclusions.csv before inference. Do not discard extraction failures later.\n"
            "5. Run confirmation-freeze, then confirmation-evaluate as documented in RUNNING.md.\n\n"
            "These CSVs, images, locks and per-receipt results must remain local. The tool can check distinct reviewer IDs and label agreement; it cannot prove human independence or collection timing.\n",
            encoding="utf-8",
        )
    return {
        "folder": str(folder),
        "target_receipts": 50,
        "target_merchants": 10,
        "status": "awaiting_real_receipts_and_annotations",
    }


def _rows(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(required) <= set(reader.fieldnames or []):
            raise ValueError(f"Missing CSV headers in {Path(path).name}")
        rows = list(reader)
    if any(not row.get("filename", "").strip() for row in rows):
        raise ValueError("CSV contains an empty filename")
    if len({row["filename"] for row in rows}) != len(rows):
        raise ValueError("Duplicate filenames in CSV")
    return rows


def freeze_confirmation(folder=PRIVATE_ROOT, *, name="confirmation-v1", checkpoint=None):
    from .novelty_evaluation import CHECKPOINT, checkpoint_identity, policy_fingerprint

    folder = Path(folder)
    checkpoint = checkpoint or CHECKPOINT
    if not name or any(
        ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in name
    ):
        raise ValueError("Use a simple confirmation freeze name")
    collection = _rows(folder / "collection.csv", COLLECTION_HEADERS)
    if len(collection) != 50:
        raise ValueError("Confirmation requires exactly 50 selected receipts")
    merchants = {}
    for row in collection:
        if not row["merchant_id"].strip() or not row["selected_at"].strip():
            raise ValueError("Collection requires merchant IDs and selection timestamps")
        merchants[row["merchant_id"]] = merchants.get(row["merchant_id"], 0) + 1
    if len(merchants) != 10 or set(merchants.values()) != {5}:
        raise ValueError("Confirmation requires ten merchants with five receipts each")
    a = {r["filename"]: r for r in _rows(folder / "labels_a.csv", LABEL_HEADERS)}
    b = {r["filename"]: r for r in _rows(folder / "labels_b.csv", LABEL_HEADERS)}
    adjudications = {
        r["filename"]: r for r in _rows(folder / "adjudications.csv", [*LABEL_HEADERS, "reason"])
    }
    filenames = {r["filename"] for r in collection}
    if set(a) != filenames or set(b) != filenames or not set(adjudications) <= filenames:
        raise ValueError("Both annotation files must cover exactly the selected receipts")
    excluded = _rows(folder / "exclusions.csv", ["filename", "reason", "recorded_at"])
    if any(
        not r["reason"].strip() or not r["recorded_at"].strip() or r["filename"] in filenames
        for r in excluded
    ):
        raise ValueError("Exclusions need a reason/time and must not be selected receipts")
    old_hashes = set()
    for split in ("train", "validation", "test"):
        path = ROOT / "data/manifests" / f"{split}.jsonl"
        if path.exists():
            old_hashes.update(e.get("sha256") for e in read_jsonl(path))
    entries, image_hashes = [], set()
    for row in collection:
        filename = row["filename"]
        image_path = (folder / "images" / filename).resolve()
        if not image_path.is_relative_to((folder / "images").resolve()):
            raise ValueError("Receipt path escapes private image folder")
        raw = image_path.read_bytes()
        if not raw or len(raw) > 10 * 1024 * 1024:
            raise ValueError("Confirmation image exceeds upload limits")
        sha = digest(raw)
        if sha in image_hashes or sha in old_hashes:
            raise ValueError("Confirmation image is duplicated or overlaps development data")
        image_hashes.add(sha)
        with Image.open(image_path) as image:
            if (
                image.format not in {"PNG", "JPEG"}
                or min(image.size) < 24
                or image.width * image.height > 40_000_000
            ):
                raise ValueError("Unsupported confirmation image")
            image.verify()
        left, right = a[filename], b[filename]
        reviewers = {left["reviewer"].strip(), right["reviewer"].strip()}
        if "" in reviewers or len(reviewers) != 2:
            raise ValueError("Two distinct reviewer IDs are required")
        if any(not left[f].strip() or not right[f].strip() for f in FIELDS):
            raise ValueError("All four visible fields need two annotations")
        agreement = all(clean_text(left[f]) == clean_text(right[f]) for f in FIELDS)
        reference = {f: left[f] for f in FIELDS}
        adjudicated = False
        if not agreement:
            resolution = adjudications.get(filename)
            if (
                not resolution
                or not resolution["reviewer"].strip()
                or resolution["reviewer"].strip() in reviewers
                or not resolution["reason"].strip()
                or any(not resolution[f].strip() for f in FIELDS)
            ):
                raise ValueError("Disagreements require complete third-reviewer adjudication and reason")
            reference = {f: resolution[f] for f in FIELDS}
            adjudicated = True
        entries.append(
            {
                "receipt_id": sha[:20],
                "source_key": filename,
                "sha256": sha,
                "image_path": str(image_path),
                "merchant_id": row["merchant_id"],
                "reference": reference,
                "split": "confirmation",
                "adjudicated": adjudicated,
            }
        )
    config = load_config()
    # A finished validation ablation using this exact policy must predate the freeze.
    candidates = list((ROOT / "artifacts/novelty").glob("*/summary.json"))
    validation = []
    ocr_fingerprint = get_engine().fingerprint
    for path in candidates:
        record = read_json(path)
        recorded_ocr = record.get("base_ocr_fingerprint")
        if recorded_ocr is None and (path.parent / "rules_none_predictions.jsonl").exists():
            observed = {
                p["ocr_fingerprint"] for p in read_jsonl(path.parent / "rules_none_predictions.jsonl")
            }
            recorded_ocr = next(iter(observed)) if len(observed) == 1 else None
        if (
            record.get("complete")
            and record.get("purpose", "").startswith("validation")
            and record.get("configuration_hash") == config.fingerprint
            and record.get("policy_fingerprint") == policy_fingerprint()
            and recorded_ocr == ocr_fingerprint
        ):
            validation.append(record)
    if not validation:
        raise ValueError("Complete validation ablations and freeze the current policy before confirmation")
    output = folder / "frozen" / name
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "manifest.jsonl"
    write_jsonl(manifest, entries)
    lock = {
        "created_at": utc_now(),
        "n_receipts": 50,
        "n_merchants": 10,
        "manifest_sha256": digest(manifest.read_bytes()),
        "configuration_hash": config.fingerprint,
        "policy_fingerprint": policy_fingerprint(),
        "base_ocr_fingerprint": ocr_fingerprint,
        "protocol_fingerprint": protocol_fingerprint(),
        "checkpoint": checkpoint_identity(checkpoint, True),
        "annotation_files": {
            name: digest((folder / name).read_bytes())
            for name in [
                "collection.csv",
                "labels_a.csv",
                "labels_b.csv",
                "adjudications.csv",
                "exclusions.csv",
            ]
        },
        "private_collection": str(folder.resolve()),
        "annotation_check_scope": "distinct supplied reviewer IDs, agreement/adjudication; human independence is an attestation",
    }
    write_json(output / "lock.json", lock)
    return {"manifest": str(manifest), "lock": str(output / "lock.json"), "status": "frozen"}


def verify_confirmation(manifest, lock_path, *, checkpoint):
    from .novelty_evaluation import checkpoint_identity, policy_fingerprint

    manifest, lock = Path(manifest), read_json(lock_path)
    if digest(manifest.read_bytes()) != lock["manifest_sha256"]:
        raise ValueError("Confirmation labels/manifest changed after freeze")
    if (
        load_config().fingerprint != lock["configuration_hash"]
        or policy_fingerprint() != lock["policy_fingerprint"]
    ):
        raise ValueError("Confirmation policy changed after freeze")
    if checkpoint_identity(checkpoint, True) != lock["checkpoint"]:
        raise ValueError("Confirmation checkpoint changed after freeze")
    if get_engine().fingerprint != lock["base_ocr_fingerprint"]:
        raise ValueError("Confirmation OCR engine changed after freeze")
    if protocol_fingerprint() != lock["protocol_fingerprint"]:
        raise ValueError("Confirmation evaluation protocol changed after freeze")
    folder = Path(lock["private_collection"])
    for name, sha in lock["annotation_files"].items():
        if digest((folder / name).read_bytes()) != sha:
            raise ValueError("Confirmation annotation source changed after freeze")
    entries = read_jsonl(manifest)
    if len(entries) != 50 or any(e.get("split") != "confirmation" for e in entries):
        raise ValueError("Invalid confirmation split")
    for entry in entries:
        if digest(Path(entry["image_path"]).read_bytes()) != entry["sha256"]:
            raise ValueError("Confirmation receipt changed after freeze")
    return True
