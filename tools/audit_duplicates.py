"""Flag visually near-identical cross-split images for review; never auto-drop by pHash."""

import cv2
import numpy as np
from PIL import Image, ImageOps

from receiptsense.common import ROOT, read_jsonl, write_json


def thumbnail(path):
    with Image.open(path) as image:
        image.draft("L", (256, 256))
        return np.asarray(ImageOps.exif_transpose(image).convert("L").resize((256, 256)), dtype=np.float32)


def main():
    rows = []
    for split in ["train", "validation", "test"]:
        for entry in read_jsonl(ROOT / "data/manifests" / f"{split}.jsonl"):
            thumb = thumbnail(entry["image_path"])
            dct = cv2.dct(cv2.resize(thumb, (32, 32)))[:8, :8]
            bits = (dct > np.median(dct[1:, :])).reshape(-1)
            value = int.from_bytes(np.packbits(bits).tobytes(), "big")
            rows.append((entry, value, thumb))
    candidates = []
    for i, (a, ha, ta) in enumerate(rows):
        for b, hb, tb in rows[i + 1 :]:
            if a["split"] == b["split"] or (ha ^ hb).bit_count() > 2:
                continue
            error = float(np.mean(np.abs(ta - tb)))
            if error < 2:
                candidates.append(
                    {
                        "first_id": a["receipt_id"],
                        "first_key": a["source_key"],
                        "first_split": a["split"],
                        "second_id": b["receipt_id"],
                        "second_key": b["source_key"],
                        "second_split": b["split"],
                        "phash_hamming": (ha ^ hb).bit_count(),
                        "thumbnail_mae_255_scale": error,
                        "status": "candidate_requires_visual_review_not_automatically_excluded",
                    }
                )
    candidates.sort(key=lambda r: r["thumbnail_mae_255_scale"])
    output = {
        "receipts_checked": len(rows),
        "policy": "cross-split pHash Hamming <=2 and thumbnail MAE <2/255; manual review required",
        "candidates": candidates,
        "warning": "Similar receipt templates can be false positives; these are not confirmed duplicates.",
    }
    write_json(ROOT / "data/near_duplicate_audit.json", output)
    print(output)


if __name__ == "__main__":
    main()
