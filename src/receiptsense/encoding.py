"""Word-boundary windows with deterministic overlap ownership and first-subword labels."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from PIL import Image

from .common import LABELS
from .contracts import ImageRecord, OCRRecord


def normalized_boxes(ocr: OCRRecord) -> list[list[int]]:
    boxes = []
    for unit in ocr.units:
        x1, y1, x2, y2 = unit.bbox_xyxy
        box = [
            round(1000 * x1 / ocr.image_width),
            round(1000 * y1 / ocr.image_height),
            round(1000 * x2 / ocr.image_width),
            round(1000 * y2 / ocr.image_height),
        ]
        box = [max(0, min(1000, value)) for value in box]
        if box[0] == box[2]:
            box[0], box[2] = max(0, box[0] - 1), min(1000, box[2] + 1)
        if box[1] == box[3]:
            box[1], box[3] = max(0, box[1] - 1), min(1000, box[3] + 1)
        boxes.append(box)
    return boxes


def window_ranges(counts: list[int], budget: int = 510, overlap: int = 32) -> list[tuple[int, int]]:
    if any(c <= 0 or c > budget for c in counts):
        raise ValueError("An OCR unit is empty or longer than the model text budget")
    ranges = []
    start = 0
    while start < len(counts):
        end, used = start, 0
        while end < len(counts) and used + counts[end] <= budget:
            used += counts[end]
            end += 1
        ranges.append((start, end))
        if end == len(counts):
            break
        start = max(start + 1, end - overlap)
    return ranges


def window_owners(ranges: list[tuple[int, int]], n_units: int) -> dict[int, int]:
    owners = {}
    for index in range(n_units):
        candidates = [
            (min(index - start, end - 1 - index), -window, window)
            for window, (start, end) in enumerate(ranges)
            if start <= index < end
        ]
        if not candidates:
            raise ValueError("Windowing lost an OCR unit")
        owners[index] = max(candidates)[2]
    return owners


@dataclass
class EncodedWindow:
    inputs: dict
    positions: dict[int, int]
    labels: object | None


def encode_document(
    processor, image: ImageRecord, ocr: OCRRecord, labels: list[str] | None = None, max_length: int = 512
) -> list[EncodedWindow]:
    import torch

    if not ocr.units:
        return []
    words = [u.text for u in ocr.units]
    boxes = normalized_boxes(ocr)
    if labels is not None and len(labels) != len(words):
        raise ValueError("Label count does not match OCR units")
    untruncated = processor.tokenizer(words, boxes=boxes, add_special_tokens=False, truncation=False)
    counts = Counter(untruncated.word_ids())
    budget = max_length - processor.tokenizer.num_special_tokens_to_add()
    ranges = window_ranges([counts[i] for i in range(len(words))], budget)
    owners = window_owners(ranges, len(words))
    selected_image = Image.open(image.processed_path).convert("RGB")
    windows = []
    for index, (start, end) in enumerate(ranges):
        # A single window needs no padding to 512; retain every real text unit.
        encoding = processor(
            images=selected_image,
            text=words[start:end],
            boxes=boxes[start:end],
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        if encoding["input_ids"].shape[1] > max_length:
            raise ValueError("Window exceeded the verified tokenizer budget")
        positions = {}
        for token_index, word_index in enumerate(encoding.word_ids(batch_index=0)):
            if word_index is not None:
                global_index = start + word_index
                if owners[global_index] == index:
                    positions.setdefault(global_index, token_index)
        targets = None
        if labels is not None:
            targets = torch.full_like(encoding["input_ids"], -100)
            for global_index, token_index in positions.items():
                label = labels[global_index]
                if label != "IGNORE":
                    targets[0, token_index] = LABELS.index(label)
        windows.append(EncodedWindow(dict(encoding), positions, targets))
    return windows
