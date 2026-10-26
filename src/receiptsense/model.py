"""Inference requires a trained ReceiptSense bundle; a base checkpoint is rejected."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .common import LABELS, digest, read_json
from .contracts import ImageRecord, OCRRecord, PredictionRecord
from .decoding import decode_fields
from .encoding import encode_document


def trained_bundle_available(checkpoint: str | Path) -> bool:
    folder = Path(checkpoint)
    try:
        metadata = read_json(folder / "receiptsense_model.json")
        return bool(
            metadata.get("trained")
            and metadata.get("labels") == LABELS
            and all(
                (folder / name).is_file()
                for name in ["model.safetensors", "config.json", "tokenizer.json", "preprocessor_config.json"]
            )
        )
    except (OSError, ValueError):
        return False


class LayoutExtractor:
    def __init__(self, checkpoint: str | Path, device: str | None = None):
        import torch
        from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

        self.path = Path(checkpoint)
        manifest = self.path / "receiptsense_model.json"
        if not manifest.exists():
            raise ValueError(
                "No trained ReceiptSense model was found. Run the training command or select Rules."
            )
        self.metadata = read_json(manifest)
        if self.metadata.get("labels") != LABELS or not self.metadata.get("trained"):
            raise ValueError("Checkpoint has an incompatible or untrained receipt head")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.set_num_threads(4)
        self.processor = LayoutLMv3Processor.from_pretrained(
            self.path, apply_ocr=False, local_files_only=True
        )
        self.model = (
            LayoutLMv3ForTokenClassification.from_pretrained(self.path, local_files_only=True)
            .to(self.device)
            .eval()
        )
        actual_labels = [self.model.config.id2label[i] for i in range(len(LABELS))]
        if actual_labels != LABELS:
            raise ValueError("Model label mapping differs from the saved training manifest")
        self.revision = digest(self.metadata)

    def predict(self, image: ImageRecord, ocr: OCRRecord) -> PredictionRecord:
        import torch

        labels, scores = ["O"] * len(ocr.units), [0.0] * len(ocr.units)
        windows = encode_document(self.processor, image, ocr)
        with torch.inference_mode():
            for window in windows:
                logits = self.model(**{k: v.to(self.device) for k, v in window.inputs.items()}).logits
                probabilities = logits[0].softmax(-1).cpu()
                for unit_index, token_index in window.positions.items():
                    score, index = probabilities[token_index].max(-1)
                    labels[unit_index], scores[unit_index] = LABELS[int(index)], float(score)
        prediction = PredictionRecord(
            receipt_id=ocr.receipt_id,
            method="layoutlmv3",
            fields=decode_fields(ocr, labels, scores),
            ocr_fingerprint=ocr.ocr_fingerprint,
            model_revision=self.revision,
            configuration_hash=digest({"model": self.revision, "ocr": ocr.ocr_fingerprint}),
        )
        prediction.check_evidence(ocr)
        return prediction


@lru_cache(maxsize=1)
def get_extractor(checkpoint: str) -> LayoutExtractor:
    return LayoutExtractor(checkpoint)
