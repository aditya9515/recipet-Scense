"""Paddle-family OCR via ONNX or native Paddle; geometry precision is explicit."""

from __future__ import annotations

import importlib.metadata
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from .common import ROOT, digest, read_json, write_json
from .contracts import ImageRecord, OCRRecord, OCRUnit
from .reading_order import order_units

OCR_CONFIG = {
    "detector": "PP-OCRv4_mobile_det",
    "recognizer": "en_PP-OCRv4_mobile_rec",
    "language": "en",
    "device": "cpu",
    "cpu_threads": 4,
    "orientation": False,
    "unwarping": False,
    "mkldnn": False,
    "detection_max_side": 960,
    "detection_limit_type": "max",
    "recognition_batch": 16,
    "unit_policy": "split_text_replicate_line_geometry",
    "adapter": "paddle-v2",
}


class OCRError(RuntimeError):
    pass


def canonicalize(image, texts, scores, polygons, metadata, fingerprint):
    if not (len(texts) == len(scores) == len(polygons)):
        raise OCRError("OCR returned inconsistent text/score/region counts")
    width, height = image.processed_size
    units = []
    for index, (text, score, polygon) in enumerate(zip(texts, scores, polygons, strict=True)):
        points = np.asarray(polygon, dtype=float)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise OCRError("OCR returned invalid polygon geometry")
        points[:, 0] = np.clip(points[:, 0], 0, width)
        points[:, 1] = np.clip(points[:, 1], 0, height)
        bbox = (*points.min(axis=0).tolist(), *points.max(axis=0).tolist())
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        words = str(text).split()
        for word_index, word in enumerate(words):
            units.append(
                OCRUnit(
                    unit_id=f"r{index:04d}w{word_index:03d}",
                    text=word,
                    polygon=points.tolist(),
                    bbox_xyxy=bbox,
                    recognition_score=float(score),
                    line_id=f"l{index:04d}",
                    reading_order=len(units),
                    box_source="replicated_line" if len(words) > 1 else "engine",
                )
            )
    return OCRRecord(
        receipt_id=image.receipt_id,
        image_width=width,
        image_height=height,
        ocr_fingerprint=fingerprint,
        engine=metadata,
        units=order_units(units),
    )


class OnnxEngine:
    """PP-OCRv4 models packaged by RapidOCR; optimized CPU deployment."""

    def __init__(self):
        import rapidocr_onnxruntime
        from rapidocr_onnxruntime import RapidOCR

        self.engine = RapidOCR(intra_op_num_threads=4, inter_op_num_threads=1)
        folder = Path(rapidocr_onnxruntime.__file__).parent / "models"
        hashes = {path.name: digest(path.read_bytes()) for path in sorted(folder.glob("*.onnx"))}
        self.metadata = {
            "backend": "RapidOCR/ONNX Runtime",
            "model_family": "PP-OCRv4",
            "versions": {
                name: importlib.metadata.version(name) for name in ["rapidocr-onnxruntime", "onnxruntime"]
            },
            "artifact_hashes": hashes,
            "device": "cpu",
            "intra_op_threads": 4,
            "inter_op_threads": 1,
            "classification": False,
            "unit_policy": "split_text_replicate_line_geometry",
            "adapter": "onnx-v1",
        }
        self.fingerprint = digest(self.metadata)

    def recognize(self, image):
        try:
            result, _ = self.engine(image.processed_path, use_cls=False)
            result = result or []
            return canonicalize(
                image,
                [r[1] for r in result],
                [r[2] for r in result],
                [r[0] for r in result],
                self.metadata,
                self.fingerprint,
            )
        except OCRError:
            raise
        except Exception as exc:
            raise OCRError(f"ONNX OCR failed for this receipt: {exc}") from exc


class PaddleEngine:
    def __init__(self):
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        try:
            from paddleocr import PaddleOCR

            self.engine = PaddleOCR(
                text_detection_model_name=OCR_CONFIG["detector"],
                text_recognition_model_name=OCR_CONFIG["recognizer"],
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="cpu",
                cpu_threads=4,
                text_det_limit_side_len=960,
                text_det_limit_type="max",
                text_recognition_batch_size=16,
                enable_mkldnn=False,
            )
        except Exception as exc:
            raise OCRError(f"PaddleOCR could not start. Check the OCR installation/models: {exc}") from exc
        versions = {package: importlib.metadata.version(package) for package in ["paddleocr", "paddlepaddle"]}
        hashes = {}
        for name in [OCR_CONFIG["detector"], OCR_CONFIG["recognizer"]]:
            folder = Path.home() / ".paddlex" / "official_models" / name
            for path in sorted(folder.glob("*")):
                if path.is_file() and path.suffix in {".json", ".yml", ".yaml", ".pdiparams"}:
                    hashes[f"{name}/{path.name}"] = digest(path.read_bytes())
        self.metadata = {**OCR_CONFIG, "versions": versions, "artifact_hashes": hashes}
        self.fingerprint = digest(self.metadata)

    def recognize(self, image: ImageRecord) -> OCRRecord:
        try:
            results = list(self.engine.predict(image.processed_path))
            if len(results) != 1:
                raise OCRError("Expected one OCR result for the selected receipt")
            raw = results[0].json
            if isinstance(raw, str):
                raw = json.loads(raw)
            raw = raw.get("res", raw)
            texts, scores = raw.get("rec_texts", []), raw.get("rec_scores", [])
            polygons = raw.get("rec_polys", raw.get("dt_polys", []))
            return canonicalize(image, texts, scores, polygons, self.metadata, self.fingerprint)
        except OCRError:
            raise
        except Exception as exc:
            raise OCRError(f"OCR failed for this image: {exc}") from exc


@lru_cache(maxsize=1)
def get_engine():
    backend = os.environ.get("RECEIPTSENSE_OCR_BACKEND", "onnx")
    if backend == "onnx":
        return OnnxEngine()
    if backend == "paddle":
        return PaddleEngine()
    raise OCRError("RECEIPTSENSE_OCR_BACKEND must be onnx or paddle")


def run_ocr(
    image: ImageRecord, *, artifact_root: str | Path = ROOT / "artifacts", use_cache: bool = True, engine=None
) -> tuple[OCRRecord, bool]:
    engine = engine or get_engine()
    key = digest(
        {
            "image": image.processed_sha256,
            "preprocessing": image.preprocessing_fingerprint,
            "engine": engine.fingerprint,
        }
    )
    path = Path(artifact_root) / "ocr" / f"{key}.json"
    if use_cache and path.exists():
        result = OCRRecord.model_validate(read_json(path))
        if result.receipt_id != image.receipt_id or result.ocr_fingerprint != engine.fingerprint:
            raise OCRError("OCR cache provenance mismatch")
        return result, True
    result = engine.recognize(image)
    write_json(path, result.model_dump(mode="json"))
    return result, False
