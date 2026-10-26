"""One processing path for CLI, UI and API; no silent method fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .common import ROOT, write_json
from .contracts import ImageRecord, OCRRecord, PredictionRecord
from .imaging import prepare_image
from .ocr_adapter import run_ocr
from .rules import extract_rules
from .validation import validate_fields


@dataclass
class PipelineResult:
    image: ImageRecord
    ocr: OCRRecord
    prediction: PredictionRecord
    cache_hit: bool


def process_receipt(
    data: bytes,
    filename: str = "receipt.png",
    *,
    method: str = "rules",
    variant: str = "identity",
    checkpoint: str | Path | None = None,
    artifact_root: str | Path = ROOT / "artifacts",
    use_cache: bool = True,
    engine=None,
    model=None,
    enhancement: str = "none",
    recovery_config=None,
) -> PipelineResult:
    from .recovery_config import ENHANCEMENTS, load_config

    if enhancement not in ENHANCEMENTS:
        raise ValueError(f"Unknown enhancement: {enhancement}")
    start = perf_counter()
    image = prepare_image(data, filename, variant=variant, artifact_root=artifact_root)
    image_done = perf_counter()
    ocr, hit = run_ocr(image, artifact_root=artifact_root, use_cache=use_cache, engine=engine)
    if (ocr.image_width, ocr.image_height) != image.processed_size:
        raise ValueError("OCR coordinates and selected image dimensions do not match")
    ocr_done = perf_counter()
    if method == "rules":
        extract = extract_rules
    elif method == "layoutlmv3":
        from .model import get_extractor

        extractor = model or get_extractor(
            str(checkpoint or Path(artifact_root) / "models/receiptsense-layoutlmv3")
        )

        def extract(record):
            return extractor.predict(image, record)
    else:
        raise ValueError(f"Unknown method: {method}")
    prediction = extract(ocr)
    extraction_done = perf_counter()
    prediction = validate_fields(prediction)
    validation_done = perf_counter()
    base_ocr = ocr
    if enhancement != "none":
        from .recovery import enhance_prediction

        ocr, prediction = enhance_prediction(
            image,
            ocr,
            prediction,
            mode=enhancement,
            config=recovery_config or load_config(),
            extract=extract,
            method=method,
            artifact_root=artifact_root,
            use_cache=use_cache,
            engine=engine,
        )
    recovery_done = perf_counter()
    prediction.check_evidence(ocr)
    prediction.timings = {
        "image_seconds": image_done - start,
        "ocr_seconds": ocr_done - image_done,
        "extraction_seconds": extraction_done - ocr_done,
        "validation_seconds": validation_done - extraction_done + perf_counter() - recovery_done,
        "total_seconds": perf_counter() - start,
        "ocr_cache_hit": float(hit),
        "recovery_seconds": prediction.recovery.elapsed_seconds if prediction.recovery else 0,
        "recovery_attempts": prediction.recovery.logical_attempts if prediction.recovery else 0,
        "uncached_ocr_calls": int(not hit)
        + (prediction.recovery.uncached_ocr_calls if prediction.recovery else 0),
    }
    output = Path(artifact_root) / "predictions" / prediction.prediction_id
    write_json(output / "image.json", image.model_dump(mode="json"))
    write_json(output / "ocr.json", ocr.model_dump(mode="json"))
    if enhancement != "none":
        write_json(output / "base_ocr.json", base_ocr.model_dump(mode="json"))
    write_json(output / "prediction.json", prediction.model_dump(mode="json"))
    return PipelineResult(image, ocr, prediction, hit)
