"""Optional local HTTP API around the shared pipeline and immutable review store."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from receiptsense.common import ROOT, read_json, write_json
from receiptsense.export import csv_export, json_export
from receiptsense.imaging import ImageInputError, prepare_image
from receiptsense.model import trained_bundle_available
from receiptsense.pipeline import process_receipt
from receiptsense.recovery_config import Enhancement
from receiptsense.review_store import ReviewStore

app = FastAPI(title="ReceiptSense", version="0.1.0", description="Local receipt extraction and review API")


@lru_cache
def get_store():
    return ReviewStore()


class ExtractRequest(BaseModel):
    method: Literal["rules", "layoutlmv3"] = "rules"
    variant: Literal["identity", "deskew", "contrast"] = "identity"
    enhancement: Enhancement = "none"


class ReviewRequest(BaseModel):
    values: dict[str, str | None]
    confirm: bool = False


def uploaded(receipt_id: str) -> tuple[Path, dict]:
    if not re.fullmatch(r"[a-f0-9]{20}", receipt_id):
        raise HTTPException(404, "Receipt not found")
    folder = ROOT / "artifacts/uploads" / receipt_id
    if not (folder / "source.bin").exists():
        raise HTTPException(404, "Receipt not found")
    return folder, read_json(folder / "metadata.json")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "trained_model_available": trained_bundle_available(
            ROOT / "artifacts/models/receiptsense-layoutlmv3"
        ),
    }


@app.post("/receipts", status_code=201)
async def create_receipt(file: UploadFile = File(...)):
    data = await file.read(10 * 1024 * 1024 + 1)
    try:
        image = prepare_image(data, file.filename or "receipt.png")
    except ImageInputError as exc:
        raise HTTPException(422, str(exc)) from exc
    folder = ROOT / "artifacts/uploads" / image.receipt_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "source.bin").write_bytes(data)
    write_json(folder / "metadata.json", image.model_dump(mode="json"))
    return image


@app.post("/receipts/{receipt_id}/extract")
def extract(receipt_id: str, request: ExtractRequest):
    folder, metadata = uploaded(receipt_id)
    try:
        result = process_receipt(
            (folder / "source.bin").read_bytes(),
            metadata["filename"],
            method=request.method,
            variant=request.variant,
            enhancement=request.enhancement,
        )
        get_store().save_prediction(result.prediction)
        return result.prediction
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/predictions/{prediction_id}")
def get_prediction(prediction_id: str):
    try:
        return get_store().get_prediction(prediction_id)
    except KeyError as exc:
        raise HTTPException(404, "Prediction not found") from exc


@app.post("/predictions/{prediction_id}/reviews", status_code=201)
def review(prediction_id: str, request: ReviewRequest):
    try:
        return get_store().save_review(prediction_id, request.values, confirm=request.confirm)
    except KeyError as exc:
        raise HTTPException(404, "Prediction not found") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/reviews/{review_id}/export")
def export_review(review_id: str, format: Literal["json", "csv"] = "json"):
    try:
        reviewed = get_store().get_review(review_id)
        prediction = get_store().get_prediction(reviewed.prediction_id)
    except KeyError as exc:
        raise HTTPException(404, "Review not found") from exc
    payload = json_export(prediction, reviewed) if format == "json" else csv_export(prediction, reviewed)
    return Response(
        payload,
        media_type="application/json" if format == "json" else "text/csv",
        headers={"Content-Disposition": f'attachment; filename="receipt_{reviewed.receipt_id}.{format}"'},
    )
