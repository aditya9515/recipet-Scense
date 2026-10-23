"""Validated records: one schema for OCR, both extractors, review and evaluation."""

from __future__ import annotations

import math
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import FIELDS, utc_now


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImageRecord(Record):
    schema_version: str = "1.0"
    receipt_id: str
    source_sha256: str
    filename: str
    source_path: str | None = None
    raw_size: tuple[int, int]
    display_size: tuple[int, int]
    processed_size: tuple[int, int]
    display_path: str
    processed_path: str
    processed_sha256: str
    preprocessing_fingerprint: str
    display_to_processed: list[list[float]]
    exif_orientation: int = 1
    operations: list[str] = Field(default_factory=list)
    quality: dict[str, float] = Field(default_factory=dict)


class OCRUnit(Record):
    unit_id: str
    text: str
    granularity: Literal["word", "line"] = "word"
    polygon: list[tuple[float, float]]
    bbox_xyxy: tuple[float, float, float, float]
    recognition_score: float | None = None
    line_id: str
    reading_order: int = 0
    box_source: Literal["engine", "replicated_line", "reference"] = "engine"

    @model_validator(mode="after")
    def valid_geometry(self):
        if len(self.polygon) != 4 or not self.text.strip():
            raise ValueError("Each text region needs text and four corners")
        values = [v for point in self.polygon for v in point] + list(self.bbox_xyxy)
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Region coordinates must be finite")
        x1, y1, x2, y2 = self.bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Text regions need positive width and height")
        if self.recognition_score is not None and not 0 <= self.recognition_score <= 1:
            raise ValueError("Recognition score must lie in [0,1]")
        return self


class OCRRecord(Record):
    schema_version: str = "1.0"
    receipt_id: str
    image_frame: Literal["processed"] = "processed"
    image_width: int
    image_height: int
    ocr_fingerprint: str
    engine: dict = Field(default_factory=dict)
    units: list[OCRUnit] = Field(default_factory=list)
    status: Literal["ok", "empty_ocr"] = "ok"

    @model_validator(mode="after")
    def valid_units(self):
        ids = [u.unit_id for u in self.units]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate OCR unit IDs")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Image dimensions must be positive")
        for unit in self.units:
            for x, y in unit.polygon:
                if not (-1 <= x <= self.image_width + 1 and -1 <= y <= self.image_height + 1):
                    raise ValueError("OCR polygon is outside its image frame")
        if not self.units:
            self.status = "empty_ocr"
        return self


class FieldPrediction(Record):
    raw_value: str | None = None
    normalized_value: str | None = None
    evidence_unit_ids: list[str] = Field(default_factory=list)
    status: Literal["predicted", "ambiguous", "missing", "invalid"] = "missing"
    score: float | None = None
    score_kind: str | None = None
    review_reasons: list[str] = Field(default_factory=list)
    normalization_actions: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    currency: str | None = None


class PredictionRecord(Record):
    schema_version: str = "1.0"
    prediction_id: str = Field(default_factory=lambda: uuid4().hex)
    receipt_id: str
    created_at: str = Field(default_factory=utc_now)
    method: str
    fields: dict[str, FieldPrediction]
    ocr_fingerprint: str
    model_revision: str
    decoder_version: str = "decode-v1"
    validation_version: str = "parse-v1"
    configuration_hash: str = ""
    timings: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def four_fields(self):
        if set(self.fields) != set(FIELDS):
            raise ValueError("Prediction must contain exactly company/address/date/total")
        return self

    def check_evidence(self, ocr: OCRRecord) -> None:
        if ocr.receipt_id != self.receipt_id:
            raise ValueError("Prediction and OCR refer to different receipts")
        ids = {u.unit_id for u in ocr.units}
        for field in self.fields.values():
            if not set(field.evidence_unit_ids) <= ids:
                raise ValueError("Prediction references nonexistent OCR evidence")
            if field.raw_value and not field.evidence_unit_ids:
                raise ValueError("Automated values require OCR evidence")


class ReviewedRecord(Record):
    schema_version: str = "1.0"
    review_id: str = Field(default_factory=lambda: uuid4().hex)
    prediction_id: str
    receipt_id: str
    created_at: str = Field(default_factory=utc_now)
    state: Literal["edited", "confirmed"]
    values: dict[str, str | None]
    normalized_values: dict[str, str | None]
    review_reasons: dict[str, list[str]]
    edits: list[dict]

    @model_validator(mode="after")
    def four_values(self):
        if set(self.values) != set(FIELDS) or set(self.normalized_values) != set(FIELDS):
            raise ValueError("Reviewed records require all four field keys")
        return self
