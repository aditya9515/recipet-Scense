"""Versioned CPU recovery policy; confidence values are heuristics, not calibration."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from .common import ROOT, digest, read_json
from .contracts import Record

Enhancement = Literal["none", "address", "retry", "combined"]
ENHANCEMENTS = ("none", "address", "retry", "combined")


class RecoveryConfig(Record):
    version: str = "recovery-v1"
    max_attempts: int = Field(default=2, ge=0, le=2)
    max_crop_side: int = Field(default=1600, ge=24, le=1600)
    upscale: float = Field(default=2.0, ge=1, le=2)
    crop_padding_heights: float = Field(default=2.0, ge=0, le=5)
    low_ocr_score: float = Field(default=0.8, ge=0, le=1)
    low_model_score: float = Field(default=0.6, ge=0, le=1)
    header_fraction: float = Field(default=0.5, gt=0, le=0.75)
    payment_start: float = Field(default=0.45, ge=0, lt=1)
    address_gap_heights: float = Field(default=2.0, gt=0, le=4)
    address_overlap: float = Field(default=0.35, ge=0, le=1)
    address_max_lines: int = Field(default=6, ge=1, le=8)
    address_min_score: float = 5.5
    address_margin: float = Field(default=1.25, ge=0)
    address_gain: float = Field(default=0.75, ge=0)

    @property
    def fingerprint(self):
        return digest(self.model_dump(mode="json"))


def load_config(path: str | Path | None = None) -> RecoveryConfig:
    path = Path(path or ROOT / "configs/recovery-v1.json")
    return RecoveryConfig.model_validate(read_json(path))
