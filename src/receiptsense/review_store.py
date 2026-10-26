"""Append-only predictions and review versions in local SQLite."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .common import FIELDS, ROOT, utc_now
from .contracts import FieldPrediction, PredictionRecord, ReviewedRecord
from .validation import validate_field


class ReviewStore:
    def __init__(self, path: str | Path = ROOT / "artifacts/receiptsense.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS predictions (
                  prediction_id TEXT PRIMARY KEY, receipt_id TEXT NOT NULL,
                  created_at TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS reviews (
                  review_id TEXT PRIMARY KEY, prediction_id TEXT NOT NULL,
                  created_at TEXT NOT NULL, payload TEXT NOT NULL,
                  FOREIGN KEY(prediction_id) REFERENCES predictions(prediction_id));
                CREATE TABLE IF NOT EXISTS correction_events (
                  id INTEGER PRIMARY KEY AUTOINCREMENT, review_id TEXT NOT NULL,
                  field TEXT NOT NULL, previous_value TEXT, edited_value TEXT,
                  source TEXT NOT NULL, created_at TEXT NOT NULL,
                  FOREIGN KEY(review_id) REFERENCES reviews(review_id));
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()

    def save_prediction(self, prediction: PredictionRecord) -> None:
        payload = prediction.model_dump_json()
        with self.connect() as db:
            existing = db.execute(
                "SELECT payload FROM predictions WHERE prediction_id=?", (prediction.prediction_id,)
            ).fetchone()
            if existing:
                if PredictionRecord.model_validate_json(existing[0]) != prediction:
                    raise ValueError("Automated predictions are immutable")
                return
            db.execute(
                "INSERT INTO predictions VALUES (?,?,?,?)",
                (prediction.prediction_id, prediction.receipt_id, prediction.created_at, payload),
            )

    def get_prediction(self, prediction_id: str) -> PredictionRecord:
        with self.connect() as db:
            row = db.execute(
                "SELECT payload FROM predictions WHERE prediction_id=?", (prediction_id,)
            ).fetchone()
        if row is None:
            raise KeyError(prediction_id)
        return PredictionRecord.model_validate_json(row[0])

    def get_review(self, review_id: str) -> ReviewedRecord:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM reviews WHERE review_id=?", (review_id,)).fetchone()
        if row is None:
            raise KeyError(review_id)
        return ReviewedRecord.model_validate_json(row[0])

    def latest_review(self, prediction_id: str) -> ReviewedRecord | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT payload FROM reviews WHERE prediction_id=? ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (prediction_id,),
            ).fetchone()
        return ReviewedRecord.model_validate_json(row[0]) if row else None

    def save_review(
        self, prediction_id: str, values: dict[str, str | None], *, confirm: bool = False
    ) -> ReviewedRecord:
        if set(values) != set(FIELDS):
            raise ValueError("Review requires exactly the four receipt fields")
        prediction = self.get_prediction(prediction_id)
        prior_review = self.latest_review(prediction_id)
        prior_values = (
            prior_review.values if prior_review else {n: f.raw_value for n, f in prediction.fields.items()}
        )
        checked = {
            name: validate_field(name, FieldPrediction(raw_value=values[name], status="predicted"))
            for name in FIELDS
        }
        edits = [
            {
                "field": name,
                "previous_value": prior_values[name],
                "edited_value": values[name],
                "source": "user_entered",
                "created_at": utc_now(),
            }
            for name in FIELDS
            if values[name] != prior_values[name]
        ]
        review = ReviewedRecord(
            prediction_id=prediction_id,
            receipt_id=prediction.receipt_id,
            state="confirmed" if confirm else "edited",
            values=values,
            normalized_values={name: checked[name].normalized_value for name in FIELDS},
            review_reasons={name: checked[name].review_reasons for name in FIELDS},
            edits=edits,
        )
        with self.connect() as db:
            db.execute(
                "INSERT INTO reviews VALUES (?,?,?,?)",
                (review.review_id, prediction_id, review.created_at, review.model_dump_json()),
            )
            for event in edits:
                db.execute(
                    "INSERT INTO correction_events(review_id,field,previous_value,edited_value,source,created_at) VALUES (?,?,?,?,?,?)",
                    (
                        review.review_id,
                        event["field"],
                        event["previous_value"],
                        event["edited_value"],
                        event["source"],
                        event["created_at"],
                    ),
                )
        return review

    def recent(self, limit: int = 20) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT payload FROM predictions ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 200)),)
            ).fetchall()
        records = [json.loads(row[0]) for row in rows]
        for record in records:
            latest = self.latest_review(record["prediction_id"])
            record["latest_review"] = latest.model_dump(mode="json") if latest else None
        return records
