import io

import pytest
from PIL import Image

from receiptsense.contracts import OCRRecord, OCRUnit


@pytest.fixture
def receipt_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (620, 900), "white").save(stream, format="PNG")
    return stream.getvalue()


def make_ocr(lines, receipt_id="sample"):
    units = []
    for index, text in enumerate(lines):
        y = 20 + index * 45
        for word_index, word in enumerate(text.split()):
            x = 20 + word_index * 65
            units.append(
                OCRUnit(
                    unit_id=f"l{index}w{word_index}",
                    text=word,
                    polygon=[(x, y), (x + 60, y), (x + 60, y + 30), (x, y + 30)],
                    bbox_xyxy=(x, y, x + 60, y + 30),
                    recognition_score=0.95,
                    line_id=f"l{index}",
                    reading_order=len(units),
                )
            )
    return OCRRecord(
        receipt_id=receipt_id, image_width=1000, image_height=2000, ocr_fingerprint="fixture-v1", units=units
    )


@pytest.fixture
def clear_ocr():
    return make_ocr(
        [
            "EXAMPLE STORE",
            "12 MAIN ROAD",
            "CENTRAL CITY",
            "DATE 2025-02-14",
            "SUBTOTAL 470.00",
            "TAX 23.50",
            "TOTAL 493.50",
            "CASH 500.00",
            "CHANGE 6.50",
        ]
    )
