"""Constructed receipts for transparent demos and integration checks, not benchmark data."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .common import ROOT, write_json


def create_samples(folder: str | Path = ROOT / "examples") -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    font_path = Path("C:/Windows/Fonts/consola.ttf")
    font = ImageFont.truetype(str(font_path), 26) if font_path.exists() else ImageFont.load_default(size=26)
    definitions = [
        (
            "clear",
            [
                "EXAMPLE STORE",
                "12 MAIN ROAD",
                "CENTRAL CITY 560001",
                "DATE 2025-02-14",
                "",
                "NOTEBOOK                 470.00",
                "SUBTOTAL                 470.00",
                "TAX                       23.50",
                "TOTAL                    493.50",
                "CASH                     500.00",
                "CHANGE                     6.50",
                "",
                "THANK YOU",
            ],
            {
                "company": "EXAMPLE STORE",
                "address": "12 MAIN ROAD CENTRAL CITY 560001",
                "date": "2025-02-14",
                "total": "493.50",
            },
        ),
        (
            "ambiguous",
            [
                "RIVER BOOK STORE",
                "24 RIVER ROAD",
                "CENTRAL CITY 560002",
                "DATE 03/04/2025",
                "",
                "BOOKS                     40.00",
                "TOTAL                     40.00",
                "CASH                      50.00",
                "CHANGE                    10.00",
            ],
            {
                "company": "RIVER BOOK STORE",
                "address": "24 RIVER ROAD CENTRAL CITY 560002",
                "date": "03/04/2025",
                "total": "40.00",
            },
        ),
    ]
    paths = []
    for name, lines, reference in definitions:
        canvas = Image.new("RGB", (620, 100 + len(lines) * 44), "#fffdf7")
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((8, 8, canvas.width - 8, canvas.height - 8), outline="#d5d2cb", width=1)
        for index, line in enumerate(lines):
            draw.text((38, 42 + index * 44), line, fill="#172d29", font=font)
        path = folder / f"receipt_{name}.png"
        canvas.save(path)
        write_json(
            folder / f"receipt_{name}.json",
            {
                "synthetic": True,
                "reference": reference,
                "purpose": "Constructed interaction/OCR demo, never held-out benchmark evidence",
            },
        )
        paths.append(path)
    return paths
