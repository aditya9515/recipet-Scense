"""Import the nine public papers cited by ReceiptSense; validate every PDF.

Run from any directory with Python and pypdf installed. Existing valid files
are reused. Network failures are recorded, never disguised as successful PDFs.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PAPERS = [
    (
        "P01",
        "LayoutLMv3",
        "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking",
        "2204.08387v3",
        "2022",
        "Main extraction model",
    ),
    (
        "P02",
        "SROIE",
        "ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction",
        "2103.10213v1",
        "2019; preprint 2021",
        "Benchmark and field definitions",
    ),
    (
        "P03",
        "DBNet",
        "Real-time Scene Text Detection with Differentiable Binarization",
        "1911.08947v2",
        "2020; preprint 2019",
        "Detection background",
    ),
    (
        "P04",
        "CRNN",
        "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition",
        "1507.05717v1",
        "2017; preprint 2015",
        "Recognition background",
    ),
    (
        "P05",
        "LayoutLM",
        "LayoutLM: Pre-training of Text and Layout for Document Image Understanding",
        "1912.13318v5",
        "2020",
        "Text and layout background",
    ),
    (
        "P06",
        "BROS",
        "BROS: A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents",
        "2108.04539v5",
        "2022; preprint 2021",
        "Spatial reasoning; optional comparison",
    ),
    (
        "P07",
        "TrOCR",
        "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models",
        "2109.10282v5",
        "2023; preprint 2021",
        "Optional recognition comparison",
    ),
    (
        "P08",
        "Donut",
        "OCR-free Document Understanding Transformer",
        "2111.15664v5",
        "2022",
        "Optional OCR-free comparison",
    ),
    (
        "P09",
        "Joint_Receipt_Model",
        "Receipt Information Extraction with Joint Multi-Modal Transformer and Rule-Based Model",
        None,
        "2025",
        "Related hybrid receipt extraction",
    ),
]


def import_one(item):
    pid, short, title, version, year, role = item
    folder = ROOT / "research" / "papers"
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"{pid}_{short}.pdf"
    target = folder / filename
    if version:
        canonical = f"https://arxiv.org/abs/{version}"
        urls = [f"https://arxiv.org/pdf/{version}"]
    else:
        canonical = "https://doi.org/10.3390/make7040167"
        urls = [
            "https://mdpi-res.com/d_attachment/make/make-07-00167/article_deploy/make-07-00167.pdf",
            "https://www.mdpi.com/2504-4990/7/4/167/pdf",
        ]
    record = dict(
        id=pid,
        short_name=short,
        title=title,
        publication_year=year,
        role=role,
        canonical_url=canonical,
        version=version or "Publisher version of record",
        local_file=f"papers/{filename}",
    )
    errors = []
    for url in urls:
        try:
            cached = target.exists()
            if cached:
                data = target.read_bytes()
            else:
                req = Request(url, headers={"User-Agent": "Mozilla/5.0 ReceiptSense-Academic-Reading"})
                with urlopen(req, timeout=45) as response:
                    data = response.read()
            if not data.startswith(b"%PDF-"):
                raise ValueError("Response is not a PDF")
            reader = PdfReader(BytesIO(data))
            page_count = len(reader.pages)
            if not page_count:
                raise ValueError("PDF has no pages")
            first_page = reader.pages[0].extract_text() or ""
            if not first_page.strip():
                raise ValueError("PDF has no extractable title page")
            # Detect an unrelated PDF even if a remote server returned PDF bytes.
            compact = "".join(first_page.casefold().split())
            title_start = "".join(title.casefold().split())[:32]
            if title_start not in compact:
                raise ValueError("Expected paper title not found on first page")
            if not cached:
                target.write_bytes(data)
            record.update(
                status="downloaded_and_validated",
                download_url=url,
                bytes=len(data),
                pages=page_count,
                sha256=sha256(data).hexdigest(),
                checked_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            print(f"{pid}: OK, {page_count} pages, {len(data)} bytes", flush=True)
            return record
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    record.update(status="failed", errors=errors)
    print(f"{pid}: FAILED: {'; '.join(errors)}", flush=True)
    return record


def main():
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(import_one, PAPERS))
    path = ROOT / "research" / "paper_manifest.json"
    path.write_text(
        json.dumps({"purpose": "ReceiptSense literature import", "papers": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    if any(r["status"] != "downloaded_and_validated" for r in records):
        raise SystemExit("One or more downloads failed; inspect research/paper_manifest.json")


if __name__ == "__main__":
    main()
