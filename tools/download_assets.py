"""Download pinned public dataset/model assets without executing remote code."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DATASET = "jsdnrs/ICDAR2019-SROIE"
DATA_REVISION = "bffe40c26759f3376ec2b3ae9031dbba54cd587c"
MODEL_REVISION = "cfbbbff0762e6aab37086fdd4739ad14fe7d5db4"


def download(url: str, target: Path, expected_hash: str | None = None) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)

    def checksum(path):
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    if target.exists() and (expected_hash is None or checksum(target) == expected_hash):
        return {
            "path": str(target.relative_to(ROOT)),
            "sha256": checksum(target),
            "bytes": target.stat().st_size,
            "url": url,
        }
    temporary = target.with_suffix(target.suffix + ".part")
    if expected_hash:
        if target.exists():
            target.replace(temporary)
        request_url = url + "?download=true"
        with urlopen(Request(request_url, headers={"Range": "bytes=0-0"}), timeout=30) as response:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith("bytes 0-0/"):
                raise ValueError("Server did not support a verifiable ranged download")
            total = int(content_range.split("/")[1])
        start = temporary.stat().st_size if temporary.exists() else 0
        if start >= total:
            start = 0
        ranges = [(i, min(i + 4 * 1024 * 1024 - 1, total - 1)) for i in range(start, total, 4 * 1024 * 1024)]

        def fetch(bounds):
            first, last = bounds
            for attempt in range(4):
                try:
                    with urlopen(
                        Request(request_url, headers={"Range": f"bytes={first}-{last}"}), timeout=40
                    ) as response:
                        if response.headers.get("Content-Range") != f"bytes {first}-{last}/{total}":
                            raise ValueError("Remote range mismatch")
                        data = response.read()
                    if len(data) != last - first + 1:
                        raise ValueError("Remote range was truncated")
                    return data
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(1)

        with temporary.open("ab" if start else "wb") as stream, ThreadPoolExecutor(max_workers=4) as parts:
            for data in parts.map(fetch, ranges):
                stream.write(data)
        if temporary.stat().st_size != total:
            raise ValueError(f"Incomplete file: {target.name}")
    else:
        with (
            urlopen(
                Request(url, headers={"User-Agent": "ReceiptSense academic prototype"}), timeout=40
            ) as response,
            temporary.open("wb") as stream,
        ):
            data = response.read()
            if response.headers.get("Content-Length") and len(data) != int(
                response.headers["Content-Length"]
            ):
                raise ValueError("Truncated response")
            stream.write(data)
    actual = checksum(temporary)
    if expected_hash and actual != expected_hash:
        raise ValueError(f"Checksum mismatch for {target.name}")
    temporary.replace(target)
    print(f"Downloaded {target.name}: {target.stat().st_size / 1e6:.1f} MB", flush=True)
    return {
        "path": str(target.relative_to(ROOT)),
        "sha256": actual,
        "bytes": target.stat().st_size,
        "url": url,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["dataset", "model", "all"], default="all")
    args = parser.parse_args()
    jobs = []
    if args.kind in {"all", "dataset"}:
        hashes = {
            "train": "b18c16b4d8481e5e4537a1700e4616907fe4acd92d6362a7e430b0e866213887",
            "test": "04f8f31b45944cc6e6459a7a95c851a721fc93ffec0a5c29ece9ded734a684c2",
        }
        for split, value in hashes.items():
            name = f"data/{split}-00000-of-00001.parquet"
            jobs.append(
                (
                    f"https://huggingface.co/datasets/{DATASET}/resolve/{DATA_REVISION}/{name}",
                    ROOT / "data/raw" / f"{split}.parquet",
                    value,
                )
            )
        for name in ["README.md", "LICENSE", "NOTICE"]:
            jobs.append(
                (
                    f"https://huggingface.co/datasets/{DATASET}/resolve/{DATA_REVISION}/{name}",
                    ROOT / "data/raw" / name,
                    None,
                )
            )
    if args.kind in {"all", "model"}:
        for name in [
            "config.json",
            "model.safetensors",
            "merges.txt",
            "vocab.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
            "README.md",
        ]:
            expected = (
                "3c6bd09290e28867c32adf634a6b36574ff43a9e53539633f5ac540015ac2baf"
                if name == "model.safetensors"
                else None
            )
            jobs.append(
                (
                    f"https://huggingface.co/microsoft/layoutlmv3-base/resolve/{MODEL_REVISION}/{name}",
                    ROOT / "artifacts/models/layoutlmv3-base" / name,
                    expected,
                )
            )
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(lambda job: download(*job), jobs))
    out = ROOT / "artifacts" / f"download_manifest_{args.kind}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "dataset_revision": DATA_REVISION,
                "model_revision": MODEL_REVISION,
                "files": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
