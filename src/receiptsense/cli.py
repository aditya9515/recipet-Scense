"""Offline entry points for the same code used by the application."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from .common import ROOT


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        prog="receiptsense", description="ReceiptSense local OCR and experiment toolkit"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Inspect dependencies, hardware and artifact readiness")
    sub.add_parser("samples", help="Create the explicitly synthetic example receipts")
    sub.add_parser("prepare", help="Audit downloaded SROIE parquet and freeze manifests")
    ocr = sub.add_parser("ocr", help="Build shared deployment OCR records")
    ocr.add_argument("manifest", type=Path)
    ocr.add_argument("--variant", choices=["identity", "deskew", "contrast"], default="identity")
    ocr.add_argument("--limit", type=int)
    align = sub.add_parser("align", help="Align train/validation field strings to real OCR")
    align.add_argument("manifest", type=Path)
    align.add_argument("--records", type=Path, default=ROOT / "data/processed/records/identity")
    train = sub.add_parser("train", help="Train the receipt head or fine-tune all parameters")
    train.add_argument("--labels", type=Path, default=ROOT / "data/processed/labels_train.jsonl")
    train.add_argument("--validation", type=Path, default=ROOT / "data/manifests/validation.jsonl")
    train.add_argument("--records", type=Path, default=ROOT / "data/processed/records/identity")
    train.add_argument("--output", type=Path, default=ROOT / "artifacts/models/receiptsense-layoutlmv3")
    train.add_argument("--mode", choices=["head", "full"], default="head")
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--patience", type=int, default=8)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--limit", type=int)
    evaluate = sub.add_parser("evaluate", help="Score immutable held-out predictions from shared OCR")
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument(
        "--methods", nargs="+", choices=["rules", "layoutlmv3"], default=["rules", "layoutlmv3"]
    )
    evaluate.add_argument("--records", type=Path, default=ROOT / "data/processed/records/identity")
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--name")
    robust = sub.add_parser("robustness", help="Run fixed paired image-degradation probes")
    robust.add_argument("manifest", type=Path)
    robust.add_argument("--limit", type=int, default=12)
    robust.add_argument("--checkpoint", type=Path)
    robust.add_argument("--name")
    timing = sub.add_parser("benchmark", help="Measure warm uncached pipeline latency")
    timing.add_argument("manifest", type=Path)
    timing.add_argument("--limit", type=int, default=10)
    predict = sub.add_parser("predict", help="Process a supported receipt image")
    predict.add_argument("image", type=Path)
    predict.add_argument("--method", choices=["rules", "layoutlmv3"], default="rules")
    predict.add_argument("--variant", choices=["identity", "deskew", "contrast"], default="identity")
    predict.add_argument("--checkpoint", type=Path)
    predict.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    if args.command == "doctor":
        versions = {}
        for name in [
            "receiptsense",
            "rapidocr-onnxruntime",
            "onnxruntime",
            "paddleocr",
            "paddlepaddle",
            "torch",
            "transformers",
            "streamlit",
        ]:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "not installed"
        try:
            import torch

            cuda = torch.cuda.is_available()
        except ImportError:
            torch, cuda = None, False
        result = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": versions,
            "cuda_available": cuda,
            "device": torch.cuda.get_device_name(0) if cuda else "CPU",
            "base_model_downloaded": (ROOT / "artifacts/models/layoutlmv3-base/model.safetensors").exists(),
            "trained_receipt_model": (
                ROOT / "artifacts/models/receiptsense-layoutlmv3/receiptsense_model.json"
            ).exists(),
        }
    elif args.command == "samples":
        from .samples import create_samples

        result = [str(path) for path in create_samples()]
    elif args.command == "prepare":
        from .dataset import prepare_dataset

        result = prepare_dataset()
    elif args.command == "ocr":
        from .dataset import build_ocr_records

        result = build_ocr_records(args.manifest, variant=args.variant, limit=args.limit)
    elif args.command == "align":
        from .alignment import build_alignments

        result = build_alignments(args.manifest, args.records)
    elif args.command == "train":
        from .training import train_model

        result = train_model(
            args.labels,
            args.validation,
            records=args.records,
            output=args.output,
            mode=args.mode,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            limit=args.limit,
        )
    elif args.command == "evaluate":
        from .evaluation import evaluate_manifest

        result = evaluate_manifest(
            args.manifest,
            methods=tuple(args.methods),
            records=args.records,
            checkpoint=args.checkpoint,
            limit=args.limit,
            run_name=args.name,
        )
    elif args.command == "robustness":
        from .robustness import run_robustness

        result = run_robustness(args.manifest, limit=args.limit, checkpoint=args.checkpoint, name=args.name)
    elif args.command == "benchmark":
        from .benchmark import benchmark

        result = benchmark(args.manifest, limit=args.limit)
    else:
        from .pipeline import process_receipt
        from .review_store import ReviewStore

        output = process_receipt(
            args.image.read_bytes(),
            args.image.name,
            method=args.method,
            variant=args.variant,
            checkpoint=args.checkpoint,
            use_cache=not args.no_cache,
        )
        ReviewStore().save_prediction(output.prediction)
        result = output.prediction.model_dump(mode="json")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
