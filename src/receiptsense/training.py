"""Reproducible head adaptation and optional end-to-end LayoutLMv3 fine-tuning."""

from __future__ import annotations

import copy
import platform
import random
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from .common import LABELS, ROOT, digest, read_json, read_jsonl, source_fingerprint, utc_now, write_json
from .contracts import ImageRecord, OCRRecord, PredictionRecord
from .decoding import decode_fields
from .encoding import encode_document
from .evaluation import score_predictions
from .validation import validate_fields


def seed_everything(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_record(path: str | Path):
    record = read_json(path)
    return ImageRecord.model_validate(record["image"]), OCRRecord.model_validate(record["ocr"])


def train_model(
    labels_path: str | Path,
    validation_manifest: str | Path,
    *,
    records: str | Path = ROOT / "data/processed/records/identity",
    base: str | Path = ROOT / "artifacts/models/layoutlmv3-base",
    output: str | Path = ROOT / "artifacts/models/receiptsense-layoutlmv3",
    mode: str = "head",
    epochs: int = 40,
    learning_rate: float | None = None,
    seed: int = 42,
    limit: int | None = None,
    patience: int = 8,
) -> dict:
    import torch
    from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

    if mode not in {"head", "full"}:
        raise ValueError("Training mode must be head or full")
    rows = read_jsonl(labels_path)
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("Optimizer input must contain training-split labels only")
    rows = [r for r in rows if r["usable_for_training"]]
    rows = rows[:limit] if limit else rows
    if not rows:
        raise ValueError(
            "No fully alignable training receipts. Inspect alignment coverage or add reviewed labels."
        )
    validation_entries = read_jsonl(validation_manifest)
    if any(row.get("split") != "validation" for row in validation_entries):
        raise ValueError("Checkpoint selection requires the validation split, never test labels")
    train_ids = {r["receipt_id"] for r in rows}
    if train_ids & {r["receipt_id"] for r in validation_entries}:
        raise ValueError("Training and validation overlap")
    seed_everything(seed)
    started = perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = LayoutLMv3Processor.from_pretrained(base, apply_ocr=False, local_files_only=True)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        base,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
        local_files_only=True,
        ignore_mismatched_sizes=True,
    ).to(device)
    output = Path(output)
    if (output / "receiptsense_model.json").exists():
        raise ValueError("Choose a new output directory; trained bundles are not overwritten implicitly")
    output.mkdir(parents=True, exist_ok=True)
    log = {
        "created_at": utc_now(),
        "trained": False,
        "labels": LABELS,
        "mode": mode,
        "source_fingerprint": source_fingerprint(),
        "base_checkpoint": str(Path(base).resolve()),
        "base_revision": "cfbbbff0762e6aab37086fdd4739ad14fe7d5db4",
        "seed": seed,
        "training_receipts": len(rows),
        "validation_receipts": len(validation_entries),
        "training_ids_hash": digest(sorted(train_ids)),
        "validation_ids_hash": digest([r["receipt_id"] for r in validation_entries]),
        "annotation_policy": "exact-span-partial-supervision-v2",
        "label_provenance": "automatically aligned weak labels; see annotation audit",
        "device": str(device),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "max_epochs": epochs,
        "patience": patience,
        "selection_metric": "validation strict-v1 macro field EM",
        "subwords": "first_subword_only",
        "window_owner": "center_preference-v1",
        "history": [],
    }
    learning_rate = learning_rate if learning_rate is not None else (1e-3 if mode == "head" else 2e-5)
    log["learning_rate"] = learning_rate
    label_counter = Counter(label for row in rows for label in row["labels"] if label != "IGNORE")
    weights = torch.tensor([1 / max(1, label_counter[label]) ** 0.5 for label in LABELS], dtype=torch.float32)
    weights = weights / weights.mean()
    weights = weights.clamp(0.1, 5).to(device)
    log["class_counts"] = dict(label_counter)
    log["class_weights"] = weights.cpu().tolist()
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    base_fingerprint = digest({"base": log["base_revision"], "encoding": "word-windows-dynamic-v2"})

    def features(image: ImageRecord, ocr: OCRRecord):
        key = digest(
            {"base": base_fingerprint, "image": image.processed_sha256, "ocr": ocr.model_dump(mode="json")}
        )
        cache = ROOT / "artifacts/head_features" / f"{key}.pt"
        if cache.exists():
            return torch.load(cache, map_location="cpu", weights_only=True)
        model.eval()
        result = torch.zeros((len(ocr.units), model.config.hidden_size), dtype=torch.float16)
        with torch.inference_mode():
            for window in encode_document(processor, image, ocr):
                hidden = model.layoutlmv3(
                    **{k: v.to(device) for k, v in window.inputs.items()}
                ).last_hidden_state
                for unit_index, token_index in window.positions.items():
                    result[unit_index] = hidden[0, token_index].cpu().half()
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, cache)
        return result

    validation_records = []
    for entry in validation_entries:
        path = Path(records) / f"{entry['receipt_id']}.json"
        if path.exists():
            image, ocr = load_record(path)
            validation_records.append((image, ocr, features(image, ocr) if mode == "head" else None))
            print(f"Validation features {len(validation_records)}/{len(validation_entries)}", flush=True)
    if not validation_records:
        raise ValueError("No validation OCR records available")
    fingerprints = set()
    if mode == "head":
        for parameter in model.layoutlmv3.parameters():
            parameter.requires_grad_(False)
        xs, ys = [], []
        for index, row in enumerate(rows):
            image, ocr = load_record(row["record_path"])
            if row["unit_ids"] != [u.unit_id for u in ocr.units]:
                raise ValueError("Alignment labels do not match this OCR record")
            fingerprints.add(ocr.ocr_fingerprint)
            xs.append(features(image, ocr).float())
            ys.append(
                torch.tensor([LABELS.index(label) if label != "IGNORE" else -100 for label in row["labels"]])
            )
            print(f"Training features {index + 1}/{len(rows)}", flush=True)
        all_x, all_y = torch.cat(xs), torch.cat(ys)
        log["training_units"] = len(all_y)
        # A diagnostic fit on training-only features catches broken label/head
        # plumbing. Reset the head afterwards; this is never a generalization score.
        initial_head = copy.deepcopy(model.classifier.state_dict())
        tiny_x, tiny_y = torch.cat(xs[:8]).to(device), torch.cat(ys[:8]).to(device)
        known = tiny_y != -100
        tiny_x, tiny_y = tiny_x[known], tiny_y[known]
        tiny_optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.005, weight_decay=0)
        initial_loss = float(loss_fn(model.classifier(tiny_x), tiny_y).detach())
        tiny_accuracy = 0.0
        for tiny_step in range(150):
            tiny_optimizer.zero_grad(set_to_none=True)
            tiny_logits = model.classifier(tiny_x)
            tiny_loss = loss_fn(tiny_logits, tiny_y)
            tiny_loss.backward()
            tiny_optimizer.step()
            tiny_accuracy = float((model.classifier(tiny_x).argmax(-1) == tiny_y).float().mean())
            if tiny_step >= 20 and tiny_accuracy >= 0.98:
                break
        diagnostic = {
            "purpose": "training-only tiny classifier-head overfit check; not held-out performance",
            "receipts": min(8, len(rows)),
            "supervised_units": int(known.sum()),
            "initial_loss": initial_loss,
            "final_loss": float(tiny_loss.detach()),
            "supervised_training_accuracy": tiny_accuracy,
            "steps": tiny_step + 1,
            "passed": tiny_accuracy >= 0.95,
            "head_reset_before_main_training": True,
        }
        write_json(output / "tiny_overfit_check.json", diagnostic)
        model.classifier.load_state_dict(initial_head)
        if not diagnostic["passed"]:
            raise RuntimeError(
                "Tiny classifier-head overfit check failed; inspect labels and features before training"
            )
        log["tiny_overfit_check"] = diagnostic
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=learning_rate, weight_decay=0.01)
    else:
        model.gradient_checkpointing_enable()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    best, best_state, stale = -1.0, None, 0

    def validate():
        model.eval()
        predictions = []
        with torch.inference_mode():
            for image, ocr, cached in validation_records:
                unit_labels, unit_scores = ["O"] * len(ocr.units), [0.0] * len(ocr.units)
                if mode == "head":
                    probabilities = model.classifier(cached.float().to(device)).softmax(-1).cpu()
                    for i, probability in enumerate(probabilities):
                        score, label = probability.max(-1)
                        unit_labels[i], unit_scores[i] = LABELS[int(label)], float(score)
                else:
                    for window in encode_document(processor, image, ocr):
                        probabilities = (
                            model(**{k: v.to(device) for k, v in window.inputs.items()})
                            .logits[0]
                            .softmax(-1)
                            .cpu()
                        )
                        for i, token_index in window.positions.items():
                            score, label = probabilities[token_index].max(-1)
                            unit_labels[i], unit_scores[i] = LABELS[int(label)], float(score)
                prediction = PredictionRecord(
                    receipt_id=ocr.receipt_id,
                    method="layoutlmv3",
                    fields=decode_fields(ocr, unit_labels, unit_scores),
                    ocr_fingerprint=ocr.ocr_fingerprint,
                    model_revision="training-validation",
                )
                predictions.append(validate_fields(prediction))
        return score_predictions(validation_entries, predictions), predictions

    for epoch in range(epochs):
        model.train()
        losses = []
        if mode == "head":
            order = torch.randperm(len(all_y))
            for indices in order.split(512):
                targets = all_y[indices].to(device)
                if not (targets != -100).any():
                    continue
                optimizer.zero_grad(set_to_none=True)
                logits = model.classifier(model.dropout(all_x[indices].to(device)))
                loss = loss_fn(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
        else:
            shuffled = rows.copy()
            random.shuffle(shuffled)
            optimizer.zero_grad(set_to_none=True)
            steps = 0
            accumulation = 8
            for row in shuffled:
                image, ocr = load_record(row["record_path"])
                fingerprints.add(ocr.ocr_fingerprint)
                for window in encode_document(processor, image, ocr, row["labels"]):
                    if not (window.labels != -100).any():
                        continue
                    logits = model(**{k: v.to(device) for k, v in window.inputs.items()}).logits
                    loss = loss_fn(logits.reshape(-1, len(LABELS)), window.labels.to(device).reshape(-1))
                    (loss / accumulation).backward()
                    losses.append(float(loss.detach()))
                    steps += 1
                    if steps % accumulation == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
            if steps % accumulation:
                # Correct the final partial accumulation's effective learning rate.
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(accumulation / (steps % accumulation))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        metrics, predictions = validate()
        score = metrics["macro_field_em"]
        history = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)) if losses else None,
            "validation_macro_field_em": score,
            "validation_all_four_em": metrics["all_four_em"],
        }
        log["history"].append(history)
        print(history, flush=True)
        if score > best:
            best, stale = score, 0
            best_state = copy.deepcopy(
                model.classifier.state_dict()
                if mode == "head"
                else {k: v.cpu().clone() for k, v in model.state_dict().items()}
            )
            log["selected_epoch"] = epoch + 1
            write_json(output / "validation_metrics.json", metrics)
            from .common import write_jsonl

            write_jsonl(
                output / "validation_predictions.jsonl", [p.model_dump(mode="json") for p in predictions]
            )
        else:
            stale += 1
        write_json(output / "training_log.json", log)
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("No valid checkpoint was selected")
    if mode == "head":
        model.classifier.load_state_dict(best_state)
    else:
        model.load_state_dict(best_state)
    model.save_pretrained(output, safe_serialization=True)
    processor.save_pretrained(output)
    log.update(
        trained=True,
        best_validation_macro_field_em=best,
        elapsed_seconds=perf_counter() - started,
        ocr_fingerprints=sorted(fingerprints),
        adaptation="frozen-backbone classifier-head adaptation"
        if mode == "head"
        else "all-parameter fine-tuning",
        weight_sha256=digest((output / "model.safetensors").read_bytes()),
    )
    write_json(output / "receiptsense_model.json", log)
    write_json(output / "training_log.json", log)
    return log
