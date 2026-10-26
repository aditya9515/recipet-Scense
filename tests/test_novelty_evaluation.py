import csv
import json

import pytest
from PIL import Image

from receiptsense import confirmation, novelty_evaluation
from receiptsense.common import FIELDS, read_json, read_jsonl, write_json
from receiptsense.novelty_evaluation import address_token_f1, merchant_bootstrap, run_ablation
from receiptsense.recovery_config import load_config


def test_address_token_f1_counts_multiline_overlap_and_missing():
    assert address_token_f1("12 MAIN ROAD\nUDUPI", "12 Main Road UDUPI") == 1
    assert address_token_f1(None, "12 MAIN ROAD") == 0
    assert address_token_f1("12 MAIN", "12 MAIN ROAD") == 0.8


def test_merchant_bootstrap_clusters_receipts_and_is_reproducible():
    entries = [{"receipt_id": str(i), "merchant_id": f"m{i // 2}"} for i in range(6)]
    first = {"rows": [{"receipt_id": str(i), "correct": {"address": False}} for i in range(6)]}
    second = {"rows": [{"receipt_id": str(i), "correct": {"address": True}} for i in range(6)]}
    a = merchant_bootstrap(entries, first, second, 100)
    assert a["difference"] == 1 and a["ci95"] == [1, 1] and a["n_merchants"] == 3
    assert a == merchant_bootstrap(entries, first, second, 100)
    assert merchant_bootstrap([{"receipt_id": "x"}], first, second) is None


def test_previously_seen_test_set_is_not_confirmation(tmp_path):
    manifest = tmp_path / "test.jsonl"
    manifest.write_text(json.dumps({"split": "test"}) + "\n")
    with pytest.raises(ValueError, match="validation"):
        run_ablation(manifest, name="must-not-run")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def confirmation_fixture(tmp_path, monkeypatch):
    # Explicitly synthetic test data, never a real confirmation experiment.
    monkeypatch.setattr(confirmation, "ROOT", tmp_path)
    from types import SimpleNamespace

    monkeypatch.setattr(confirmation, "get_engine", lambda: SimpleNamespace(fingerprint="ocr-fixture"))
    monkeypatch.setattr(novelty_evaluation, "policy_fingerprint", lambda: "policy-fixture")
    monkeypatch.setattr(
        novelty_evaluation, "checkpoint_identity", lambda *a, **kw: {"weight_sha256": "fixture"}
    )
    folder = tmp_path / "private"
    confirmation.initialize_confirmation(folder)
    config = load_config()
    write_json(
        tmp_path / "artifacts/novelty/validation/summary.json",
        {
            "complete": True,
            "purpose": "validation development",
            "configuration_hash": config.fingerprint,
            "policy_fingerprint": "policy-fixture",
            "base_ocr_fingerprint": "ocr-fixture",
        },
    )
    collection, left, right = [], [], []
    for i in range(50):
        filename = f"synthetic-{i}.png"
        Image.new("RGB", (24, 24), (i, 20, 30)).save(folder / "images" / filename)
        collection.append(
            {
                "filename": filename,
                "merchant_id": f"M{i // 5:02d}",
                "selected_at": "2026-10-26T10:00:00+05:30",
            }
        )
        values = dict(zip(FIELDS, ["SYNTHETIC STORE", "12 ROAD 576104", "2026-10-26", "10.00"], strict=True))
        left.append({"filename": filename, "reviewer": "reviewer-a", **values})
        right.append({"filename": filename, "reviewer": "reviewer-b", **values})
    write_csv(folder / "collection.csv", collection)
    write_csv(folder / "labels_a.csv", left)
    write_csv(folder / "labels_b.csv", right)
    return folder, right


def test_confirmation_requires_real_complete_collection(tmp_path):
    confirmation.initialize_confirmation(tmp_path)
    with pytest.raises(ValueError, match="50"):
        confirmation.freeze_confirmation(tmp_path)


def test_confirmation_freeze_verifies_labels_images_and_policy(confirmation_fixture, monkeypatch):
    folder, _ = confirmation_fixture
    frozen = confirmation.freeze_confirmation(folder)
    assert len(read_jsonl(frozen["manifest"])) == 50
    assert confirmation.verify_confirmation(frozen["manifest"], frozen["lock"], checkpoint="fixture")
    monkeypatch.setattr(novelty_evaluation, "policy_fingerprint", lambda: "changed-policy")
    with pytest.raises(ValueError, match="policy changed"):
        confirmation.verify_confirmation(frozen["manifest"], frozen["lock"], checkpoint="fixture")
    monkeypatch.setattr(novelty_evaluation, "policy_fingerprint", lambda: "policy-fixture")
    from types import SimpleNamespace

    with monkeypatch.context() as local:
        local.setattr(confirmation, "get_engine", lambda: SimpleNamespace(fingerprint="changed-ocr"))
        with pytest.raises(ValueError, match="OCR engine changed"):
            confirmation.verify_confirmation(frozen["manifest"], frozen["lock"], checkpoint="fixture")
    with (folder / "labels_a.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="annotation source changed"):
        confirmation.verify_confirmation(frozen["manifest"], frozen["lock"], checkpoint="fixture")
    assert read_json(frozen["lock"])["n_merchants"] == 10


def test_disagreement_needs_third_reviewer_adjudication(confirmation_fixture):
    folder, right = confirmation_fixture
    right[0]["total"] = "11.00"
    write_csv(folder / "labels_b.csv", right)
    with pytest.raises(ValueError, match="third-reviewer"):
        confirmation.freeze_confirmation(folder)
    resolution = {
        **right[0],
        "reviewer": "reviewer-c",
        "total": "10.00",
        "reason": "Synthetic fixture adjudication",
    }
    write_csv(folder / "adjudications.csv", [resolution])
    result = confirmation.freeze_confirmation(folder)
    assert read_jsonl(result["manifest"])[0]["adjudicated"]
