import pytest

from receiptsense.common import ROOT
from receiptsense.contracts import OCRRecord, OCRUnit
from receiptsense.encoding import encode_document
from receiptsense.imaging import prepare_image


@pytest.mark.skipif(
    not (ROOT / "artifacts/models/layoutlmv3-base/vocab.json").exists(),
    reason="Optional local pretrained tokenizer assets are not installed",
)
def test_real_tokenizer_preserves_bottom_total_and_supervises_each_unit_once(receipt_bytes, tmp_path):
    from transformers import LayoutLMv3Processor

    image = prepare_image(receipt_bytes, artifact_root=tmp_path)
    units = [
        OCRUnit(
            unit_id=f"u{i}",
            text="item" if i < 799 else "493.50",
            polygon=[(10, 10), (60, 10), (60, 35), (10, 35)],
            bbox_xyxy=(10, 10, 60, 35),
            line_id=f"l{i}",
            reading_order=i,
        )
        for i in range(800)
    ]
    ocr = OCRRecord(
        receipt_id=image.receipt_id,
        image_width=620,
        image_height=900,
        ocr_fingerprint="long-fixture",
        units=units,
    )
    processor = LayoutLMv3Processor.from_pretrained(
        ROOT / "artifacts/models/layoutlmv3-base", apply_ocr=False, local_files_only=True
    )
    windows = encode_document(processor, image, ocr, ["O"] * 799 + ["B-TOTAL"])
    assert len(windows) > 1
    assert sum(int((w.labels != -100).sum()) for w in windows) == 800
    owner = next(w for w in windows if 799 in w.positions)
    assert owner.labels[0, owner.positions[799]] == 7
    assert all(w.inputs["input_ids"].shape[1] <= 512 for w in windows)
