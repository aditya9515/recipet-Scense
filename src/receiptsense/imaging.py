"""Conservative receipt preparation with reversible display coordinates."""

from __future__ import annotations

import io
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from .common import ROOT, digest
from .contracts import ImageRecord, OCRRecord

Image.MAX_IMAGE_PIXELS = 40_000_000
COLORS = {"company": "#16a085", "address": "#367bdc", "date": "#9467d5", "total": "#e29324"}


class ImageInputError(ValueError):
    """An actionable image validation failure."""


def prepare_image(
    data: bytes,
    filename: str = "receipt.png",
    *,
    variant: str = "identity",
    deskew_angle: float | None = None,
    corners: list | None = None,
    artifact_root: str | Path = ROOT / "artifacts",
) -> ImageRecord:
    if not data or len(data) > 10 * 1024 * 1024:
        raise ImageInputError("Choose a nonempty PNG/JPEG image smaller than 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            opened = Image.open(io.BytesIO(data))
            source_format = opened.format
            preserve_container = opened.mode == "RGB"
            if opened.format not in {"PNG", "JPEG"}:
                raise ImageInputError("This version accepts PNG and JPEG receipts.")
            raw_size = opened.size
            if raw_size[0] * raw_size[1] > 40_000_000:
                raise ImageInputError("Choose an image smaller than 40 megapixels.")
            orientation = int(opened.getexif().get(274, 1))
            original = ImageOps.exif_transpose(opened)
            if original.mode in {"RGBA", "LA"} or "transparency" in original.info:
                rgba = original.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                original = Image.alpha_composite(background, rgba)
            original = original.convert("RGB")
            original.load()
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ImageInputError("The image could not be decoded safely. Try a PNG/JPEG photo.") from exc
    if min(original.size) < 24:
        raise ImageInputError("The receipt image is too small to read.")
    if variant not in {"identity", "deskew", "contrast", "perspective"}:
        raise ValueError(f"Unknown preprocessing variant: {variant}")
    source_hash = digest(data)
    receipt_id = source_hash[:20]
    rgb = np.asarray(original)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    matrix = np.eye(3, dtype=float)
    operations = ["exif_to_display", "RGB"]
    if variant == "deskew":
        if deskew_angle is None:
            scale = min(1.0, 1280 / max(gray.shape))
            angle_image = (
                cv2.resize(gray, (max(1, round(gray.shape[1] * scale)), max(1, round(gray.shape[0] * scale))))
                if scale < 1
                else gray
            )
            lines = cv2.HoughLinesP(
                cv2.Canny(angle_image, 50, 150),
                1,
                np.pi / 180,
                threshold=60,
                minLineLength=max(30, angle_image.shape[1] // 5),
                maxLineGap=10,
            )
            angles = (
                []
                if lines is None
                else [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for [[x1, y1, x2, y2]] in lines]
            )
            horizontal = [a for a in angles if abs(a) <= 15]
            deskew_angle = float(np.median(horizontal)) if horizontal else 0.0
        if abs(deskew_angle) > 15:
            raise ImageInputError("Deskew is limited to 15 degrees; rotate the image first.")
        w, h = original.size
        affine = cv2.getRotationMatrix2D((w / 2, h / 2), deskew_angle, 1.0)
        cosine, sine = abs(affine[0, 0]), abs(affine[0, 1])
        new_w, new_h = int(np.ceil(h * sine + w * cosine)), int(np.ceil(h * cosine + w * sine))
        if new_w * new_h > 40_000_000:
            raise ImageInputError("Deskew would exceed the image-size limit. Select Original preparation.")
        affine[0, 2] += (new_w - w) / 2
        affine[1, 2] += (new_h - h) / 2
        rgb = cv2.warpAffine(rgb, affine, (new_w, new_h), borderValue=(255, 255, 255))
        matrix[:2] = affine
        operations.append(f"deskew:{deskew_angle:.4f}")
    elif variant == "contrast":
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        operations.append("CLAHE:1.5:8x8")
    elif variant == "perspective":
        if corners is None or np.asarray(corners).shape != (4, 2):
            raise ImageInputError("Perspective correction needs four corners in TL,TR,BR,BL order.")
        points = np.asarray(corners, dtype=np.float32)
        if not cv2.isContourConvex(points) or abs(cv2.contourArea(points)) < 100:
            raise ImageInputError("Receipt corners must form a nonempty convex quadrilateral.")
        if (
            np.any(points < 0)
            or np.any(points[:, 0] > original.width)
            or np.any(points[:, 1] > original.height)
        ):
            raise ImageInputError("Receipt corners must lie inside the displayed image.")
        w = int(max(np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[3])))
        h = int(max(np.linalg.norm(points[3] - points[0]), np.linalg.norm(points[2] - points[1])))
        if min(w, h) < 24 or w * h > 40_000_000:
            raise ImageInputError("The selected receipt corners produce an unsupported image size.")
        matrix = cv2.getPerspectiveTransform(
            points, np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
        )
        rgb = cv2.warpPerspective(rgb, matrix, (w, h), borderValue=(255, 255, 255))
        operations.append("perspective:manual_corners")
    processed = Image.fromarray(rgb)
    config = {"variant": variant, "operations": operations, "matrix": matrix.tolist(), "version": "image-v2"}
    fingerprint = digest(config)
    folder = Path(artifact_root) / "images" / receipt_id
    folder.mkdir(parents=True, exist_ok=True)
    source_path = folder / ("source.jpg" if source_format == "JPEG" else "source.png")
    if not source_path.exists():
        source_path.write_bytes(data)
    if preserve_container and orientation == 1:
        display_path = folder / ("display.jpg" if source_format == "JPEG" else "display.png")
        if not display_path.exists():
            display_path.write_bytes(data)
    else:
        display_path = folder / "display.png"
        original.save(display_path, compress_level=1)
    if variant == "identity":
        processed_path = display_path
    else:
        processed_path = folder / f"{fingerprint[:16]}.png"
        processed.save(processed_path, compress_level=1)
    diagnostic = gray
    if max(gray.shape) > 1280:
        scale = 1280 / max(gray.shape)
        diagnostic = cv2.resize(
            gray, (max(1, round(gray.shape[1] * scale)), max(1, round(gray.shape[0] * scale)))
        )
    return ImageRecord(
        receipt_id=receipt_id,
        source_sha256=source_hash,
        filename=Path(filename).name,
        source_path=str(source_path.resolve()),
        raw_size=raw_size,
        display_size=original.size,
        processed_size=processed.size,
        display_path=str(display_path.resolve()),
        processed_path=str(processed_path.resolve()),
        processed_sha256=digest(processed_path.read_bytes()),
        preprocessing_fingerprint=fingerprint,
        display_to_processed=matrix.tolist(),
        exif_orientation=orientation,
        operations=operations,
        quality={
            "blur_variance": float(cv2.Laplacian(diagnostic, cv2.CV_64F).var()),
            "contrast_std": float(diagnostic.std()),
            "diagnostic_max_side": 1280.0,
        },
    )


def transform_points(points: list, matrix: list | np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    homogeneous = np.c_[points, np.ones(len(points))] @ np.asarray(matrix, dtype=float).T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-9):
        raise ValueError("Transform maps an evidence point to infinity")
    return homogeneous[:, :2] / homogeneous[:, 2:]


def evidence_image(
    image: ImageRecord, ocr: OCRRecord, ids: list[str], *, frame: str = "display", color: str = "#16a085"
) -> Image.Image:
    if frame not in {"display", "processed"}:
        raise ValueError("Evidence frame must be display or processed")
    canvas = Image.open(image.display_path if frame == "display" else image.processed_path).convert("RGB")
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    inverse = np.linalg.inv(np.asarray(image.display_to_processed))
    for unit in ocr.units:
        if unit.unit_id not in ids:
            continue
        points = np.asarray(unit.polygon)
        if frame == "display":
            points = transform_points(points, inverse)
        xy = [tuple(point) for point in points.tolist()]
        from PIL import ImageColor

        draw.polygon(
            xy, fill=(*ImageColor.getrgb(color), 38), outline=(*ImageColor.getrgb(color), 255), width=3
        )
    return Image.alpha_composite(canvas.convert("RGBA"), layer).convert("RGB")


def retry_region_image(image: ImageRecord, boxes: list, *, frame="display") -> Image.Image:
    if frame not in {"display", "processed"}:
        raise ValueError("Unknown image frame")
    canvas = Image.open(image.display_path if frame == "display" else image.processed_path).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for index, (left, top, right, bottom) in enumerate(boxes):
        points = [[left, top], [right, top], [right, bottom], [left, bottom]]
        if frame == "display":
            points = transform_points(points, np.linalg.inv(image.display_to_processed)).tolist()
        points = [tuple(p) for p in points]
        draw.line([*points, points[0]], fill="#cb643a", width=3)
        draw.text(points[0], f" Retry {index + 1}", fill="#8a3518")
    return canvas
