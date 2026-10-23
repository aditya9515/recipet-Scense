"""Scale-aware line grouping, preserving native line regions when available."""

from __future__ import annotations

from statistics import median

from .contracts import OCRUnit


def order_units(units: list[OCRUnit]) -> list[OCRUnit]:
    if not units:
        return []
    heights = [u.bbox_xyxy[3] - u.bbox_xyxy[1] for u in units]
    tolerance = max(3.0, median(heights) * 0.55)
    lines: list[list[OCRUnit]] = []
    for unit in sorted(
        units, key=lambda u: ((u.bbox_xyxy[1] + u.bbox_xyxy[3]) / 2, u.bbox_xyxy[0], u.unit_id)
    ):
        cy = (unit.bbox_xyxy[1] + unit.bbox_xyxy[3]) / 2
        candidates = [
            (abs(cy - median((v.bbox_xyxy[1] + v.bbox_xyxy[3]) / 2 for v in line)), i)
            for i, line in enumerate(lines)
        ]
        distance, index = min(candidates, default=(float("inf"), -1))
        if distance <= tolerance:
            lines[index].append(unit)
        else:
            lines.append([unit])
    ordered = []
    for index, line in enumerate(lines):
        for unit in sorted(line, key=lambda u: (u.bbox_xyxy[0], u.reading_order, u.unit_id)):
            ordered.append(
                unit.model_copy(update={"line_id": f"l{index:04d}", "reading_order": len(ordered)})
            )
    return ordered


def grouped_lines(units: list[OCRUnit]) -> list[list[OCRUnit]]:
    groups: dict[str, list[OCRUnit]] = {}
    for unit in sorted(units, key=lambda u: u.reading_order):
        groups.setdefault(unit.line_id, []).append(unit)
    return list(groups.values())
