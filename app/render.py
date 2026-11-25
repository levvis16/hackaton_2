"""Отрисовка результатов детекции поверх кадра (bbox, класс, confidence, HUD)."""
from __future__ import annotations

from typing import List, Sequence

import cv2
import numpy as np
from .inference.taxonomy import color_for, label_for


def draw_detections(frame, detections, hud_lines: Sequence[str] | None = None):
    """Возвращает копию кадра с нанесёнными рамками и панелью метрик."""
    canvas = frame.copy()
    for det in detections:
        color = color_for(det.class_name)
        x1, y1 = det.x, det.y
        x2, y2 = det.x + det.width, det.y + det.height
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"{label_for(det.class_name)} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(0, y1 - th - 6)
        cv2.rectangle(canvas, (x1, ty), (x1 + tw + 6, ty + th + 6), color, -1)
        cv2.putText(
            canvas, text, (x1 + 3, ty + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )
    if hud_lines:
        _draw_hud(canvas, hud_lines)
    return canvas


def _draw_hud(canvas, lines: Sequence[str]) -> None:
    overlay = canvas.copy()
    h, w = canvas.shape[:2]
    panel_h = 12 + 22 * len(lines)
    cv2.rectangle(overlay, (0, 0), (min(w, 430), panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    y = 24
    for line in lines:
        cv2.putText(
            canvas, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1, cv2.LINE_AA,
        )
        y += 22


def encode_jpeg(frame, quality: int = 80) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None
