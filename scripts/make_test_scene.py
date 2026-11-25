"""Генерация тестовой сцены для демонстрации классов smoke_fire и spill.

Создаёт media/test_scene.jpg с синтетическими областями дыма/огня и разлива,
на которых срабатывают эвристические детекторы (без нейросети).

    python scripts/make_test_scene.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.io_utils import imwrite_unicode  # noqa: E402


def main() -> None:
    w, h = 1280, 720
    # Фон: серый пол (асфальт) с текстурой.
    frame = np.full((h, w, 3), 90, dtype=np.uint8)
    noise = np.random.default_rng(7).integers(-12, 12, (h, w, 1))
    frame = np.clip(frame.astype(int) + noise, 0, 255).astype(np.uint8)
    # Несколько светлых полос разметки.
    cv2.rectangle(frame, (0, 560), (w, 600), (170, 170, 170), -1)
    cv2.putText(frame, "TEST SCENE: smoke/fire + spill", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30), 2)

    # --- Огонь: насыщенные тёплые пятна (HSV: hue<=25, S>=140, V>=180) ---
    fire_center = (360, 470)
    for i, (rad, color) in enumerate([(90, (0, 0, 255)), (65, (0, 80, 255)), (38, (0, 200, 255))]):
        cv2.circle(frame, (fire_center[0] + i * 4, fire_center[1] - i * 10), rad, color, -1)
    for _ in range(26):
        rx = fire_center[0] + int(np.random.default_rng(_).integers(-70, 70))
        ry = fire_center[1] - int(np.random.default_rng(_ + 100).integers(60, 150))
        cv2.circle(frame, (rx, ry), int(np.random.default_rng(_ + 9).integers(4, 12)),
                   (0, 120, 255), -1)

    # --- Дым: серые малоконтрастные области (S<=40, 95<=V<=205), верх кадра ---
    smoke = np.zeros((h, w), dtype=np.uint8)
    for cx, cy, rad in [(950, 200, 150), (1080, 250, 130), (860, 260, 110)]:
        cv2.circle(smoke, (cx, cy), rad, 255, -1)
    smoke = cv2.GaussianBlur(smoke, (121, 121), 0)
    smoke_color = np.full((h, w, 3), 175, dtype=np.uint8)
    mask = (smoke.astype(np.float32) / 255.0)[:, :, None]
    frame = (frame * (1 - mask * 0.8) + smoke_color * (mask * 0.8)).astype(np.uint8)

    # --- Разлив: тёмная насыщенная жидкость (S>=70, V<=95) ---
    spill = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(spill, (760, 560), (200, 70), 8, 0, 360, 255, -1)
    cv2.ellipse(spill, (900, 600), (120, 45), -12, 0, 360, 255, -1)
    spill_color = np.array([30, 70, 40], dtype=np.uint8)  # тёмный ГСМ-оттенок
    mask2 = (spill.astype(np.float32) / 255.0)[:, :, None]
    frame = (frame * (1 - mask2) + spill_color * mask2).astype(np.uint8)

    dst = os.path.join(ROOT, "media", "test_scene.jpg")
    imwrite_unicode(dst, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"Создано: {dst}")


if __name__ == "__main__":
    main()
