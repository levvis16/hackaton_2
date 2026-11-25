"""Абстракция источника видеоданных.

Поддерживаются: файл, камера, RTSP-поток, статичное изображение и
синтетический генератор (для запуска без камеры и видеофайла).
Ошибки открытия/чтения выражаются через VideoSourceError, чтобы сервис
не завершался аварийно.
"""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import cv2
import numpy as np

from ..io_utils import imread_unicode


class VideoSourceError(RuntimeError):
    """Ошибка источника видео (нет файла, камера занята и т.п.)."""


class VideoSource:
    def __init__(self, cfg, base_dir: Optional[str] = None) -> None:
        self.cfg = cfg
        self.base_dir = base_dir or os.getcwd()
        self.cap: Optional[cv2.VideoCapture] = None
        self._static_image: Optional[np.ndarray] = None
        self._synthetic_index = 0
        self.source_type = (cfg.source_type or "file").strip().lower()
        self.description = ""

    # ------------------------------------------------------------------
    def _resolve(self, path: str) -> str:
        if not path:
            return path
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.base_dir, path))

    def open(self) -> "VideoSource":
        st = self.source_type
        if st == "file":
            path = self._resolve(self.cfg.path)
            if not path or not os.path.exists(path):
                raise VideoSourceError(f"Видеофайл не найден: {path}")
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise VideoSourceError(f"Не удалось открыть видеофайл: {path}")
            self.description = path
        elif st == "camera":
            index = int(self.cfg.camera_index)
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(index, backend)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(index)
            if not self.cap.isOpened():
                raise VideoSourceError(f"Камера {index} недоступна")
            self.description = f"camera:{index}"
        elif st == "rtsp":
            url = self.cfg.rtsp_url or self.cfg.path
            if not url:
                raise VideoSourceError("Не задан RTSP URL")
            self.cap = cv2.VideoCapture(url)
            if not self.cap.isOpened():
                raise VideoSourceError(f"RTSP-поток недоступен: {url}")
            self.description = url
        elif st == "image":
            path = self._resolve(self.cfg.path)
            img = imread_unicode(path)
            if img is None:
                raise VideoSourceError(f"Изображение не найдено или повреждено: {path}")
            self._static_image = img
            self.description = path
        elif st == "synthetic":
            self.description = "synthetic (генерируемый кадр)"
        else:
            raise VideoSourceError(f"Неизвестный тип источника: {st!r}")
        return self

    # ------------------------------------------------------------------
    def is_static(self) -> bool:
        return self.source_type in ("image", "synthetic")

    def is_file(self) -> bool:
        return self.source_type == "file"

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.source_type == "synthetic":
            return True, self._synthetic_frame()
        if self.source_type == "image":
            return True, self._static_image.copy()
        if self.cap is None:
            return False, None
        ok, frame = self.cap.read()
        if not ok and self.source_type == "file" and self.cfg.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return ok, frame

    def reopen(self) -> "VideoSource":
        self.release()
        return self.open()

    def release(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            finally:
                self.cap = None

    # ------------------------------------------------------------------
    def input_fps(self) -> float:
        if self.cap is not None:
            return float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return 0.0

    def info(self) -> dict:
        meta = {
            "source_type": self.source_type,
            "description": self.description,
            "loop": bool(self.cfg.loop),
        }
        if self.cap is not None:
            meta.update(
                width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                input_fps=round(self.input_fps(), 2),
                frame_count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            )
        elif self._static_image is not None:
            h, w = self._static_image.shape[:2]
            meta.update(width=w, height=h)
        return meta

    # ------------------------------------------------------------------
    def _synthetic_frame(self) -> np.ndarray:
        """Простой генерируемый кадр: движущийся объект на фоне сетки."""
        h, w = 480, 640
        frame = np.full((h, w, 3), 30, dtype=np.uint8)
        for x in range(0, w, 40):
            cv2.line(frame, (x, 0), (x, h), (50, 50, 50), 1)
        for y in range(0, h, 40):
            cv2.line(frame, (0, y), (w, y), (50, 50, 50), 1)

        self._synthetic_index += 1
        t = self._synthetic_index / 30.0
        cx = int(w / 2 + math.sin(t) * (w / 3))
        cy = int(h / 2 + math.cos(t * 0.7) * (h / 4))
        cv2.circle(frame, (cx, cy), 30, (0, 180, 255), -1)
        cv2.putText(
            frame, "SYNTHETIC SOURCE", (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2,
        )
        return frame
