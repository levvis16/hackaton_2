"""Mock-детектор для демонстрации цикла API/события без модели.

Выдаёт по одному синтетическому объекту, перебирая целевые классы,
чтобы можно было проверить формирование событий, UI и API.
"""
from __future__ import annotations

import random
from typing import List

from .base import BaseDetector, Detection
from .taxonomy import TARGET_CLASSES


class MockDetector(BaseDetector):
    name = "mock"

    def __init__(self, every_n_frames: int = 20, logger=None) -> None:
        self.every_n_frames = max(1, int(every_n_frames))
        self.log = logger
        self._frame = 0
        self._classes = list(TARGET_CLASSES.keys())
        self._seed = random.Random(42)

    def detect(self, frame) -> List[Detection]:
        self._frame += 1
        if self._frame % self.every_n_frames != 0:
            return []
        h, w = frame.shape[:2]
        cls = self._classes[(self._frame // self.every_n_frames) % len(self._classes)]
        bw = self._seed.randint(int(w * 0.12), int(w * 0.3))
        bh = self._seed.randint(int(h * 0.15), int(h * 0.4))
        x = self._seed.randint(0, max(0, w - bw))
        y = self._seed.randint(0, max(0, h - bh))
        conf = round(self._seed.uniform(0.55, 0.93), 3)
        det = Detection(cls, conf, x, y, bw, bh, raw_class=f"mock_{cls}", detector=self.name)
        if self.log:
            self.log.debug("mock-детекция: %s %.2f", cls, conf)
        return [det]

    def info(self) -> dict:
        return {"name": self.name, "available": True, "mode": "mock",
                "every_n_frames": self.every_n_frames}
