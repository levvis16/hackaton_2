"""Базовые типы детекции."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class Detection:
    """Одна детекция: целевой класс, уверенность и рамка (x, y, w, h)."""

    class_name: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    raw_class: str = ""
    detector: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def clip(self, frame_w: int, frame_h: int) -> "Detection":
        x1 = max(0, min(self.x, frame_w - 1))
        y1 = max(0, min(self.y, frame_h - 1))
        x2 = max(0, min(self.x + self.width, frame_w))
        y2 = max(0, min(self.y + self.height, frame_h))
        self.x, self.y = x1, y1
        self.width, self.height = max(0, x2 - x1), max(0, y2 - y1)
        return self

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": {
                "x": int(self.x),
                "y": int(self.y),
                "width": int(self.width),
                "height": int(self.height),
            },
            "raw_class": self.raw_class,
            "detector": self.detector,
        }


class BaseDetector(ABC):
    """Интерфейс детектора. Новый класс/модель добавляется реализацией detect()."""

    name = "base"

    @abstractmethod
    def detect(self, frame) -> List[Detection]:
        raise NotImplementedError

    def available(self) -> bool:
        return True

    def info(self) -> dict:
        return {"name": self.name, "available": self.available()}
