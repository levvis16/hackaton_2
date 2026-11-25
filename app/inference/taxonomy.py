"""Таксономия целевых классов детекции и правила отображения в события.

Целевые классы (по ТЗ):
  person                  - человек
  person_without_helmet   - человек без каски / нарушение СИЗ
  vehicle                 - транспорт / техника
  smoke_fire              - дым / огонь
  spill                   - разлив / пятно / визуальная аномалия
  obstacle                - препятствие перед камерой
"""
from __future__ import annotations

from typing import Dict

# Соответствие целевого класса -> (тип события, критичность, человекочитаемая метка)
TARGET_CLASSES: Dict[str, dict] = {
    "person": {
        "event_type": "person_detected",
        "severity": "medium",
        "label": "Человек",
    },
    "person_without_helmet": {
        "event_type": "ppe_violation",
        "severity": "high",
        "label": "Человек без каски (нарушение СИЗ)",
    },
    "vehicle": {
        "event_type": "vehicle_detected",
        "severity": "low",
        "label": "Транспорт / техника",
    },
    "smoke_fire": {
        "event_type": "smoke_fire_detected",
        "severity": "critical",
        "label": "Дым / огонь",
    },
    "spill": {
        "event_type": "spill_detected",
        "severity": "high",
        "label": "Разлив / визуальная аномалия",
    },
    "obstacle": {
        "event_type": "obstacle_detected",
        "severity": "high",
        "label": "Препятствие перед камерой",
    },
}

# Соответствие классов датасета COCO (YOLOv8) целевым классам модуля.
COCO_TO_TARGET: Dict[str, str] = {
    "person": "person",
    "bicycle": "vehicle",
    "car": "vehicle",
    "motorcycle": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "train": "vehicle",
}

# Цвета отрисовки рамок (BGR) по целевому классу.
CLASS_COLORS: Dict[str, tuple[int, int, int]] = {
    "person": (0, 200, 0),
    "person_without_helmet": (0, 0, 230),
    "vehicle": (230, 150, 0),
    "smoke_fire": (0, 60, 255),
    "spill": (220, 0, 200),
    "obstacle": (0, 140, 255),
}


def event_type_for(class_name: str) -> str:
    return TARGET_CLASSES.get(class_name, {}).get("event_type", "object_detected")


def severity_for(class_name: str) -> str:
    return TARGET_CLASSES.get(class_name, {}).get("severity", "medium")


def label_for(class_name: str) -> str:
    return TARGET_CLASSES.get(class_name, {}).get("label", class_name)


def color_for(class_name: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(class_name, (200, 200, 200))


def map_coco(name: str) -> str | None:
    return COCO_TO_TARGET.get(name)
