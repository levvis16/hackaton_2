"""Пакет CV-инференса."""
from .base import BaseDetector, Detection
from .ensemble import DetectionEnsemble
from .taxonomy import TARGET_CLASSES, event_type_for, severity_for

__all__ = [
    "BaseDetector",
    "Detection",
    "DetectionEnsemble",
    "TARGET_CLASSES",
    "event_type_for",
    "severity_for",
]
