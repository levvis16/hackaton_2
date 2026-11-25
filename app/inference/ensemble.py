"""Ансамбль детекторов: объединяет нейросеть, эвристики и mock.

Позволяет расширять модуль: чтобы добавить новый класс/модель, достаточно
добавить детектор и включить его в сборку (factory / DetectionEnsemble.detect).
"""
from __future__ import annotations

import os
from typing import List, Optional

from .base import Detection
from .heuristics import HeuristicDetector
from .mock import MockDetector
from .yolo_onnx import YoloOnnxDetector


class DetectionEnsemble:
    def __init__(self, cfg, logger=None) -> None:
        self.cfg = cfg
        self.log = logger
        self.yolo: Optional[YoloOnnxDetector] = None
        self.heuristics: Optional[HeuristicDetector] = None
        self.mock: Optional[MockDetector] = None
        self.backend: str = "none"
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        backend = (self.cfg.model.backend or "auto").lower()

        if backend in ("auto", "yolo_onnx"):
            detector = YoloOnnxDetector(
                model_path=self.cfg.model_path,
                labels_path=self.cfg.labels_path,
                input_size=self.cfg.model.input_size,
                confidence_threshold=self.cfg.model.confidence_threshold,
                iou_threshold=self.cfg.model.iou_threshold,
                intra_op_num_threads=getattr(self.cfg.model, "intra_op_num_threads", 0),
                providers=list(self.cfg.model.providers),
                logger=self.log,
            )
            if detector.load():
                self.yolo = detector
                self.backend = "yolo_onnx"
            elif self.log:
                self.log.warning(
                    "YOLO ONNX недоступна (%s). Продолжаем на эвристиках.",
                    detector.info().get("error"),
                )

        if backend in ("auto", "yolo_onnx", "heuristics"):
            self.heuristics = HeuristicDetector(self.cfg.heuristics, logger=self.log)

        # Если нейросеть не поднялась и явно не запрошены только эвристики - добавим mock,
        # чтобы цикл «детекция -> событие -> API/UI» был работоспособен всегда.
        if backend == "mock" or (self.yolo is None and backend == "auto"):
            self.mock = MockDetector(every_n_frames=20, logger=self.log)

        # Итоговая метка активного бэкенда для /status и /metrics.
        if self.yolo is not None:
            self.backend = "yolo_onnx"
        elif self.mock is not None:
            self.backend = "mock"
        elif self.heuristics is not None:
            self.backend = "heuristics"

    # ------------------------------------------------------------------
    def detect(self, frame) -> List[Detection]:
        detections: List[Detection] = []
        if self.yolo is not None:
            detections += self.yolo.detect(frame)
        if self.mock is not None:
            detections += self.mock.detect(frame)
        if self.heuristics is not None:
            detections += self.heuristics.detect(frame)
            detections += self.heuristics.postprocess(frame, detections)
        return _suppress_duplicates(detections)

    def set_confidence(self, value: float) -> None:
        if self.yolo is not None:
            self.yolo.set_confidence(value)

    def info(self) -> dict:
        return {
            "backend": self.backend,
            "yolo": self.yolo.info() if self.yolo else None,
            "heuristics": self.heuristics.info() if self.heuristics else None,
            "mock": self.mock.info() if self.mock else None,
            "model_file": os.path.basename(self.cfg.model.path) if self.yolo else None,
        }


def _suppress_duplicates(detections: List[Detection], iou_threshold: float = 0.6) -> List[Detection]:
    """Убирает почти совпадающие детекции одного класса."""
    result: List[Detection] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(
            not (keep.class_name == det.class_name and _iou(keep, det) > iou_threshold)
            for keep in result
        ):
            result.append(det)
    return result


def _iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.width, a.y + a.height
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.width, b.y + b.height
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0
