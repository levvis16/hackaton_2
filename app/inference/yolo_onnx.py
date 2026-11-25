"""Детектор объектов на базе YOLOv8 в формате ONNX (ONNX Runtime).

Модель: YOLOv8n, вход 1x3x640x640 (float32, BGR->RGB, /255),
выход 1x84x8400. Классы детекции приводятся к таксономии модуля
(человек, транспорт/техника).

Модуль не требует фреймворка обучения: только onnxruntime + opencv.
"""
from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np

from .base import BaseDetector, Detection
from .taxonomy import map_coco

try:  # onnxruntime нужен только для этого детектора
    import onnxruntime as ort
except Exception:  # pragma: no cover - отсутствие зависимости не роняет модуль
    ort = None


class YoloOnnxDetector(BaseDetector):
    name = "yolo_onnx"

    def __init__(
        self,
        model_path: str,
        labels_path: str,
        input_size: int = 640,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        intra_op_num_threads: int = 0,
        providers: Optional[list] = None,
        logger=None,
    ) -> None:
        self.model_path = model_path
        self.labels_path = labels_path
        self.input_size = int(input_size)
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.intra_op_num_threads = int(intra_op_num_threads or 0)
        self.providers = providers or ["CPUExecutionProvider"]
        self.log = logger
        self.session = None
        self.input_name = "images"
        self.classes: List[str] = []
        self._load_error: Optional[str] = None

    # ------------------------------------------------------------------
    def load(self) -> bool:
        if ort is None:
            self._load_error = "onnxruntime не установлен"
            return False
        if not os.path.exists(self.model_path):
            self._load_error = f"файл модели не найден: {self.model_path}"
            return False
        try:
            available = ort.get_available_providers()
            providers = [p for p in self.providers if p in available] or ["CPUExecutionProvider"]
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            threads = int(getattr(self, "intra_op_num_threads", 0) or 0)
            if threads > 0:
                so.intra_op_num_threads = threads
                so.inter_op_num_threads = 1
            self.session = ort.InferenceSession(self.model_path, sess_options=so, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"ошибка загрузки ONNX: {exc}"
            return False

        try:
            with open(self.labels_path, "r", encoding="utf-8") as fh:
                self.classes = [line.strip() for line in fh if line.strip()]
        except OSError:
            # Запасной вариант: стандартный список классов COCO (80 шт.).
            self.classes = _FALLBACK_COCO
        if self.log:
            self.log.info(
                "YOLO ONNX загружена: %s | классов: %d | провайдеры: %s",
                os.path.basename(self.model_path), len(self.classes), providers,
            )
        return True

    def available(self) -> bool:
        return self.session is not None

    def set_confidence(self, value: float) -> None:
        self.confidence_threshold = float(value)

    # ------------------------------------------------------------------
    def detect(self, frame) -> List[Detection]:
        if self.session is None:
            return []
        orig_h, orig_w = frame.shape[:2]
        blob, ratio, pad = self._preprocess(frame)
        outputs = self.session.run(None, {self.input_name: blob})
        return self._postprocess(outputs[0], ratio, pad, orig_w, orig_h)

    # ------------------------------------------------------------------
    def _preprocess(self, frame):
        size = self.input_size
        h, w = frame.shape[:2]
        ratio = min(size / h, size / w)
        new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
        if (w, h) != (new_w, new_h):
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame
        dw, dh = (size - new_w) // 2, (size - new_h) // 2
        top, bottom = dh, size - new_h - dh
        left, right = dw, size - new_w - dw
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(blob), ratio, (left, top)

    def _postprocess(self, output, ratio, pad, orig_w, orig_h) -> List[Detection]:
        pred = np.squeeze(output)          # (84, 8400)
        if pred.ndim != 2:
            return []
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T                   # (8400, 84)
        boxes = pred[:, :4]
        scores = pred[:, 4:]
        class_ids = scores.argmax(axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        keep = confidences >= self.confidence_threshold
        boxes, confidences, class_ids = boxes[keep], confidences[keep], class_ids[keep]
        if len(boxes) == 0:
            return []

        # xywh (center) -> xyxy
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1).tolist()
        idxs = cv2.dnn.NMSBoxes(
            xyxy, confidences.tolist(), self.confidence_threshold, self.iou_threshold
        )
        if len(idxs) == 0:
            return []

        left, top = pad
        detections: List[Detection] = []
        for i in np.array(idxs).reshape(-1):
            raw_name = self._class_name(int(class_ids[i]))
            target = map_coco(raw_name)
            if target is None:
                continue  # классы вне таксономии модуля игнорируем
            bx1, by1, bx2, by2 = xyxy[i]
            x1 = int((bx1 - left) / ratio)
            y1 = int((by1 - top) / ratio)
            x2 = int((bx2 - left) / ratio)
            y2 = int((by2 - top) / ratio)
            det = Detection(
                class_name=target,
                confidence=float(confidences[i]),
                x=max(0, min(orig_w - 1, x1)),
                y=max(0, min(orig_h - 1, y1)),
                width=max(0, min(orig_w, x2) - max(0, min(orig_w - 1, x1))),
                height=max(0, min(orig_h, y2) - max(0, min(orig_h - 1, y1))),
                raw_class=raw_name,
                detector=self.name,
            )
            if det.width > 2 and det.height > 2:
                detections.append(det)
        # Сначала самые уверенные детекции
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def _class_name(self, idx: int) -> str:
        if 0 <= idx < len(self.classes):
            return self.classes[idx]
        return str(idx)

    def info(self) -> dict:
        return {
            "name": self.name,
            "available": self.available(),
            "model_path": self.model_path,
            "input_size": self.input_size,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "intra_op_num_threads": self.intra_op_num_threads,
            "num_classes": len(self.classes),
            "providers": self.providers,
            "error": self._load_error,
        }


_FALLBACK_COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]
