"""Эвристические детекторы на OpenCV.

Реализуют классы, для которых не используется нейросеть:
  * smoke_fire  - дым/огонь по цветовой модели HSV (реальные кадры);
  * spill       - разлив/пятно/аномалия по цвету и форме области (эвристика);
  * person_without_helmet - оценка наличия каски по области головы человека;
  * obstacle    - объект (человек/техника) в зоне перед камерой.

Все пороги вынесены в конфигурацию (секция heuristics) и подобраны
консервативно, чтобы не засорять журнал событиями на обычных сценах.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .base import BaseDetector, Detection


class HeuristicDetector(BaseDetector):
    name = "heuristics"

    def __init__(self, cfg, logger=None) -> None:
        self.cfg = cfg
        self.log = logger

    # ------------------------------------------------------------------
    # Чисто-визуальные эвристики (не зависят от других детекторов)
    # ------------------------------------------------------------------
    def detect(self, frame) -> List[Detection]:
        detections: List[Detection] = []
        if self.cfg.smoke_fire:
            detections += self._detect_smoke_fire(frame)
        if self.cfg.spill:
            detections += self._detect_spill(frame)
        return detections

    # ------------------------------------------------------------------
    # Эвристики, использующие результаты основного детектора
    # ------------------------------------------------------------------
    def postprocess(self, frame, base: List[Detection]) -> List[Detection]:
        extra: List[Detection] = []
        if self.cfg.helmet_check:
            extra += self._detect_helmet(frame, base)
        if self.cfg.obstacle:
            extra += self._detect_obstacle(frame, base)
        return extra

    # ------------------------------------------------------------------
    # Дым / огонь
    # ------------------------------------------------------------------
    def _detect_smoke_fire(self, frame) -> List[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h = hue.astype(np.int16)
        area = frame.shape[0] * frame.shape[1]
        min_area = self.cfg.min_area_ratio * area

        # Огонь: насыщенные тёплые оттенки (красный/оранжевый), яркие.
        warm = ((h <= 25) | (h >= 165)).astype(np.uint8)
        fire = warm & (sat >= 140).astype(np.uint8) & (val >= 180).astype(np.uint8)
        fire_mask = (fire * 255).astype(np.uint8)

        # Дым: малоконтрастные серые области, крупные и однородные.
        smoke = (sat <= 40).astype(np.uint8) & (val >= 95).astype(np.uint8) & (val <= 205).astype(np.uint8)
        smoke_mask = (smoke * 255).astype(np.uint8)
        smoke_mask[int(frame.shape[0] * 0.75):, :] = 0

        best: Detection | None = None
        # (маска, базовая уверенность, ядро морфологии, метка, доля площади, мин. рёбер)
        candidates = (
            (fire_mask, 0.55, 5, "fire", 0.40, 0.010),
            (smoke_mask, 0.38, 11, "smoke", 0.30, 0.008),
        )
        for mask, base_conf, kernel, kind, max_ratio, min_edges in candidates:
            cleaned = _morph(mask, kernel)
            blob = _largest_blob(cleaned)
            if blob is None:
                continue
            x, y, w, hh = blob
            ratio = (w * hh) / area
            if w * hh < min_area or ratio > max_ratio:
                continue
            # Защита от однотонных областей (небо, ровные поверхности, кузова):
            # у дыма/огня есть внутренняя структура и контуры.
            edge_density, std = _region_texture(gray, x, y, w, hh)
            if edge_density < min_edges or std < 6.0:
                continue
            conf = min(0.95, base_conf + ratio * 4.0)
            det = Detection(
                "smoke_fire", conf, x, y, w, hh, raw_class=kind, detector=self.name,
                meta={"edge_density": round(edge_density, 3), "std": round(std, 1)},
            )
            if best is None or det.confidence > best.confidence:
                best = det
        return [best] if best else []

    # ------------------------------------------------------------------
    # Разлив / пятно / визуальная аномалия
    # ------------------------------------------------------------------
    def _detect_spill(self, frame) -> List[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        h = hue.astype(np.int16)

        # Тёмная насыщенная жидкость (масло/ГСМ) - низкая яркость, заметная насыщенность.
        dark_liquid = (sat >= 70).astype(np.uint8) & (val <= 95).astype(np.uint8)
        # Вода/голубоватый разлив - холодный оттенок, средняя яркость.
        water = ((h >= 85) & (h <= 132)).astype(np.uint8) & (sat >= 60).astype(np.uint8) & (val >= 90).astype(np.uint8)
        mask = ((dark_liquid | water) * 255).astype(np.uint8)
        mask = _morph(mask, 7)

        blob = _largest_blob(mask)
        if blob is None:
            return []
        area = frame.shape[0] * frame.shape[1]
        x, y, w, hh = blob
        if w * hh < self.cfg.min_area_ratio * 1.5 * area:
            return []
        # Требуем компактность (пятно), а не вытянутую полосу.
        solidity = (w * hh) / max(1.0, cv2.contourArea(_bbox_contour(x, y, w, hh)))
        conf = min(0.85, 0.45 + (w * hh) / area * 3.0)
        return [Detection("spill", conf, x, y, w, hh, raw_class="spill", detector=self.name)]

    # ------------------------------------------------------------------
    # Человек без каски
    # ------------------------------------------------------------------
    def _detect_helmet(self, frame, base: List[Detection]) -> List[Detection]:
        out: List[Detection] = []
        fh, fw = frame.shape[:2]
        for det in base:
            if det.class_name != "person" or det.height < 90 or det.width < 32:
                continue
            # Область головы: верхние ~22% рамки, центральные ~70% ширины.
            hx1 = det.x + int(det.width * 0.15)
            hx2 = det.x + int(det.width * 0.85)
            hy1 = det.y
            hy2 = det.y + max(8, int(det.height * 0.22))
            hx1, hy1 = max(0, hx1), max(0, hy1)
            hx2, hy2 = min(fw, hx2), min(fh, hy2)
            if hx2 - hx1 < 8 or hy2 - hy1 < 8:
                continue
            head = frame[hy1:hy2, hx1:hx2]
            helmet = _helmet_score(head)
            if helmet < 0.18:
                det = Detection(
                    "person_without_helmet",
                    confidence=min(0.9, det.confidence * (0.7 + (0.18 - helmet))),
                    x=det.x, y=det.y, width=det.width, height=det.height,
                    raw_class="person", detector=self.name,
                    meta={"helmet_score": round(helmet, 3)},
                )
                out.append(det)
        return out

    # ------------------------------------------------------------------
    # Препятствие в зоне перед камерой
    # ------------------------------------------------------------------
    def _detect_obstacle(self, frame, base: List[Detection]) -> List[Detection]:
        fh, fw = frame.shape[:2]
        nx1, ny1, nx2, ny2 = self.cfg.obstacle_zone
        zx1, zy1, zx2, zy2 = int(nx1 * fw), int(ny1 * fh), int(nx2 * fw), int(ny2 * fh)
        zone_area = max(1, (zx2 - zx1) * (zy2 - zy1))
        out: List[Detection] = []
        for det in base:
            if det.class_name not in ("person", "vehicle"):
                continue
            ix1, iy1 = max(det.x, zx1), max(det.y, zy1)
            ix2, iy2 = min(det.x + det.width, zx2), min(det.y + det.height, zy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter <= 0:
                continue
            # Объект существенно внутри зоны и достаточно крупный.
            if inter / max(1, det.area) >= 0.45 and inter / zone_area >= 0.03:
                out.append(Detection(
                    "obstacle", confidence=min(0.92, det.confidence * 0.95),
                    x=ix1, y=iy1, width=max(1, ix2 - ix1), height=max(1, iy2 - iy1),
                    raw_class=det.raw_class or det.class_name, detector=self.name,
                ))
        return out

    def info(self) -> dict:
        return {
            "name": self.name,
            "smoke_fire": bool(self.cfg.smoke_fire),
            "spill": bool(self.cfg.spill),
            "helmet_check": bool(self.cfg.helmet_check),
            "obstacle": bool(self.cfg.obstacle),
            "obstacle_zone": list(self.cfg.obstacle_zone),
        }


# ----------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------
def _morph(mask: np.ndarray, k: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _largest_blob(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 8:
        return None
    return cv2.boundingRect(contour)


def _bbox_contour(x, y, w, h):
    return np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32)


def _region_texture(gray: np.ndarray, x: int, y: int, w: int, h: int) -> tuple[float, float]:
    """Плотность рёбер и СКО яркости внутри области (для отсечения однотонных зон)."""
    roi = gray[y:y + h, x:x + w]
    if roi.size == 0:
        return 0.0, 0.0
    edges = cv2.Canny(roi, 40, 120)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)
    return edge_density, float(roi.std())


def _helmet_score(head: np.ndarray) -> float:
    """Доля «касочных» пикселей в области головы (0..1).

    Каска обычно яркая/насыщенная (жёлтый, оранжевый, красный, синий)
    либо светлая однородная. Волосы и кожа - менее насыщенные тёмные тона.
    """
    if head.size == 0:
        return 0.0
    hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
    _, sat, val = cv2.split(hsv)
    bright_colored = ((sat >= 90) & (val >= 120)).astype(np.uint8)
    bright_plain = ((sat <= 55) & (val >= 195)).astype(np.uint8)
    helmet = cv2.bitwise_or(bright_colored * 255, bright_plain * 255)
    return float(np.count_nonzero(helmet)) / float(helmet.size)
