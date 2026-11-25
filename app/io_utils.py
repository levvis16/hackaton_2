"""Утилиты ввода-вывода изображений с поддержкой Unicode-путей.

Штатные cv2.imread/cv2.imwrite на Windows не работают с не-ASCII путями
(например, «Рабочий стол»). Здесь чтение и запись выполняются через
np.fromfile/np.tofile + cv2.imdecode/cv2.imencode.
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    try:
        if not path or not os.path.exists(path):
            return None
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except (OSError, ValueError):
        return None


def imwrite_unicode(path: str, image: np.ndarray, params: Optional[list] = None) -> bool:
    try:
        ext = os.path.splitext(path)[1].lower() or ".jpg"
        ok, buffer = cv2.imencode(ext, image, params or [])
        if not ok:
            return False
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        buffer.tofile(path)
        return True
    except (OSError, ValueError):
        return False
