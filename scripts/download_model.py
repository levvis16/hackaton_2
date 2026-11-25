"""Загрузка модели YOLOv8 (ONNX), списка классов и демо-видео.

Используется для воспроизводимой установки:
    python scripts/download_model.py
Файлы кладутся в model/ и media/ рядом с проектом.
"""
from __future__ import annotations

import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASSETS = [
    (
        "models/yolov8n.onnx",
        "https://raw.githubusercontent.com/yoobright/yolo-onnx/master/yolov8n.onnx",
        "YOLOv8n, ONNX (детекция COCO: person, car, truck, bus, bicycle, ...)",
    ),
    (
        "models/coco.names",
        "https://raw.githubusercontent.com/yoobright/yolo-onnx/master/coco.names",
        "Список классов COCO (80 классов)",
    ),
    (
        "media/demo_person_bicycle_car.mp4",
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4",
        "Демо-видео (люди, велосипед, автомобиль)",
    ),
]


def download(rel_path: str, url: str, description: str) -> None:
    dst = os.path.join(ROOT, rel_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"[skip] {rel_path} уже существует ({os.path.getsize(dst)} байт)")
        return
    print(f"[get ] {description}\n       {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=180) as response, open(dst, "wb") as fh:
        total = 0
        while True:
            chunk = response.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
            total += len(chunk)
    print(f"[ok  ] {rel_path} ({total} байт)")


def main() -> int:
    for rel_path, url, description in ASSETS:
        try:
            download(rel_path, url, description)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {rel_path}: {exc}", file=sys.stderr)
            return 1
    print("\nГотово. При отсутствии модели модуль автоматически перейдёт в режим mock")
    print("(цикл детекция -> событие -> API/UI продолжит работать).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
