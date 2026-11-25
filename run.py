"""Запуск сервиса CV-модуля САТК.

Примеры:
  python run.py
  python run.py --source media/demo_person_bicycle_car.mp4 --conf 0.4
  python run.py --source-type camera --camera-index 0
  python run.py --mode edge --backend yolo_onnx
  python run.py --reload
"""
from __future__ import annotations

import argparse
import os

import uvicorn

from app.config import load_config
from app.logging_conf import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CV-модуль САТК — сервис детекции и событий")
    parser.add_argument("--config", default="config.yaml", help="путь к YAML-конфигурации")
    parser.add_argument("--host", help="адрес прослушивания")
    parser.add_argument("--port", type=int, help="порт API")
    parser.add_argument("--mode", choices=["demo", "edge"], help="режим запуска")
    parser.add_argument("--source-type", choices=["file", "camera", "rtsp", "image", "synthetic"])
    parser.add_argument("--source", help="путь к видеофайлу или изображению")
    parser.add_argument("--camera-index", type=int, help="индекс камеры")
    parser.add_argument("--conf", type=float, help="порог confidence (модель и события)")
    parser.add_argument("--backend", choices=["auto", "yolo_onnx", "heuristics", "mock"])
    parser.add_argument("--threads", type=int, help="число потоков ONNX Runtime (0 = авто)")
    parser.add_argument("--no-autostart", action="store_true", help="не запускать обработку сразу")
    parser.add_argument("--reload", action="store_true", help="автоперезапуск (для разработки)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Параметры передаются приложению через переменные окружения, чтобы
    # приложение создавалось ровно один раз (в процессе uvicorn).
    os.environ["CV_CONFIG"] = os.path.abspath(args.config)
    env = {
        "CV_HOST": args.host,
        "CV_PORT": args.port,
        "CV_MODE": args.mode,
        "CV_SOURCE_TYPE": args.source_type,
        "CV_SOURCE": args.source,
        "CV_CAMERA_INDEX": args.camera_index,
        "CV_CONF": args.conf,
        "CV_BACKEND": args.backend,
        "CV_THREADS": args.threads,
    }
    for key, value in env.items():
        if value is not None:
            os.environ[key] = str(value)
    os.environ["CV_AUTOSTART"] = "0" if args.no_autostart else "1"
    if args.conf is not None:
        os.environ["CV_MIN_CONF"] = str(args.conf)

    cfg = load_config(args.config)
    logger = setup_logging(cfg)
    url = f"http://{'127.0.0.1' if cfg.app.host in ('0.0.0.0', '::') else cfg.app.host}:{cfg.app.port}"
    logger.info("Demo UI: %s  |  API-документация: %s/docs", url, url)

    uvicorn.run(
        "app.main:app",
        host=cfg.app.host,
        port=cfg.app.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
