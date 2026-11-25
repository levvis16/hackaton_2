"""Замер производительности: FPS и latency на выбранном источнике.

Формирует отчёт docs/BENCHMARK.md и docs/benchmark.json.

    python scripts/benchmark.py --duration 20
    python scripts/benchmark.py --backend heuristics
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.config import load_config          # noqa: E402
from app.logging_conf import setup_logging  # noqa: E402
from app.pipeline import Pipeline           # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Замер FPS и latency CV-модуля")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--duration", type=float, default=20.0, help="длительность замера, с")
    parser.add_argument("--backend", choices=["auto", "yolo_onnx", "heuristics", "mock"])
    parser.add_argument("--source", help="путь к видеофайлу")
    args = parser.parse_args()

    cfg = load_config(os.path.join(ROOT, args.config))
    cfg.app.autostart = False
    cfg.video.loop = True
    cfg.video.target_fps = 0
    if args.backend:
        cfg.model.backend = args.backend
    if args.source:
        cfg.video.path = args.source

    logger = setup_logging(cfg)
    pipeline = Pipeline(cfg, logger)
    model_info = pipeline.detector.info()

    ok, message = pipeline.start()
    if not ok:
        print(f"Не удалось запустить обработку: {message}")
        return 1

    start = time.time()
    while time.time() - start < args.duration:
        time.sleep(0.5)
        status = pipeline.status()
        if status["state"] == "error":
            print(f"Ошибка обработки: {status['error']}")
            break
    pipeline.stop()

    snapshot = pipeline.metrics_snapshot()
    report = {
        "host": f"{platform.system()} {platform.release()} / Python {platform.python_version()}",
        "backend": model_info.get("backend"),
        "model_file": (model_info.get("yolo") or {}).get("model_path") if model_info.get("yolo") else None,
        "source": cfg.video.path or f"camera:{cfg.video.camera_index}",
        "resolution_note": f"process_width={cfg.video.process_width}, input_size={cfg.model.input_size}",
        "duration_sec": round(time.time() - start, 1),
        **snapshot,
    }

    docs_dir = os.path.join(ROOT, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    with open(os.path.join(docs_dir, "benchmark.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    markdown = _render_markdown(report)
    with open(os.path.join(docs_dir, "BENCHMARK.md"), "w", encoding="utf-8") as fh:
        fh.write(markdown)

    print(markdown)
    print(f"\nОтчёты сохранены: docs/BENCHMARK.md, docs/benchmark.json")
    return 0


def _render_markdown(report: dict) -> str:
    classes = report.get("per_class_events") or {}
    classes_md = "\n".join(f"| {k} | {v} |" for k, v in classes.items()) or "| — | 0 |"
    return f"""# Отчёт по производительности CV-модуля

Среда запуска: **{report['host']}**

| Параметр | Значение |
|---|---|
| Backend инференса | {report['backend']} |
| Файл модели | {report.get('model_file') or '—'} |
| Источник видео | {report['source']} |
| Параметры кадра | {report['resolution_note']} |
| Длительность замера, с | {report['duration_sec']} |

## Метрики

| Метрика | Значение |
|---|---|
| FPS (текущий, скользящее окно 10 с) | {report['fps']} |
| FPS (средний за сессию) | {report.get('fps_avg', '—')} |
| Latency (EMA), мс | {report['latency_ms']} |
| Latency средняя, мс | {report['latency_avg_ms']} |
| Latency p95, мс | {report['latency_p95_ms']} |
| Обработано кадров | {report['frames_processed']} |
| Пропущено кадров (frame_skip) | {report['frames_dropped']} |
| Создано событий | {report['events_created']} |
| Ошибок | {report['errors']} |

## События по классам

| Класс | Событий |
|---|---|
{classes_md}
"""


if __name__ == "__main__":
    raise SystemExit(main())
