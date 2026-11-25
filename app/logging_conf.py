"""Настройка логирования: консоль + файл.

Логируются запуск/остановка обработки, ошибки чтения источника,
созданные события и метрики.
"""
from __future__ import annotations

import logging
import os
import sys

_LOGGER_NAME = "cv"


def setup_logging(cfg) -> logging.Logger:
    # Корректный вывод кириллицы в консоль Windows.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(getattr(logging, str(cfg.logging.level).upper(), logging.INFO))
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if cfg.logging.console:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        logger.addHandler(console)

    try:
        log_path = cfg.log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError:
        logger.warning("Не удалось открыть файл логов: %s", cfg.log_path)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME if name is None else f"{_LOGGER_NAME}.{name}")
