"""Точка сборки FastAPI-приложения CV-модуля.

Создаёт конвейер обработки, подключает REST API и demo-интерфейс,
запускает/останавливает обработку вместе с жизненным циклом сервиса.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, shutdown
from .api import create_api_router
from .config import Config, load_config
from .logging_conf import setup_logging
from .pipeline import Pipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")

DESCRIPTION = """
Модуль компьютерного зрения для самоходного автоматизированного
телеметрического комплекса (САТК). Цикл: **видео → детекция → событие → API → UI**.

Целевая платформа: SiPeed Lichee Module 4A (TH1520). Demo-запуск — на ноутбуке.
""".strip()


def create_app(config: Config | None = None) -> FastAPI:
    cfg = config or load_config()
    logger = setup_logging(cfg)
    pipeline = Pipeline(cfg, logger)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        shutdown.reset()
        logger.info(
            "CV-модуль v%s запущен | режим=%s | backend=%s",
            __version__, cfg.app.mode, pipeline.detector.backend,
        )
        if cfg.app.autostart:
            pipeline.start()
        try:
            yield
        finally:
            # Сначала будим долгоживущие потоки (MJPEG), потом гасим конвейер:
            # так uvicorn не ждёт бесконечное соединение при Ctrl+C.
            shutdown.request_shutdown()
            pipeline.stop()
            logger.info("CV-модуль остановлен")

    app = FastAPI(
        title="CV-модуль САТК (SiPeed Lichee Module 4A)",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )
    app.state.config = cfg
    app.state.pipeline = pipeline

    app.include_router(create_api_router(pipeline))

    if os.path.isdir(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def demo_ui() -> FileResponse:
        index = os.path.join(STATIC_DIR, "index.html")
        if not os.path.exists(index):
            return FileResponse(index, status_code=404)  # type: ignore[arg-type]
        return FileResponse(index, media_type="text/html")

    return app


# Приложение по умолчанию: используется командой `uvicorn app.main:app`.
app = create_app()
