"""REST API CV-модуля.

Методы: /health, /status, /events, /events/{id}, /start, /stop, /config,
/frame/latest, /stream.mjpg, /metrics.
"""
from __future__ import annotations

import asyncio
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse

from .. import shutdown
from ..events.store import utc_now_iso
from ..pipeline import Pipeline
from ..schemas import (
    ActionResponse,
    ConfigRequest,
    Event,
    MetricsResponse,
    StartRequest,
    StatusResponse,
)

MJPEG_BOUNDARY = "frame"


async def mjpeg_stream(get_frame, stream_fps: int):
    """Асинхронный генератор кадров MJPEG.

    Завершается при отключении клиента и при остановке сервиса (Ctrl+C):
    иначе uvicorn бесконечно ждёт закрытия потока и «висит» при завершении.
    """
    interval = 1.0 / stream_fps
    last = None
    try:
        while not shutdown.is_shutting_down():
            data = get_frame()
            if data is not None and data is not last:
                last = data
                yield (
                    f"--{MJPEG_BOUNDARY}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(data)}\r\n\r\n"
                ).encode() + data + b"\r\n"
            await asyncio.sleep(interval)
    except asyncio.CancelledError:  # клиент отключился / сервер остановлен
        return


def create_api_router(pipeline: Pipeline) -> APIRouter:
    router = APIRouter(tags=["cv"])
    log = pipeline.log

    # ------------------------------------------------------------------
    @router.get("/health", summary="Проверка работоспособности сервиса")
    def health() -> dict:
        return {
            "status": "ok",
            "service": "cv-module-satk",
            "timestamp": utc_now_iso(),
            "state": pipeline.state,
        }

    # ------------------------------------------------------------------
    @router.get("/status", response_model=StatusResponse, summary="Текущий статус обработки")
    def status() -> dict:
        return pipeline.status()

    # ------------------------------------------------------------------
    @router.get("/metrics", response_model=MetricsResponse, summary="FPS, latency и счётчики")
    def metrics() -> dict:
        return pipeline.metrics_snapshot()

    # ------------------------------------------------------------------
    @router.get("/events", response_model=List[Event], summary="Список событий")
    def list_events(
        limit: int = Query(100, ge=1, le=1000),
        object_class: Optional[str] = Query(None, description="Фильтр по целевому классу"),
        event_type: Optional[str] = Query(None, description="Фильтр по типу события"),
        severity: Optional[str] = Query(None, description="Фильтр по критичности"),
        since: Optional[str] = Query(None, description="События не раньше ISO-8601, напр. 2026-07-13T12:00:00Z"),
    ) -> list:
        return pipeline.events.list(
            limit=limit, event_class=object_class,
            event_type=event_type, severity=severity, since=since,
        )

    # ------------------------------------------------------------------
    @router.get("/events/{event_id}", response_model=Event, summary="Получить событие по id")
    def get_event(event_id: str) -> dict:
        event = pipeline.events.get(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail=f"Событие {event_id} не найдено")
        return event

    # ------------------------------------------------------------------
    @router.get("/events/{event_id}/frame", summary="Кадр события (JPEG)")
    def get_event_frame(event_id: str) -> FileResponse:
        path = pipeline.events.frame_abs_path(event_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Кадр события не найден")
        return FileResponse(path, media_type="image/jpeg")

    # ------------------------------------------------------------------
    @router.post("/start", response_model=ActionResponse, summary="Запуск обработки видеопотока")
    def start(body: Optional[StartRequest] = None) -> dict:
        override = body.model_dump(exclude_none=True) if body else None
        ok, message = pipeline.start(override)
        return {"ok": ok, "message": message, "state": pipeline.state, "status": pipeline.status()}

    # ------------------------------------------------------------------
    @router.post("/stop", response_model=ActionResponse, summary="Остановка обработки")
    def stop() -> dict:
        ok, message = pipeline.stop()
        return {"ok": ok, "message": message, "state": pipeline.state, "status": pipeline.status()}

    # ------------------------------------------------------------------
    @router.post("/config", response_model=ActionResponse, summary="Изменение параметров (порог confidence, источник)")
    def update_config(body: ConfigRequest) -> dict:
        changed = []
        if body.confidence_threshold is not None:
            pipeline.set_confidence(body.confidence_threshold)
            changed.append(f"confidence_threshold={body.confidence_threshold}")
        if body.min_confidence is not None:
            pipeline.set_min_confidence(body.min_confidence)
            changed.append(f"min_confidence={body.min_confidence}")
        if body.frame_skip is not None:
            pipeline.cfg.video.frame_skip = int(body.frame_skip)
            changed.append(f"frame_skip={body.frame_skip}")
        if body.target_fps is not None:
            pipeline.cfg.video.target_fps = float(body.target_fps)
            changed.append(f"target_fps={body.target_fps}")

        source_changed = False
        video = pipeline.cfg.video
        if body.source_type is not None:
            video.source_type = body.source_type
            source_changed = True
        if body.path is not None:
            video.path = body.path
            source_changed = True
        if body.camera_index is not None:
            video.camera_index = int(body.camera_index)
            source_changed = True
        if body.rtsp_url is not None:
            video.rtsp_url = body.rtsp_url
            source_changed = True

        message = "Конфигурация обновлена: " + (", ".join(changed) if changed else "изменений нет")
        if source_changed:
            message += " (источник применится при следующем /start)"
            log.info("Источник видео изменён через API: %s", video.source_type)
        return {"ok": True, "message": message, "state": pipeline.state, "status": pipeline.status()}

    # ------------------------------------------------------------------
    @router.delete("/events", summary="Очистить журнал событий (в памяти)")
    def clear_events() -> dict:
        removed = pipeline.events.clear()
        return {"ok": True, "message": f"Удалено событий: {removed}", "removed": removed}

    # ------------------------------------------------------------------
    @router.get("/frame/latest", summary="Последний обработанный кадр (JPEG)")
    def latest_frame() -> Response:
        data = pipeline.latest_frame()
        if data is None:
            raise HTTPException(status_code=404, detail="Кадр ещё не сформирован")
        return Response(content=data, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    # ------------------------------------------------------------------
    @router.get("/stream.mjpg", summary="Видеопоток MJPEG с рамками детекций")
    def stream(stream_fps: int = Query(15, ge=1, le=60)) -> StreamingResponse:
        headers = {
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "Connection": "close",
        }
        return StreamingResponse(
            mjpeg_stream(pipeline.latest_frame, stream_fps),
            media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
            headers=headers,
        )

    return router
