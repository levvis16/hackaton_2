"""Конвейер обработки видеопотока.

Поток-исполнитель: чтение кадра -> предобработка -> инференс -> формирование
событий -> отрисовка -> публикация кадра для UI. Метрики FPS/latency
обновляются на каждом обработанном кадре. Ошибки источника и инференса
логируются, но не приводят к аварийному завершению сервиса.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, Optional, Tuple

from .config import Config
from .events.store import EventStore
from .inference.ensemble import DetectionEnsemble
from .input.video_source import VideoSource, VideoSourceError
from .metrics import Metrics
from .render import draw_detections, encode_jpeg


class Pipeline:
    def __init__(self, cfg: Config, logger) -> None:
        self.cfg = cfg
        self.log = logger
        self.metrics = Metrics()
        self.events = EventStore(
            cfg.events, cfg.app.camera_id, cfg.base_dir, logger=logger
        )
        self.detector = DetectionEnsemble(cfg, logger)

        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.state = "stopped"          # stopped | starting | running | finished | error
        self.error: Optional[str] = None
        self._source_meta: Dict = {}
        self._latest_jpeg: Optional[bytes] = None
        self._latest_ts: float = 0.0

    # ------------------------------------------------------------------
    # Управление
    # ------------------------------------------------------------------
    def start(self, override: Optional[dict] = None) -> Tuple[bool, str]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False, "Обработка уже запущена"
            self._apply_override(override)
            self._stop.clear()
            self.state = "starting"
            self.error = None
            self._source_meta = {}
            self._thread = threading.Thread(target=self._run, name="cv-pipeline", daemon=True)
            self._thread.start()
        self.log.info(
            "Запуск обработки: source_type=%s source=%s",
            self.cfg.video.source_type, self.cfg.video.path or self.cfg.video.camera_index,
        )
        return True, "Запуск обработки"

    def stop(self, timeout: float = 5.0) -> Tuple[bool, str]:
        with self._lock:
            thread = self._thread
            if thread is None or not thread.is_alive():
                if self.state not in ("error", "finished"):
                    self.state = "stopped"
                return False, "Обработка уже остановлена"
            self._stop.set()
        thread.join(timeout)
        with self._lock:
            if thread.is_alive():
                self.log.warning("Поток обработки не завершился за %.1f c", timeout)
            if self.state not in ("error", "finished"):
                self.state = "stopped"
        self.log.info("Обработка остановлена")
        return True, "Остановка обработки"

    def is_running(self) -> bool:
        with self._lock:
            return self.state in ("running", "starting")

    # ------------------------------------------------------------------
    # Основной цикл
    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            source = VideoSource(self.cfg.video, self.cfg.base_dir).open()
        except VideoSourceError as exc:
            with self._lock:
                self.state = "error"
                self.error = str(exc)
                self._thread = None
            self.log.error("Источник видео недоступен: %s", exc)
            return

        with self._lock:
            self.state = "running"
            self.error = None
            self._source_meta = source.info()
            self.events.source = source.description
        self.log.info("Обработка запущена: %s", self._source_meta)

        frame_index = 0
        error_streak = 0
        min_interval = 1.0 / self.cfg.video.target_fps if self.cfg.video.target_fps > 0 else 0.0
        last_ts = time.monotonic()
        skip = max(0, int(self.cfg.video.frame_skip))

        try:
            while not self._stop.is_set():
                ok, frame = source.read()
                if not ok or frame is None:
                    action = self._on_read_failure(source)
                    if action in ("reconnect", "loop"):
                        continue
                    break

                frame_index += 1
                if skip > 0 and (frame_index % (skip + 1)) != 0:
                    self.metrics.on_drop()
                    continue

                if min_interval > 0:
                    wait = last_ts + min_interval - time.monotonic()
                    if wait > 0 and self._stop.wait(wait):
                        break
                    last_ts = time.monotonic()

                t0 = time.perf_counter()
                try:
                    detections = self.detector.detect(frame)
                    error_streak = 0
                except Exception as exc:  # noqa: BLE001
                    self.metrics.on_error()
                    error_streak += 1
                    self.log.exception("Ошибка инференса: %s", exc)
                    if error_streak > 30:
                        with self._lock:
                            self.state = "error"
                            self.error = f"Серия ошибок инференса: {exc}"
                        break
                    continue

                latency_ms = (time.perf_counter() - t0) * 1000.0
                annotated = draw_detections(frame, detections, self._hud_lines(detections, latency_ms))

                created_classes = []
                for det in detections:
                    if self.events.maybe_create(det, annotated) is not None:
                        created_classes.append(det.class_name)

                self.metrics.on_frame(latency_ms)
                if created_classes:
                    self.metrics.on_events(len(created_classes), created_classes)

                self._publish_frame(annotated)
        finally:
            source.release()
            with self._lock:
                if self.state == "running":
                    self.state = "stopped"
                self._source_meta = {**self._source_meta, "processed_frames": frame_index}
            self.log.info("Цикл обработки завершён (кадров: %d)", frame_index)

    # ------------------------------------------------------------------
    def _on_read_failure(self, source: VideoSource) -> str:
        if source.is_file():
            with self._lock:
                self.state = "finished"
            self.log.info("Видеофайл прочитан до конца")
            return "stop"
        if source.is_static():
            return "loop"

        attempts = max(0, int(self.cfg.video.reconnect_attempts))
        for attempt in range(1, attempts + 1):
            if self._stop.is_set():
                return "stop"
            self.log.warning("Поток недоступен, переподключение %d/%d", attempt, attempts)
            if self._stop.wait(max(0.1, float(self.cfg.video.reconnect_delay_sec))):
                return "stop"
            try:
                source.reopen()
                with self._lock:
                    self._source_meta = source.info()
                self.log.info("Переподключение к источнику выполнено")
                return "reconnect"
            except VideoSourceError as exc:
                self.log.warning("Переподключение не удалось: %s", exc)
        with self._lock:
            self.state = "error"
            self.error = "Не удалось восстановить видеопоток"
        return "stop"

    def _hud_lines(self, detections, latency_ms: float):
        return [
            f"FPS {self.metrics.fps():.1f} | latency {latency_ms:.0f} ms",
            f"source {self._source_meta.get('source_type', '?')} | detections {len(detections)}",
            f"mode {self.cfg.app.mode} | state {self.state} | events {self.metrics.events_created}",
        ]

    def _publish_frame(self, annotated) -> None:
        data = encode_jpeg(annotated, quality=75)
        if data:
            with self._lock:
                self._latest_jpeg = data
                self._latest_ts = time.time()

    # ------------------------------------------------------------------
    def latest_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def latest_frame_ts(self) -> float:
        with self._lock:
            return self._latest_ts

    # ------------------------------------------------------------------
    def status(self) -> dict:
        snap = self.metrics.snapshot()
        with self._lock:
            return {
                "state": self.state,
                "mode": self.cfg.app.mode,
                "source_type": self.cfg.video.source_type,
                "source": self._source_meta.get("description", self.cfg.video.path),
                "camera_id": self.cfg.app.camera_id,
                "fps": snap["fps"],
                "latency_ms": snap["latency_ms"],
                "frames_processed": snap["frames_processed"],
                "events_created": snap["events_created"],
                "uptime_sec": snap["uptime_sec"],
                "error": self.error,
                "model": self.detector.info(),
                "source_meta": dict(self._source_meta),
            }

    def metrics_snapshot(self) -> dict:
        return self.metrics.snapshot()

    # ------------------------------------------------------------------
    def set_confidence(self, value: float) -> None:
        self.cfg.model.confidence_threshold = float(value)
        self.detector.set_confidence(value)
        self.log.info("Порог confidence изменён: %.2f", value)

    def set_min_confidence(self, value: float) -> None:
        self.cfg.events.min_confidence = float(value)
        self.events.cfg.min_confidence = float(value)
        self.log.info("Порог событий изменён: %.2f", value)

    def _apply_override(self, override: Optional[dict]) -> None:
        if not override:
            return
        video = self.cfg.video
        if override.get("source_type"):
            video.source_type = str(override["source_type"])
        if override.get("path"):
            video.path = str(override["path"])
        if override.get("camera_index") is not None:
            video.camera_index = int(override["camera_index"])
        if override.get("rtsp_url"):
            video.rtsp_url = str(override["rtsp_url"])
