"""Автотесты CV-модуля САТК (без внешних зависимостей, кроме FastAPI/OpenCV).

Запуск:
    python -m unittest discover -s tests -v
или (если установлен pytest):
    pytest -q

Тесты не требуют загруженного файла модели: для API используется backend=mock,
для эвристик — синтетические кадры, собранные прямо в тесте.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import load_config  # noqa: E402
from app.events.store import EventGate, EventStore  # noqa: E402
from app.inference.base import Detection  # noqa: E402
from app.inference.heuristics import HeuristicDetector  # noqa: E402
from app.inference.taxonomy import (  # noqa: E402
    TARGET_CLASSES,
    event_type_for,
    map_coco,
    severity_for,
)
from app.input.video_source import VideoSource, VideoSourceError  # noqa: E402
from app.main import create_app  # noqa: E402


def _tmp_config() -> tuple:
    """Конфигурация с изолированным каталогом вывода (не засоряет проект)."""
    cfg = load_config(os.path.join(PROJECT_ROOT, "config.yaml"))
    tmp = tempfile.mkdtemp(prefix="cv_test_")
    cfg.base_dir = tmp
    cfg.app.autostart = True
    cfg.logging.level = "WARNING"
    cfg.logging.file = os.path.join(tmp, "test.log")
    return cfg, tmp


# ----------------------------------------------------------------------
# Таксономия классов
# ----------------------------------------------------------------------
class TestTaxonomy(unittest.TestCase):
    def test_all_target_classes_present(self):
        expected = {"person", "person_without_helmet", "vehicle",
                    "smoke_fire", "spill", "obstacle"}
        self.assertEqual(set(TARGET_CLASSES), expected)

    def test_event_type_and_severity(self):
        self.assertEqual(event_type_for("person"), "person_detected")
        self.assertEqual(event_type_for("smoke_fire"), "smoke_fire_detected")
        self.assertEqual(severity_for("smoke_fire"), "critical")
        self.assertEqual(severity_for("vehicle"), "low")

    def test_coco_mapping_keeps_only_relevant_classes(self):
        self.assertEqual(map_coco("person"), "person")
        self.assertEqual(map_coco("car"), "vehicle")
        self.assertEqual(map_coco("bicycle"), "vehicle")
        self.assertIsNone(map_coco("toaster"))


# ----------------------------------------------------------------------
# Источник видео
# ----------------------------------------------------------------------
class TestVideoSource(unittest.TestCase):
    def test_synthetic_source_reads_frames(self):
        cfg = load_config(os.path.join(PROJECT_ROOT, "config.yaml"))
        cfg.video.source_type = "synthetic"
        src = VideoSource(cfg.video, PROJECT_ROOT).open()
        ok, frame = src.read()
        self.assertTrue(ok)
        self.assertEqual(frame.shape[2], 3)

    def test_missing_file_raises_instead_of_crashing(self):
        cfg = load_config(os.path.join(PROJECT_ROOT, "config.yaml"))
        cfg.video.source_type = "file"
        cfg.video.path = "media/__no_such_file__.mp4"
        with self.assertRaises(VideoSourceError):
            VideoSource(cfg.video, PROJECT_ROOT).open()

    def test_rtsp_without_url_raises(self):
        cfg = load_config(os.path.join(PROJECT_ROOT, "config.yaml"))
        cfg.video.source_type = "rtsp"
        cfg.video.rtsp_url = ""
        cfg.video.path = ""
        with self.assertRaises(VideoSourceError):
            VideoSource(cfg.video, PROJECT_ROOT).open()


# ----------------------------------------------------------------------
# Формирование событий
# ----------------------------------------------------------------------
class TestEventStore(unittest.TestCase):
    def setUp(self):
        cfg, self.tmp = _tmp_config()
        self.cfg = cfg
        self.store = EventStore(cfg.events, "cam_test", cfg.base_dir)

    def test_event_has_required_fields(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        det = Detection("person", 0.91, 10, 12, 40, 60, raw_class="person", detector="test")
        event = self.store.maybe_create(det, frame)
        self.assertIsNotNone(event)
        for key in ("id", "timestamp", "event_type", "object_class", "confidence",
                    "bbox", "frame_path", "camera_id", "severity", "status"):
            self.assertIn(key, event)
        self.assertEqual(event["event_type"], "person_detected")
        self.assertEqual(event["severity"], "medium")
        self.assertEqual(event["camera_id"], "cam_test")
        self.assertEqual(set(event["bbox"]), {"x", "y", "width", "height"})
        self.assertEqual(self.store.count(), 1)

    def test_frame_is_saved_and_resolvable(self):
        frame = np.full((120, 160, 3), 100, dtype=np.uint8)
        det = Detection("vehicle", 0.8, 5, 5, 50, 50, raw_class="car", detector="test")
        event = self.store.maybe_create(det, frame)
        self.assertIsNotNone(event["frame_path"])
        self.assertTrue(os.path.exists(os.path.join(self.tmp, event["frame_path"])))
        self.assertIsNotNone(self.store.frame_abs_path(event["id"]))

    def test_duplicate_suppressed_within_cooldown(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        det = Detection("person", 0.9, 10, 10, 40, 60, raw_class="person")
        first = self.store.maybe_create(det, frame)
        second = self.store.maybe_create(det, frame)
        self.assertIsNotNone(first)
        self.assertIsNone(second)

    def test_confidence_below_min_is_ignored(self):
        self.store.cfg.min_confidence = 0.99
        det = Detection("person", 0.5, 10, 10, 40, 60)
        self.assertIsNone(self.store.maybe_create(det, None))

    def test_event_is_persisted_to_jsonl(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        self.store.maybe_create(Detection("spill", 0.7, 1, 2, 30, 30), frame)
        with open(self.store.store_path, "r", encoding="utf-8") as fh:
            lines = [ln for ln in fh if ln.strip()]
        self.assertEqual(len(lines), 1)


class TestEventGate(unittest.TestCase):
    def test_gate_blocks_repeat_and_assigns_track(self):
        gate = EventGate(cooldown_sec=5.0)
        det = Detection("person", 0.9, 0, 0, 100, 100)
        ok1, tid1 = gate.reserve(det)
        ok2, tid2 = gate.reserve(det)
        self.assertTrue(ok1)
        self.assertFalse(ok2)
        self.assertEqual(tid1, tid2)

    def test_gate_allows_after_cooldown(self):
        gate = EventGate(cooldown_sec=0.0)
        det = Detection("person", 0.9, 0, 0, 100, 100)
        self.assertTrue(gate.reserve(det)[0])
        self.assertTrue(gate.reserve(det)[0])


# ----------------------------------------------------------------------
# Эвристики (smoke_fire / spill)
# ----------------------------------------------------------------------
def _fire_frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 170), (330, 300), (0, 140, 255), -1)
    roi = frame[170:300, 200:330].astype(np.int16)
    noise = rng.integers(-45, 46, roi.shape, dtype=np.int16)
    frame[170:300, 200:330] = np.clip(roi + noise, 0, 255).astype(np.uint8)
    return frame


def _spill_frame() -> np.ndarray:
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    cv2.ellipse(frame, (320, 240), (95, 65), 0, 0, 360, (90, 20, 10), -1)
    return frame


class TestHeuristics(unittest.TestCase):
    def setUp(self):
        cfg, _ = _tmp_config()
        self.hev = HeuristicDetector(cfg.heuristics)

    def test_fire_is_detected(self):
        dets = self.hev.detect(_fire_frame())
        self.assertTrue(any(d.class_name == "smoke_fire" for d in dets),
                        f"fire not detected: {dets}")

    def test_spill_is_detected(self):
        dets = self.hev.detect(_spill_frame())
        self.assertTrue(any(d.class_name == "spill" for d in dets),
                        f"spill not detected: {dets}")

    def test_uniform_sky_does_not_trigger_smoke(self):
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        dets = self.hev.detect(frame)
        self.assertFalse(any(d.class_name == "smoke_fire" for d in dets),
                         "однотонное небо не должно давать ложный дым")


# ----------------------------------------------------------------------
# REST API и живучесть сервиса
# ----------------------------------------------------------------------
class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg, cls.tmp = _tmp_config()
        cfg.model.backend = "mock"
        cfg.video.source_type = "synthetic"
        cfg.video.loop = True
        cls.cfg = cfg
        cls.app = create_app(cfg)

    def test_full_api_cycle(self):
        from fastapi.testclient import TestClient

        with TestClient(self.app) as client:
            self.assertEqual(client.get("/health").json()["status"], "ok")
            self.assertEqual(client.get("/").status_code, 200)
            self.assertEqual(client.get("/docs").status_code, 200)
            self.assertEqual(client.get("/openapi.json").status_code, 200)

            # ждём первое событие от mock-детектора
            events = []
            deadline = time.time() + 20
            while time.time() < deadline:
                events = client.get("/events?limit=5").json()
                if events:
                    break
                time.sleep(0.2)
            self.assertTrue(events, "за отведённое время события не появились")

            event = events[0]
            required = ("id", "timestamp", "event_type", "object_class", "confidence",
                        "bbox", "frame_path", "camera_id", "severity", "status")
            for key in required:
                self.assertIn(key, event)

            self.assertEqual(client.get(f"/events/{event['id']}").status_code, 200)
            self.assertEqual(client.get("/events/evt_999999").status_code, 404)
            self.assertEqual(client.get("/frame/latest").status_code, 200)

            status = client.get("/status").json()
            self.assertEqual(status["state"], "running")
            self.assertEqual(status["model"]["backend"], "mock")

            metrics = client.get("/metrics").json()
            self.assertGreater(metrics["frames_processed"], 0)

            self.assertEqual(client.post("/config", json={"confidence_threshold": 0.5}).status_code, 200)
            self.assertEqual(client.post("/stop").json()["state"], "stopped")
            self.assertEqual(client.get("/health").status_code, 200)


class TestServiceSurvivesBadSource(unittest.TestCase):
    def test_missing_video_does_not_kill_service(self):
        from fastapi.testclient import TestClient

        cfg, _ = _tmp_config()
        cfg.model.backend = "mock"
        cfg.video.source_type = "file"
        cfg.video.path = "media/__definitely_missing__.mp4"
        cfg.video.loop = False
        app = create_app(cfg)

        with TestClient(app) as client:
            state = ""
            deadline = time.time() + 15
            while time.time() < deadline:
                state = client.get("/status").json()["state"]
                if state == "error":
                    break
                time.sleep(0.2)
            self.assertEqual(state, "error")
            # Сервис продолжает отвечать несмотря на недоступный источник.
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/events").status_code, 200)


class TestStreamShutdown(unittest.TestCase):
    """Поток MJPEG должен завершаться при остановке сервиса (иначе Ctrl+C висит)."""

    def test_mjpeg_generator_ends_on_shutdown(self):
        import asyncio

        from app import shutdown
        from app.api.routes import mjpeg_stream

        async def scenario():
            shutdown.reset()
            generator = mjpeg_stream(lambda: b"\xff\xd8JPEG", stream_fps=1000)
            first = await generator.__anext__()
            self.assertTrue(first.startswith(b"--frame"))
            self.assertIn(b"\xff\xd8JPEG", first)
            shutdown.request_shutdown()
            # После сигнала остановки генератор обязан завершиться сам.
            with self.assertRaises(StopAsyncIteration):
                await generator.__anext__()

        try:
            asyncio.run(scenario())
        finally:
            shutdown.reset()


if __name__ == "__main__":
    unittest.main(verbosity=2)
