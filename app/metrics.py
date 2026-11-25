"""Метрики производительности: FPS, задержка обработки кадра (latency), счётчики.

FPS считается по скользящему окну времени, latency — как EMA/среднее/p95
по последним обработанным кадрам. Все обращения потокобезопасны.
"""
from __future__ import annotations

import statistics
import threading
import time
from collections import Counter, deque
from typing import Deque, Dict, List, Tuple


class Metrics:
    def __init__(self, window_sec: float = 10.0, latency_buffer: int = 200) -> None:
        self._lock = threading.Lock()
        self._window = window_sec
        self._frames: Deque[float] = deque()          # метки времени обработанных кадров
        self._latencies: Deque[float] = deque(maxlen=latency_buffer)
        self._latency_ema: float = 0.0
        self.frames_processed: int = 0
        self.frames_dropped: int = 0
        self.events_created: int = 0
        self.errors: int = 0
        self.per_class_events: Counter = Counter()
        self._start = time.time()
        self._last_frame_ts: float | None = None

    # ----- запись -----
    def on_frame(self, latency_ms: float) -> None:
        now = time.monotonic()
        with self._lock:
            self._frames.append(now)
            self._latencies.append(latency_ms)
            self._latency_ema = (
                latency_ms if self._latency_ema == 0.0
                else 0.85 * self._latency_ema + 0.15 * latency_ms
            )
            self.frames_processed += 1
            self._last_frame_ts = now
            self._trim(now)

    def on_drop(self, count: int = 1) -> None:
        with self._lock:
            self.frames_dropped += count

    def on_events(self, count: int, classes: List[str] | None = None) -> None:
        with self._lock:
            self.events_created += count
            for cls in classes or []:
                self.per_class_events[cls] += 1

    def on_error(self) -> None:
        with self._lock:
            self.errors += 1

    def _trim(self, now: float) -> None:
        cutoff = now - self._window
        frames = self._frames
        while frames and frames[0] < cutoff:
            frames.popleft()

    # ----- чтение -----
    def _fps_locked(self, now: float) -> float:
        if len(self._frames) < 2:
            return 0.0
        span = now - self._frames[0]
        if span <= 0:
            return 0.0
        return round((len(self._frames) - 1) / span, 2)

    def fps(self) -> float:
        now = time.monotonic()
        with self._lock:
            self._trim(now)
            return self._fps_locked(now)

    def latency_ms(self) -> float:
        with self._lock:
            return round(self._latency_ema, 2)

    def snapshot(self) -> Dict:
        now = time.monotonic()
        with self._lock:
            self._trim(now)
            latencies = list(self._latencies)
            avg = round(statistics.fmean(latencies), 2) if latencies else 0.0
            p95 = round(self._p95(latencies), 2) if latencies else 0.0
            uptime = max(1e-6, time.time() - self._start)
            return {
                "fps": self._fps_locked(now),
                "fps_avg": round(self.frames_processed / uptime, 2),
                "latency_ms": round(self._latency_ema, 2),
                "latency_avg_ms": avg,
                "latency_p95_ms": p95,
                "frames_processed": self.frames_processed,
                "frames_dropped": self.frames_dropped,
                "events_created": self.events_created,
                "errors": self.errors,
                "uptime_sec": round(time.time() - self._start, 1),
                "per_class_events": dict(self.per_class_events),
                "latency_history_ms": [round(x, 2) for x in latencies[-60:]],
            }

    @staticmethod
    def _p95(values: List[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = max(0, int(round(0.95 * (len(ordered) - 1))))
        return ordered[idx]
