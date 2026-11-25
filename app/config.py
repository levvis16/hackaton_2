"""Конфигурация CV-модуля.

Загрузка из YAML-файла + переопределение отдельными ключами.
Все относительные пути (модель, видео, лог, каталог событий) разрешаются
относительно каталога конфигурационного файла, что упрощает перенос на плату.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict

import yaml


@dataclass
class AppSection:
    host: str = "0.0.0.0"
    port: int = 8000
    mode: str = "demo"            # demo | edge
    camera_id: str = "cam_01"
    autostart: bool = True        # запускать обработку при старте сервиса


@dataclass
class VideoSection:
    source_type: str = "file"     # file | camera | rtsp | image | synthetic
    path: str = "media/demo_person_bicycle_car.mp4"
    camera_index: int = 0
    rtsp_url: str = ""
    loop: bool = True             # зацикливать видеофайл
    target_fps: float = 0.0       # 0 -> без ограничения
    process_width: int = 640      # ширина кадра для предобработки (0 -> как есть)
    frame_skip: int = 0           # 0 -> обрабатывать каждый кадр
    reconnect_attempts: int = 5
    reconnect_delay_sec: float = 2.0


@dataclass
class ModelSection:
    backend: str = "auto"         # auto | yolo_onnx | heuristics | mock
    path: str = "models/yolov8n.onnx"
    labels: str = "models/coco.names"
    input_size: int = 640
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    # Число потоков intra-op для ONNX Runtime. 0 = по умолчанию (все ядра).
    # На ноутбуках ограничение числа потоков (например, 4) даёт более стабильный FPS.
    intra_op_num_threads: int = 0
    providers: list = field(default_factory=lambda: ["CPUExecutionProvider"])


@dataclass
class HeuristicsSection:
    smoke_fire: bool = True
    spill: bool = True
    helmet_check: bool = True     # эвристика "человек без каски"
    obstacle: bool = True         # препятствие в зоне перед камерой
    # Зона перед камерой в нормализованных координатах [x1, y1, x2, y2].
    obstacle_zone: list = field(default_factory=lambda: [0.2, 0.6, 0.8, 1.0])
    # Порог площади (доля кадра) для детекции дыма/огня и разлива.
    min_area_ratio: float = 0.015
    # Сколько последовательных кадров подтверждают событие (антидребезг).
    confirm_frames: int = 3


@dataclass
class EventsSection:
    save_frames: bool = True
    min_confidence: float = 0.35
    cooldown_sec: float = 4.0     # не дублировать событие по тому же объекту чаще
    frames_dir: str = "events/frames"
    store_path: str = "events/events.jsonl"
    max_memory: int = 5000        # сколько событий держать в памяти


@dataclass
class LoggingSection:
    level: str = "INFO"
    file: str = "logs/app.log"
    console: bool = True


@dataclass
class Config:
    app: AppSection = field(default_factory=AppSection)
    video: VideoSection = field(default_factory=VideoSection)
    model: ModelSection = field(default_factory=ModelSection)
    heuristics: HeuristicsSection = field(default_factory=HeuristicsSection)
    events: EventsSection = field(default_factory=EventsSection)
    logging: LoggingSection = field(default_factory=LoggingSection)
    # Каталог, относительно которого разрешаются относительные пути.
    base_dir: str = field(default_factory=os.getcwd)

    # ----- вспомогательные разрешения путей -----
    def resolve(self, path: str) -> str:
        if not path:
            return path
        if os.path.isabs(path):
            return os.path.normpath(path)
        return os.path.normpath(os.path.join(self.base_dir, path))

    @property
    def model_path(self) -> str:
        return self.resolve(self.model.path)

    @property
    def labels_path(self) -> str:
        return self.resolve(self.model.labels)

    @property
    def video_path(self) -> str:
        return self.resolve(self.video.path)

    @property
    def frames_dir(self) -> str:
        return self.resolve(self.events.frames_dir)

    @property
    def store_path(self) -> str:
        return self.resolve(self.events.store_path)

    @property
    def log_path(self) -> str:
        return self.resolve(self.logging.file)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            if is_dataclass(value) and not isinstance(value, type):
                result[f.name] = {
                    inner.name: getattr(value, inner.name) for inner in fields(value)
                }
            else:
                result[f.name] = value
        return result


def _apply(section: Any, data: Dict[str, Any]) -> None:
    """Аккуратно переносит значения словаря в dataclass-секцию."""
    if not isinstance(data, dict):
        return
    valid = {f.name for f in fields(section)}
    for key, value in data.items():
        if key in valid:
            setattr(section, key, value)


def load_config(path: str | None = None) -> Config:
    """Загружает конфигурацию из YAML и подставляет значения по умолчанию.

    Путь определяется аргументом, переменной окружения CV_CONFIG
    или файлом config.yaml в текущем каталоге.
    """
    cfg = Config()
    path = path or os.environ.get("CV_CONFIG", "config.yaml")
    path = os.path.abspath(path)

    if os.path.exists(path):
        cfg.base_dir = os.path.dirname(path)
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if isinstance(raw, dict):
            for section_name in ("app", "video", "model", "heuristics", "events", "logging"):
                if section_name in raw:
                    _apply(getattr(cfg, section_name), raw[section_name] or {})
    else:
        cfg.base_dir = os.getcwd()

    # Часть параметров удобно переопределять переменными окружения.
    env_map = {
        "CV_SOURCE_TYPE": ("video", "source_type", str),
        "CV_SOURCE": ("video", "path", str),
        "CV_CAMERA_INDEX": ("video", "camera_index", int),
        "CV_RTSP_URL": ("video", "rtsp_url", str),
        "CV_FRAME_SKIP": ("video", "frame_skip", int),
        "CV_TARGET_FPS": ("video", "target_fps", float),
        "CV_CONF": ("model", "confidence_threshold", float),
        "CV_MIN_CONF": ("events", "min_confidence", float),
        "CV_BACKEND": ("model", "backend", str),
        "CV_THREADS": ("model", "intra_op_num_threads", int),
        "CV_MODE": ("app", "mode", str),
        "CV_AUTOSTART": ("app", "autostart", lambda v: str(v).strip().lower() in ("1", "true", "yes", "on")),
        "CV_PORT": ("app", "port", int),
        "CV_HOST": ("app", "host", str),
    }
    for env_key, (sec, attr, caster) in env_map.items():
        if env_key in os.environ:
            try:
                setattr(getattr(cfg, sec), attr, caster(os.environ[env_key]))
            except (TypeError, ValueError):
                pass

    return cfg
