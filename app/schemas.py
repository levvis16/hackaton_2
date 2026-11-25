"""Pydantic-схемы API: события, статус, метрики, запросы управления."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x: int = Field(..., description="Левый верхний угол, X")
    y: int = Field(..., description="Левый верхний угол, Y")
    width: int = Field(..., description="Ширина рамки")
    height: int = Field(..., description="Высота рамки")


class Event(BaseModel):
    id: str
    timestamp: str
    event_type: str
    object_class: str
    confidence: float
    bbox: BBox
    frame_path: Optional[str] = None
    camera_id: str = "cam_01"
    severity: str = "medium"
    status: str = "new"
    # Дополнительные поля (не обязательны по ТЗ, но полезны оператору):
    raw_class: Optional[str] = None
    detector: Optional[str] = None
    source: Optional[str] = None
    track_id: Optional[int] = None


class StatusResponse(BaseModel):
    state: str = Field(..., description="running | stopped | starting | finished | error")
    mode: str
    source_type: str
    source: str
    camera_id: str
    fps: float
    latency_ms: float
    frames_processed: int
    events_created: int
    uptime_sec: float
    error: Optional[str] = None
    model: Dict[str, Any] = Field(default_factory=dict)
    source_meta: Dict[str, Any] = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    fps: float
    fps_avg: float = 0.0
    latency_ms: float
    latency_avg_ms: float
    latency_p95_ms: float
    frames_processed: int
    frames_dropped: int
    events_created: int
    errors: int
    uptime_sec: float
    per_class_events: Dict[str, int] = Field(default_factory=dict)
    latency_history_ms: List[float] = Field(default_factory=list)


class StartRequest(BaseModel):
    source_type: Optional[str] = Field(
        default=None, description="file | camera | rtsp | image | synthetic"
    )
    path: Optional[str] = None
    camera_index: Optional[int] = None
    rtsp_url: Optional[str] = None


class ConfigRequest(BaseModel):
    confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    source_type: Optional[str] = None
    path: Optional[str] = None
    camera_index: Optional[int] = None
    rtsp_url: Optional[str] = None
    frame_skip: Optional[int] = Field(default=None, ge=0, le=30)
    target_fps: Optional[float] = Field(default=None, ge=0.0)


class ActionResponse(BaseModel):
    ok: bool
    message: str
    state: Optional[str] = None
    status: Optional[StatusResponse] = None
