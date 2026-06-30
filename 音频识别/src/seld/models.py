"""w2v-SELD 数据模型 — Pydantic schemas."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ── 事件类型枚举 ──────────────────────────────────────────

class EventType(str, Enum):
    """DCASE 标准声音事件类别."""

    SPEECH = "speech"
    FALL = "fall"
    GLASS_BREAK = "glass_break"
    KNOCK = "knock"
    OBJECT_DROP = "object_drop"
    FOOTSTEP = "footstep"
    DOOR_SLAM = "door_slam"

    @classmethod
    def from_index(cls, index: int) -> "EventType":
        """从类别索引获取枚举值."""
        members = list(cls)
        if 0 <= index < len(members):
            return members[index]
        raise ValueError(f"Event class index {index} out of range (0-{len(members)-1})")

    @classmethod
    def labels(cls) -> List[str]:
        """返回所有事件标签列表."""
        return [m.value for m in cls]

    @classmethod
    def num_classes(cls) -> int:
        """返回事件类别总数."""
        return len(cls)


# ── DOA 向量 ──────────────────────────────────────────────

class DOAVector(BaseModel):
    """声源方位向量（笛卡尔坐标，单位向量）."""

    x: float = Field(..., description="X 分量 (前方为 +x)")
    y: float = Field(..., description="Y 分量 (左方为 +y)")
    z: float = Field(..., description="Z 分量 (上方为 +z)")

    @field_validator("x", "y", "z")
    @classmethod
    def check_range(cls, v: float) -> float:
        if v < -1.0 or v > 1.0:
            raise ValueError(f"DOA component must be in [-1, 1], got {v}")
        return v

    @property
    def azimuth(self) -> float:
        """水平方位角（弧度）。0=前方, pi/2=左方."""
        import math
        return math.atan2(self.y, self.x)

    @property
    def elevation(self) -> float:
        """仰角（弧度）。0=水平面, pi/2=正上方."""
        import math
        return math.asin(max(-1.0, min(1.0, self.z)))


# ── 检测结果 ──────────────────────────────────────────────

class DetectedEvent(BaseModel):
    """单个声音事件检测结果."""

    event: EventType = Field(..., description="事件类型")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="置信度 (0~1)",
    )
    t_start_ms: int = Field(..., ge=0, description="起始时间（毫秒）")
    t_end_ms: int = Field(..., ge=0, description="结束时间（毫秒）")
    doa: Optional[DOAVector] = Field(
        default=None,
        description="声源方位向量（多通道模型支持时可用）",
    )

    @field_validator("t_end_ms")
    @classmethod
    def end_after_start(cls, v: int, info) -> int:
        if "t_start_ms" in info.data and v < info.data["t_start_ms"]:
            raise ValueError(
                f"t_end_ms ({v}) must be >= t_start_ms ({info.data['t_start_ms']})"
            )
        return v

    @property
    def duration_ms(self) -> int:
        """事件持续时长（毫秒）."""
        return self.t_end_ms - self.t_start_ms


class DetectionResult(BaseModel):
    """单次推理的完整检测结果."""

    events: List[DetectedEvent] = Field(
        default_factory=list,
        description="检测到的事件列表",
    )
    inference_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="推理耗时（毫秒）",
    )
    audio_duration_ms: int = Field(
        default=0,
        ge=0,
        description="输入音频时长（毫秒）",
    )

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def has_critical_event(self) -> bool:
        """是否包含关键事件（跌倒、玻璃破碎等需要立即响应的事件）."""
        critical = {EventType.FALL, EventType.GLASS_BREAK}
        return any(e.event in critical for e in self.events)


# ── 健康检查 ──────────────────────────────────────────────

class HealthStatus(BaseModel):
    """服务健康状态."""

    status: str = Field(default="ok", description="服务状态: ok | degraded | error")
    model_loaded: bool = Field(..., description="模型是否成功加载")
    device: str = Field(default="cpu", description="推理设备")
    version: str = Field(default="0.1.0", description="模块版本")
    uptime_seconds: float = Field(default=0.0, description="服务运行时长（秒）")
    model_name: Optional[str] = Field(default=None, description="已加载的模型名称")
