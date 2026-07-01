"""音频感知输出 — 标准化数据模型.

本模块定义了听觉子系统的标准输出格式，供融合层和视觉子系统消费。
每个时间帧产生一个 AudioPerception 对象，包含事件、情绪、方位、活跃度。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from seld.models import DetectedEvent, DOAVector


@dataclass
class AudioPerception:
    """
    一帧音频感知结果（标准输出契约）。

    每次 SELD 推理 + 情绪推理完成后，包装为此结构。
    下游融合层按同样的 timestamp 与视觉感知对齐。
    """

    timestamp: float
    """Unix 时间戳（秒），由 time.time() 生成."""

    events: List[DetectedEvent] = field(default_factory=list)
    """检测到的声音事件列表（SELD 输出）."""

    emotion: Optional[str] = None
    """主导情绪标签（emotion2vec 输出），如 happy/sad/angry/neutral."""

    emotion_confidence: float = 0.0
    """情绪识别的置信度."""

    doa: Optional[DOAVector] = None
    """声源方位向量（多通道模型可用时）."""

    activity_level: float = 0.0
    """整体声音活跃度 0~1。0=完全安静，1=非常嘈杂."""

    speech_detected: bool = False
    """是否检测到语音."""

    # ── 便捷属性 ──

    @property
    def has_critical_event(self) -> bool:
        """是否包含关键事件（跌倒/玻璃破碎等）."""
        from seld.models import EventType

        critical = {EventType.FALL, EventType.GLASS_BREAK, EventType.DOOR_SLAM}
        return any(e.event in critical for e in self.events)

    @property
    def is_silent(self) -> bool:
        """是否静音帧."""
        return len(self.events) == 0 and self.activity_level < 0.01

    @property
    def dominant_event(self) -> Optional[str]:
        """置信度最高的事件类型."""
        if not self.events:
            return None
        return max(self.events, key=lambda e: e.confidence).event.value

    def summary(self) -> str:
        """单行摘要."""
        parts = []
        if self.events:
            parts.append(f"音频: {', '.join(e.event.value for e in self.events[:3])}")
        if self.emotion:
            parts.append(f"情绪: {self.emotion}")
        if self.speech_detected:
            parts.append("语音 ✓")
        if self.is_silent:
            parts.append("静音")
        return " | ".join(parts) if parts else "空帧"


@dataclass
class VisualPerception:
    """
    一帧视觉感知结果（标准输入契约）。

    由视觉子系统生成，与 AudioPerception 通过 timestamp 对齐。
    当前作为接口定义，实际实现由视觉团队填充。
    """

    timestamp: float
    """Unix 时间戳."""

    person_detected: bool = False
    """是否检测到人."""

    person_bbox: Optional[tuple] = None
    """人物边界框 (x1, y1, x2, y2)，像素坐标."""

    pose: Optional[str] = None
    """人体姿态: standing / sitting / lying / walking / unknown."""

    face_emotion: Optional[str] = None
    """面部表情情绪: happy / sad / angry / fear / neutral."""

    face_emotion_confidence: float = 0.0

    activity_label: Optional[str] = None
    """活动识别标签: cooking / watching_tv / sleeping / exercising."""

    @property
    def is_present(self) -> bool:
        return self.person_detected

    @property
    def is_prone(self) -> bool:
        """是否处于倒地/躺卧状态（跌倒关键信号）."""
        return self.pose == "lying"

    def summary(self) -> str:
        parts = []
        if self.person_detected:
            parts.append(f"姿态: {self.pose or 'unknown'}")
        if self.face_emotion:
            parts.append(f"表情: {self.face_emotion}")
        return " | ".join(parts) if parts else "未检测到人"
