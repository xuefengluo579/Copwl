"""多模态融合层.

将听觉感知 (AudioPerception) 和视觉感知 (VisualPerception)
按时间戳对齐，交叉验证后输出融合置信度和紧急程度。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from seld.perception import AudioPerception, VisualPerception
from seld.logging_utils import get_logger

logger = get_logger(__name__)


# ── 紧急等级 ──────────────────────────────────────────────

class AlertLevel(Enum):
    NORMAL = "normal"    # 一切正常
    WATCH = "watch"      # 需要关注，不报警
    ALERT = "alert"      # 立即报警/通知家人
    EMERGENCY = "emergency"  # 最高优先级，直接呼叫急救


# ── 融合结果 ──────────────────────────────────────────────

@dataclass
class FusionResult:
    """单帧融合结果."""

    timestamp: float
    alert_level: AlertLevel = AlertLevel.NORMAL
    alert_reason: str = ""

    # 各维度的置信度
    fall_confidence: float = 0.0
    distress_confidence: float = 0.0
    anomaly_confidence: float = 0.0

    # 来源
    audio: Optional[AudioPerception] = None
    visual: Optional[VisualPerception] = None

    # 交叉验证标记
    audio_visual_agree: bool = False
    audio_visual_conflict: bool = False
    audio_only: bool = False
    visual_only: bool = False

    def summary(self) -> str:
        emoji = {"normal": "✅", "watch": "👀", "alert": "🚨", "emergency": "🆘"}
        return (
            f"{emoji[self.alert_level.value]} [{self.alert_level.value}] "
            f"fall={self.fall_confidence:.2f} "
            f"distress={self.distress_confidence:.2f} "
            f"| agree={self.audio_visual_agree} "
            f"conflict={self.audio_visual_conflict} "
            f"| {self.alert_reason}"
        )


# ── 融合矩阵 ──────────────────────────────────────────────
# (听觉有异常, 视觉有异常) → (fall_confidence, alert_level, reason)
# 异常定义: 听觉=检测到关键事件, 视觉=躺姿

FUSION_MATRIX = {
    # (audio_abnormal, visual_abnormal)
    (True, True): (0.90, AlertLevel.EMERGENCY, "听觉+视觉双信号确认跌倒"),
    (True, False): (0.40, AlertLevel.ALERT, "仅听觉检测到异常，视觉未见异常"),
    (False, True): (0.50, AlertLevel.ALERT, "仅视觉检测到躺姿，未听到异常声音"),
    (False, False): (0.0, AlertLevel.NORMAL, "一切正常"),
}


# ── 融合器 ────────────────────────────────────────────────

class MultiModalFusion:
    """
    多模态融合器。

    将听觉和视觉感知结果按时间戳对齐，通过交叉验证矩阵
    和加权评分计算融合置信度和紧急程度。

    Usage:
        fusion = MultiModalFusion()
        result = fusion.fuse(audio_perception, visual_perception)
        if result.alert_level in (AlertLevel.ALERT, AlertLevel.EMERGENCY):
            send_alert(result)
    """

    def __init__(
        self,
        audio_weight: float = 0.5,
        visual_weight: float = 0.5,
        fall_threshold: float = 0.6,
        buffer_size: int = 30,
    ):
        """
        Args:
            audio_weight: 听觉在融合中的权重 (0~1).
            visual_weight: 视觉在融合中的权重 (0~1).
            fall_threshold: 触发报警的跌倒置信度阈值.
            buffer_size: 历史缓冲大小（用于平滑）.
        """
        self.audio_weight = audio_weight
        self.visual_weight = visual_weight
        self.fall_threshold = fall_threshold

        self.audio_buffer: deque[AudioPerception] = deque(maxlen=buffer_size)
        self.visual_buffer: deque[VisualPerception] = deque(maxlen=buffer_size)
        self.result_buffer: deque[FusionResult] = deque(maxlen=buffer_size)

    def fuse(
        self,
        audio: Optional[AudioPerception] = None,
        visual: Optional[VisualPerception] = None,
    ) -> FusionResult:
        """
        融合一帧感知数据。

        至少需要提供一个感知源。另一个可选。
        如果只有一个源，会在标记中注明。

        Returns:
            FusionResult 包含融合置信度和紧急等级.
        """
        if audio is None and visual is None:
            return FusionResult(
                timestamp=0,
                alert_level=AlertLevel.NORMAL,
                alert_reason="无感知数据",
            )

        ts = (audio.timestamp if audio else visual.timestamp)  # type: ignore

        # 判断异常状态
        audio_abnormal = self._is_audio_abnormal(audio)
        visual_abnormal = self._is_visual_abnormal(visual)

        # 查融合矩阵
        base_conf, base_level, base_reason = FUSION_MATRIX[
            (audio_abnormal, visual_abnormal)
        ]
        fall_conf = base_conf

        # ── 加分项 ──
        reasons = [base_reason]

        # 情绪加成
        distress = self._compute_distress(audio, visual)
        if distress > 0.5:
            fall_conf = max(fall_conf, distress * 0.7)
            reasons.append(f"情绪 distress={distress:.2f}")

        # DOA + BBox 空间一致性
        if audio and visual and audio.doa and visual.person_bbox:
            consistency = self._spatial_consistency(audio, visual)
            if consistency > 0.5:
                fall_conf += 0.1
                reasons.append(f"空间一致 +0.1")

        # 历史平滑：取最近 N 帧的平均
        self.result_buffer.append(FusionResult(
            timestamp=ts,
            fall_confidence=fall_conf,
            alert_level=base_level,
            alert_reason="; ".join(reasons),
        ))
        smoothed_conf = np.mean([r.fall_confidence for r in self.result_buffer])

        # 确定最终等级
        if smoothed_conf > 0.85:
            level = AlertLevel.EMERGENCY
        elif smoothed_conf > self.fall_threshold:
            level = AlertLevel.ALERT
        elif distress > 0.4 or audio_abnormal or visual_abnormal:
            level = AlertLevel.WATCH
        else:
            level = AlertLevel.NORMAL

        # 交叉验证标记
        agree = audio_abnormal and visual_abnormal
        conflict = (audio_abnormal and not visual_abnormal) or (
            not audio_abnormal and visual_abnormal
        )
        audio_only = audio is not None and visual is None
        visual_only = visual is not None and audio is None

        result = FusionResult(
            timestamp=ts,
            alert_level=level,
            alert_reason="; ".join(reasons),
            fall_confidence=round(smoothed_conf, 3),
            distress_confidence=round(distress, 3),
            anomaly_confidence=round(max(
                float(audio_abnormal), float(visual_abnormal)
            ), 3),
            audio=audio,
            visual=visual,
            audio_visual_agree=agree,
            audio_visual_conflict=conflict,
            audio_only=audio_only,
            visual_only=visual_only,
        )

        self.result_buffer.append(result)
        return result

    # ── 内部方法 ──

    def _is_audio_abnormal(self, audio: Optional[AudioPerception]) -> bool:
        if audio is None:
            return False
        return audio.has_critical_event

    def _is_visual_abnormal(self, visual: Optional[VisualPerception]) -> bool:
        if visual is None:
            return False
        return visual.is_prone

    def _compute_distress(
        self,
        audio: Optional[AudioPerception],
        visual: Optional[VisualPerception],
    ) -> float:
        """综合情绪/表情计算痛苦程度 0~1."""
        score = 0.0
        count = 0

        negative_emotions = {"angry", "fear", "sad", "pain"}

        if audio and audio.emotion:
            count += 1
            if audio.emotion in negative_emotions:
                score += 0.7
            elif audio.emotion == "neutral":
                score += 0.1

        if visual and visual.face_emotion:
            count += 1
            if visual.face_emotion in negative_emotions:
                score += 0.8

        return score / count if count > 0 else 0.0

    def _spatial_consistency(
        self, audio: AudioPerception, visual: VisualPerception
    ) -> float:
        """
        声源方向和人物位置的粗略空间一致性检查。

        实际系统需要使用相机内参 + 麦克风阵列几何做精确映射。
        这里做简化：检查 DOA 方向是否能匹配 bbox 位置.
        """
        if audio.doa is None or visual.person_bbox is None:
            return 0.0

        # 简化：DOA azimuth 和 bbox 水平位置的相关性
        doa_azimuth = audio.doa.azimuth  # -π ~ +π, 0=正前方
        bbox_center_x = (visual.person_bbox[0] + visual.person_bbox[2]) / 2
        # 假设画面宽度 1920，归一化 bbox 中心到 [-1, 1]
        bbox_norm = (bbox_center_x / 1920) * 2 - 1
        # 简化的方位匹配
        diff = abs(doa_azimuth / np.pi - bbox_norm)
        return max(0.0, 1.0 - diff)


# ── 自适应阈值（记忆驱动的融合） ───────────────────────────

class AdaptiveFusion(MultiModalFusion):
    """
    带记忆能力的自适应融合器。

    在 MultiModalFusion 基础上，根据历史 Persona 数据
    动态调整融合权重和报警阈值。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.time_of_day_factors = self._default_time_factors()

    @staticmethod
    def _default_time_factors() -> dict:
        """夜间更敏感，凌晨最敏感."""
        return {
            (0, 6): 1.5,   # 凌晨: 敏感度 ×1.5
            (6, 8): 1.2,   # 清晨: ×1.2
            (8, 20): 1.0,  # 白天: 正常
            (20, 22): 1.2, # 晚间: ×1.2
            (22, 24): 1.5, # 深夜: ×1.5
        }

    def adjust_for_time(self, hour: int) -> None:
        """根据当前时段调整阈值."""
        for (start, end), factor in self.time_of_day_factors.items():
            if start <= hour < end:
                self.fall_threshold = 0.6 / factor
                return

    def adjust_for_persona(self, persona_active_hour: float) -> None:
        """根据用户画像调整——平时活跃的时段降低敏感度."""
        if persona_active_hour > 0.5:
            self.fall_threshold = 0.7  # 活跃时段，减少误报
        else:
            self.fall_threshold = 0.5  # 安静时段，提高敏感度
