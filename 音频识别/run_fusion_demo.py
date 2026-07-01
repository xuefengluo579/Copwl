"""
多模态融合演示 — 音频推理 + 模拟视觉 + 融合决策。

运行:
    python run_fusion_demo.py

场景模拟:
    模拟一位老人在 5 个时间段的活动，展示融合层如何
    结合听觉和视觉信号做出不同的决策。
"""

from __future__ import annotations

import time
import random
from typing import Optional

from seld.perception import AudioPerception, VisualPerception
from seld.models import DetectedEvent, DOAVector, EventType
from seld.fusion import MultiModalFusion, FusionResult, AlertLevel


# ── 模拟数据 ──────────────────────────────────────────────

def make_audio(
    events: list,
    emotion: str = "neutral",
    activity: float = 0.3,
    speech: bool = False,
    doa_x: float = 0.0,
    doa_y: float = 0.0,
    doa_z: float = 0.0,
) -> AudioPerception:
    """构造模拟音频感知."""
    detected = []
    for event_type, conf in events:
        detected.append(DetectedEvent(
            event=EventType(event_type),
            confidence=conf,
            t_start_ms=0,
            t_end_ms=100,
            doa=DOAVector(x=doa_x, y=doa_y, z=doa_z),
        ))
    return AudioPerception(
        timestamp=time.time(),
        events=detected,
        emotion=emotion,
        emotion_confidence=0.8,
        doa=DOAVector(x=doa_x, y=doa_y, z=doa_z),
        activity_level=activity,
        speech_detected=speech,
    )


def make_visual(
    detected: bool = True,
    pose: str = "standing",
    face: str = "neutral",
    bbox: Optional[tuple] = None,
) -> VisualPerception:
    """构造模拟视觉感知."""
    if bbox is None and detected:
        bbox = (800, 100, 1120, 900)  # 画面中央
    return VisualPerception(
        timestamp=time.time(),
        person_detected=detected,
        pose=pose,
        face_emotion=face,
        face_emotion_confidence=0.85,
        person_bbox=bbox,
    )


# ── 主演示 ────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" 多模态融合演示 — 听觉 + 视觉交叉验证")
    print("=" * 60)

    fusion = MultiModalFusion()

    # 场景列表: (描述, 音频, 视觉)
    scenarios = [
        (
            "🟢 场景1: 正常活动 — 在厨房做饭",
            make_audio([("speech", 0.5), ("object_drop", 0.4)], emotion="neutral", activity=0.6, speech=True),
            make_visual(pose="standing", face="neutral"),
        ),
        (
            "🔴 场景2: 紧急情况 — 浴室摔倒 (双信号)",
            make_audio([("fall", 0.92)], emotion="fear", activity=0.9, doa_x=0.3, doa_y=-0.1, doa_z=0.05),
            make_visual(pose="lying", face="fear"),
        ),
        (
            "🟡 场景3: 仅听觉异常 — 听到巨响但人站着 (可能东西掉了)",
            make_audio([("glass_break", 0.88)], emotion="neutral", activity=0.7),
            make_visual(pose="standing", face="neutral"),
        ),
        (
            "🟠 场景4: 仅视觉异常 — 躺着但无异常声音 (可能在休息)",
            make_audio([], emotion="neutral", activity=0.05),
            make_visual(pose="lying", face="neutral"),
        ),
        (
            "🟠 场景5: 情绪异常 — 哭泣 + 悲伤表情",
            make_audio([("speech", 0.6)], emotion="sad", activity=0.4, speech=True),
            make_visual(pose="sitting", face="sad"),
        ),
        (
            "🔴 场景6: 不见人影 + 听到跌倒 (人去洗手间摔了)",
            make_audio([("fall", 0.85)], emotion="pain", activity=0.8, doa_x=-0.5, doa_y=0.3, doa_z=0.0),
            make_visual(detected=False),
        ),
    ]

    for desc, audio, visual in scenarios:
        result = fusion.fuse(audio, visual)
        print(f"\n{desc}")
        print(f"  音频: {audio.summary()}")
        print(f"  视觉: {visual.summary()}")
        print(f"  融合: {result.summary()}")
        if result.alert_reason:
            print(f"  原因: {result.alert_reason}")

    # ── 自适应融合演示 ──
    print(f"\n{'=' * 60}")
    print(" 自适应融合演示 — 时段调整 + Persona 驱动")
    print("=" * 60)

    adaptive = MultiModalFusion()
    from seld.fusion import AdaptiveFusion
    adapt = AdaptiveFusion()

    # 同一个场景，不同时段
    audio = make_audio([("fall", 0.85)], emotion="neutral", activity=0.7)
    visual = make_visual(pose="lying", face="neutral")

    for hour, label in [(3, "凌晨3点"), (10, "上午10点"), (22, "晚上10点")]:
        adapt.adjust_for_time(hour)
        result = adapt.fuse(audio, visual)
        print(f"\n  {label} — {label}: fall_threshold={adapt.fall_threshold:.2f}")
        print(f"  结果: {result.summary()}")

    print(f"\n{'=' * 60}")
    print(" 演示完成。展示了 6 种场景 × 3 个时段 = 18 种决策。")
    print("=" * 60)


if __name__ == "__main__":
    main()
