"""Emotion2vec 情绪类别 — 枚举与映射."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List


class EmotionType(str, Enum):
    """语音情绪类别（与 emotion2vec 输出对齐）."""

    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    NEUTRAL = "neutral"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    PAIN = "pain"

    @property
    def cn_label(self) -> str:
        """中文标签."""
        return self._cn_map()[self.value]

    @classmethod
    def from_str(cls, label: str) -> "EmotionType":
        """从字符串解析枚举值（不区分大小写，支持中文输入）."""
        norm = label.strip().lower()
        # 先查英文
        for e in cls:
            if e.value == norm:
                return e
        # 再查中文
        cn_to_en = {v: k for k, v in cls._cn_map().items()}
        if norm in cn_to_en:
            return cls(cn_to_en[norm])
        raise ValueError(f"Unknown emotion label: {label}")

    @classmethod
    def labels(cls) -> List[str]:
        """所有英文标签列表."""
        return [m.value for m in cls]

    @classmethod
    def cn_labels(cls) -> List[str]:
        """所有中文标签列表."""
        return list(cls._cn_map().values())

    @classmethod
    def num_classes(cls) -> int:
        """类别总数."""
        return len(cls)

    @staticmethod
    def _cn_map() -> Dict[str, str]:
        """中英文映射表."""
        return {
            "happy": "开心",
            "sad": "悲伤",
            "angry": "生气",
            "neutral": "中性",
            "fear": "恐惧",
            "surprise": "惊讶",
            "disgust": "厌恶",
            "pain": "痛苦",
        }

    @classmethod
    def distress_weight(cls, label: str) -> float:
        """情绪 → 痛苦/危险权重 (0~1)，供融合层使用."""
        weights = {
            "happy": -0.2,
            "neutral": 0.0,
            "tired": 0.2,
            "surprise": 0.3,
            "disgust": 0.4,
            "angry": 0.6,
            "sad": 0.6,
            "fear": 0.8,
            "pain": 0.95,
        }
        return weights.get(label, 0.0)

    @classmethod
    def negative_set(cls) -> set:
        """高危情绪集合（触发报警的情绪）."""
        return {"angry", "fear", "sad", "pain"}
