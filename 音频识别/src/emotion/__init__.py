"""EmotionEngine — 基于 emotion2vec 的语音情绪识别模块.

Usage:
    from emotion import EmotionEngine, EmotionConfig

    config = EmotionConfig()
    engine = EmotionEngine(config)
    emotion_label, confidence = engine.infer(audio_array)
"""

__version__ = "0.1.0"

from emotion.config import EmotionConfig
from emotion.models import EmotionType
from emotion.engine import EmotionEngine, get_emotion_engine

__all__ = [
    "__version__",
    "EmotionConfig",
    "EmotionType",
    "EmotionEngine",
    "get_emotion_engine",
]
