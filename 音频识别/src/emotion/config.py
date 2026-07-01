"""EmotionEngine 配置 — dataclass + 环境变量覆盖."""

from __future__ import annotations

import os
from dataclasses import dataclass

from seld.exceptions import ConfigurationError


@dataclass
class EmotionConfig:
    """Emotion2vec 情绪识别引擎配置."""

    # 魔搭模型标识
    model_name: str = "iic/emotion2vec_base"
    model_revision: str = "v1.0.0"

    # 设备
    device: str = "auto"  # "cuda" | "cpu" | "auto"

    # 推理参数
    confidence_threshold: float = 0.3
    sample_rate: int = 16000

    # 降级策略
    allow_cpu_fallback: bool = True

    def validate(self) -> None:
        """验证配置合法性."""
        if self.device not in ("cuda", "cpu", "auto"):
            raise ConfigurationError(
                f"device must be 'cuda', 'cpu', or 'auto', got '{self.device}'"
            )
        if not (0.0 < self.confidence_threshold <= 1.0):
            raise ConfigurationError(
                f"confidence_threshold must be in (0, 1], "
                f"got {self.confidence_threshold}"
            )

    @classmethod
    def from_env(cls) -> "EmotionConfig":
        """从环境变量构建配置."""
        config = cls()
        env_map = {
            "EMOTION_MODEL_NAME": ("model_name", str),
            "EMOTION_DEVICE": ("device", str),
            "EMOTION_CONFIDENCE_THRESHOLD": ("confidence_threshold", float),
            "EMOTION_ALLOW_CPU_FALLBACK": ("allow_cpu_fallback",
                                           lambda v: v.lower() == "true"),
        }
        for env_var, (field, cast) in env_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                setattr(config, field, cast(val))
        config.validate()
        return config
