"""w2v-SELD 配置管理 — dataclass + YAML + 环境变量三层覆盖."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from seld.exceptions import ConfigurationError


# ── 配置数据类 ────────────────────────────────────────────

@dataclass
class ModelConfig:
    """模型相关配置."""

    model_path: str = "./checkpoints/w2v_seld_base"
    checkpoint_path: str = "./checkpoints/w2v_seld_base.pt"
    device: str = "auto"  # "cuda" | "cpu" | "auto"
    use_fairseq: bool = True  # True=fairseq API, False=direct state_dict


@dataclass
class AudioConfig:
    """音频输入配置."""

    sample_rate: int = 16000
    num_channels: int = 4
    frame_duration_ms: int = 100  # 模型输出的帧间隔（毫秒）


@dataclass
class InferenceConfig:
    """推理参数配置."""

    event_threshold: float = 0.5  # SED 激活阈值 (0~1)
    min_event_duration_ms: int = 100  # 过滤短于此的事件，减少瞬态误检
    batch_size: int = 1


@dataclass
class StreamingConfig:
    """流式推理配置."""

    window_seconds: float = 2.0  # 滑动窗口长度（秒）
    hop_seconds: float = 0.5  # 窗口步进（秒）


@dataclass
class SELDConfig:
    """w2v-SELD 总配置."""

    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)

    # ── 工厂方法 ──────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SELDConfig":
        """
        从 YAML 文件加载配置，并用环境变量覆盖。

        环境变量映射（前缀 SELD_）:
            SELD_MODEL_PATH          → model.model_path
            SELD_CHECKPOINT_PATH     → model.checkpoint_path
            SELD_DEVICE              → model.device
            SELD_SAMPLE_RATE         → audio.sample_rate
            SELD_EVENT_THRESHOLD     → inference.event_threshold

        Args:
            path: YAML 配置文件路径。

        Returns:
            合并后的 SELDConfig 实例。
        """
        config = cls()

        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            config = cls._merge_dict(config, data)

        # 环境变量覆盖
        config = cls._apply_env_overrides(config)
        config.validate()
        return config

    @classmethod
    def from_env(cls) -> "SELDConfig":
        """仅从环境变量构建配置（使用默认值作为基线）."""
        config = cls()
        config = cls._apply_env_overrides(config)
        config.validate()
        return config

    def validate(self) -> None:
        """验证配置值合法性，不合法的抛出 ConfigurationError."""
        if self.model.device not in ("cuda", "cpu", "auto"):
            raise ConfigurationError(
                f"device must be 'cuda', 'cpu', or 'auto', got '{self.model.device}'"
            )
        if self.audio.sample_rate not in (8000, 16000, 22050, 44100, 48000):
            raise ConfigurationError(
                f"Unsupported sample_rate: {self.audio.sample_rate}"
            )
        if self.audio.num_channels < 1:
            raise ConfigurationError(
                f"num_channels must be >= 1, got {self.audio.num_channels}"
            )
        if not (0.0 < self.inference.event_threshold <= 1.0):
            raise ConfigurationError(
                f"event_threshold must be in (0, 1], got {self.inference.event_threshold}"
            )
        if self.inference.min_event_duration_ms < 0:
            raise ConfigurationError(
                f"min_event_duration_ms must be >= 0, got {self.inference.min_event_duration_ms}"
            )

    def to_dict(self) -> dict[str, Any]:
        """将配置序列化为字典."""
        result: dict[str, Any] = {}
        for f in fields(self):
            sub = getattr(self, f.name)
            result[f.name] = (
                {sf.name: getattr(sub, sf.name) for sf in fields(sub)}
                if hasattr(sub, "__dataclass_fields__")
                else sub
            )
        return result

    def to_yaml(self, path: str | Path) -> None:
        """将当前配置写入 YAML 文件."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False)

    # ── 内部方法 ──────────────────────────────────────────

    @staticmethod
    def _merge_dict(config: SELDConfig, data: dict) -> SELDConfig:
        """递归合并字典到 dataclass."""
        for f in fields(config):
            if f.name in data:
                sub = data[f.name]
                if isinstance(sub, dict):
                    setattr(config, f.name, type(getattr(config, f.name))(**sub))
                else:
                    setattr(config, f.name, sub)
        return config

    @staticmethod
    def _apply_env_overrides(config: SELDConfig) -> SELDConfig:
        """用 SELD_ 前缀的环境变量覆盖配置."""
        env_map = {
            "SELD_MODEL_PATH": ("model", "model_path", str),
            "SELD_CHECKPOINT_PATH": ("model", "checkpoint_path", str),
            "SELD_DEVICE": ("model", "device", str),
            "SELD_SAMPLE_RATE": ("audio", "sample_rate", int),
            "SELD_NUM_CHANNELS": ("audio", "num_channels", int),
            "SELD_EVENT_THRESHOLD": ("inference", "event_threshold", float),
            "SELD_MIN_EVENT_DURATION_MS": ("inference", "min_event_duration_ms", int),
            "SELD_BATCH_SIZE": ("inference", "batch_size", int),
            "SELD_WINDOW_SECONDS": ("streaming", "window_seconds", float),
            "SELD_HOP_SECONDS": ("streaming", "hop_seconds", float),
        }
        for env_var, (section, key, cast) in env_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                setattr(getattr(config, section), key, cast(val))
        return config
