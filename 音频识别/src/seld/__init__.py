"""
w2v-SELD — 基于 wav2vec 2.0 的声音事件检测与定位推理引擎。

Usage:
    from seld import W2vSELDEngine, SELDConfig

    config = SELDConfig.from_yaml("config/default.yaml")
    engine = W2vSELDEngine(config)
    events = engine.infer(audio_array)
"""

__version__ = "0.1.0"

from seld.config import SELDConfig, ModelConfig, AudioConfig, InferenceConfig
from seld.exceptions import (
    SELDError,
    ModelLoadError,
    InvalidAudioError,
    ChannelMismatchError,
    SilentAudioError,
    InferenceError,
    ConfigurationError,
)
from seld.audio import (
    load_audio,
    validate_audio,
    resample_audio,
    mono_to_ambisonics,
    audio_from_bytes,
)
from seld.engine import W2vSELDEngine
from seld.streaming import StreamingSELDEngine
from seld.models import (
    EventType,
    DetectedEvent,
    DOAVector,
    DetectionResult,
    HealthStatus,
)
from seld.logging_utils import setup_logging, get_logger

__all__ = [
    # Version
    "__version__",
    # Config
    "SELDConfig",
    "ModelConfig",
    "AudioConfig",
    "InferenceConfig",
    # Exceptions
    "SELDError",
    "ModelLoadError",
    "InvalidAudioError",
    "ChannelMismatchError",
    "SilentAudioError",
    "InferenceError",
    "ConfigurationError",
    # Audio
    "load_audio",
    "validate_audio",
    "resample_audio",
    "mono_to_ambisonics",
    "audio_from_bytes",
    # Engine
    "W2vSELDEngine",
    "StreamingSELDEngine",
    # Models
    "EventType",
    "DetectedEvent",
    "DOAVector",
    "DetectionResult",
    "HealthStatus",
    # Logging
    "setup_logging",
    "get_logger",
]
