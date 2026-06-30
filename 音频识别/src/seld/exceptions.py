"""w2v-SELD 自定义异常层次结构."""


class SELDError(Exception):
    """所有 SED 模块异常的基类."""

    def __init__(self, message: str = "", *, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


# ── 模型加载异常 ──────────────────────────────────────────

class ModelLoadError(SELDError):
    """模型权重加载或实例化失败."""


class WeightNotFoundError(ModelLoadError):
    """checkpoint 文件不存在."""


class ArchitectureError(ModelLoadError):
    """模型配置与 checkpoint 不兼容."""


# ── 音频输入异常 ──────────────────────────────────────────

class InvalidAudioError(SELDError):
    """音频输入校验未通过."""


class ChannelMismatchError(InvalidAudioError):
    """音频通道数与要求不一致."""


class SampleRateError(InvalidAudioError):
    """音频采样率与要求不一致（且不支持自动重采样时抛出）."""


class SilentAudioError(InvalidAudioError):
    """输入音频能量过低（静音或近静音）."""


class AudioFormatError(InvalidAudioError):
    """无法解析的音频格式."""


# ── 推理异常 ──────────────────────────────────────────────

class InferenceError(SELDError):
    """模型推理过程中发生的运行时错误（CUDA OOM 等）."""


# ── 配置异常 ──────────────────────────────────────────────

class ConfigurationError(SELDError):
    """配置值无效或不一致."""
