"""测试自定义异常层次结构."""

import pytest

from seld.exceptions import (
    SELDError,
    ModelLoadError,
    WeightNotFoundError,
    ArchitectureError,
    InvalidAudioError,
    ChannelMismatchError,
    SampleRateError,
    SilentAudioError,
    AudioFormatError,
    InferenceError,
    ConfigurationError,
)


class TestExceptionHierarchy:
    """验证异常继承关系."""

    def test_base_is_exception(self):
        assert issubclass(SELDError, Exception)

    def test_model_load_chain(self):
        assert issubclass(ModelLoadError, SELDError)
        assert issubclass(WeightNotFoundError, ModelLoadError)
        assert issubclass(ArchitectureError, ModelLoadError)

    def test_audio_chain(self):
        assert issubclass(InvalidAudioError, SELDError)
        assert issubclass(ChannelMismatchError, InvalidAudioError)
        assert issubclass(SampleRateError, InvalidAudioError)
        assert issubclass(SilentAudioError, InvalidAudioError)
        assert issubclass(AudioFormatError, InvalidAudioError)

    def test_inference_chain(self):
        assert issubclass(InferenceError, SELDError)

    def test_config_chain(self):
        assert issubclass(ConfigurationError, SELDError)


class TestExceptionCatching:
    """验证异常可被正确捕获."""

    def test_catch_by_base(self):
        with pytest.raises(SELDError):
            raise ChannelMismatchError("test")

    def test_catch_by_audio(self):
        with pytest.raises(InvalidAudioError):
            raise SilentAudioError("test")

    def test_details_preserved(self):
        exc = SELDError("message", details={"key": "value"})
        assert exc.details == {"key": "value"}
        assert str(exc) == "message"

    def test_no_details_default(self):
        exc = SELDError("test")
        assert exc.details == {}
