"""测试音频 I/O 工具."""

import io
import os

import numpy as np
import pytest
import torch
import soundfile as sf

from seld.audio import (
    validate_audio,
    load_audio,
    resample_audio,
    mono_to_ambisonics,
    audio_from_bytes,
)
from seld.exceptions import (
    InvalidAudioError,
    ChannelMismatchError,
    SilentAudioError,
    AudioFormatError,
)


class TestValidateAudio:
    """测试音频校验."""

    def test_valid_audio(self, synthetic_audio):
        validate_audio(synthetic_audio)  # 不抛异常

    def test_wrong_type(self):
        with pytest.raises(InvalidAudioError, match="np.ndarray"):
            validate_audio([1.0, 2.0, 3.0])  # type: ignore

    def test_wrong_ndim(self):
        with pytest.raises(InvalidAudioError, match="2D"):
            validate_audio(np.array([1.0, 2.0]))

    def test_wrong_channels(self):
        audio = np.zeros((2, 16000), dtype=np.float32)
        with pytest.raises(ChannelMismatchError, match="Expected 4"):
            validate_audio(audio)

    def test_nan_values(self):
        audio = np.zeros((4, 1000), dtype=np.float32)
        audio[0, 500] = np.nan
        with pytest.raises(InvalidAudioError, match="NaN"):
            validate_audio(audio)

    def test_inf_values(self):
        audio = np.zeros((4, 1000), dtype=np.float32)
        audio[1, 100] = np.inf
        with pytest.raises(InvalidAudioError, match="infinite"):
            validate_audio(audio)

    def test_empty(self):
        with pytest.raises(InvalidAudioError, match="empty"):
            validate_audio(np.zeros((4, 0), dtype=np.float32))

    def test_silence_with_allow(self, silent_audio):
        # 默认允许静音
        validate_audio(silent_audio)  # 不抛异常

    def test_silence_rejected(self, silent_audio):
        with pytest.raises(SilentAudioError, match="silent"):
            validate_audio(silent_audio, allow_silence=False)

    def test_custom_channels(self):
        audio = np.zeros((2, 1000), dtype=np.float32)
        validate_audio(audio, expected_channels=2)  # 不抛异常


class TestResampleAudio:
    """测试重采样."""

    def test_same_rate(self, synthetic_audio):
        result = resample_audio(synthetic_audio, 16000, 16000)
        assert result.shape == synthetic_audio.shape
        np.testing.assert_array_equal(result, synthetic_audio)

    def test_downsample(self):
        # 生成 48kHz 音频，降采样到 16kHz
        audio_48k = np.random.randn(4, 48000).astype(np.float32) * 0.1
        result = resample_audio(audio_48k, 48000, 16000)
        assert result.shape[0] == 4  # 通道数不变
        assert result.shape[1] == 16000  # 采样数 = 48k → 16k

    def test_upsample(self):
        audio_8k = np.random.randn(4, 8000).astype(np.float32) * 0.1
        result = resample_audio(audio_8k, 8000, 16000)
        assert result.shape[1] == 16000


class TestMonoToAmbisonics:
    """测试单通道转 Ambisonics."""

    def test_converts_1_to_4(self):
        mono = torch.randn(1, 16000)
        result = mono_to_ambisonics(mono, num_channels=4)
        assert result.shape == (4, 16000)
        # W 通道 = 原始信号
        assert torch.allclose(result[0], mono[0])
        # X/Y/Z 通道 = 零
        assert torch.all(result[1:] == 0)

    def test_already_multichannel(self):
        audio = torch.randn(4, 16000)
        result = mono_to_ambisonics(audio, num_channels=4)
        assert torch.equal(result, audio)

    def test_numpy_input(self):
        mono = np.random.randn(1, 8000).astype(np.float32)
        result = mono_to_ambisonics(mono, num_channels=4)
        assert result.shape == (4, 8000)

    def test_1d_input(self):
        mono = torch.randn(16000)
        result = mono_to_ambisonics(mono, num_channels=4)
        assert result.shape == (4, 16000)

    def test_wrong_input_channels(self):
        stereo = torch.randn(2, 16000)
        with pytest.raises(ChannelMismatchError):
            mono_to_ambisonics(stereo, num_channels=4)


class TestAudioFromBytes:
    """测试字节流解码."""

    def test_wav_bytes(self, synthetic_audio):
        buf = io.BytesIO()
        # 使用 soundfile 写入 WAV 到内存
        sf.write(buf, synthetic_audio.T, 16000, format="WAV")
        buf.seek(0)
        data = buf.read()

        result = audio_from_bytes(data, target_sample_rate=16000)
        assert result.shape[0] == 4

    def test_bad_bytes(self):
        with pytest.raises(AudioFormatError, match="Cannot decode"):
            audio_from_bytes(b"not audio data at all")


class TestLoadAudio:
    """测试文件加载."""

    def test_load_wav(self, tmp_path, synthetic_audio):
        path = str(tmp_path / "test.wav")
        # 使用 soundfile 写入 WAV（torchaudio 2.9+ 需要 torchcodec）
        sf.write(path, synthetic_audio.T, 16000)
        result = load_audio(path)
        assert result.shape[0] == 4

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_audio("/nonexistent/file.wav")

    def test_mono_auto_upmix(self, tmp_path):
        mono = np.clip(np.random.randn(1, 16000).astype(np.float32), -0.9, 0.9)
        path = str(tmp_path / "mono.wav")
        sf.write(path, mono.T, 16000)
        result = load_audio(path, target_channels=4)
        assert result.shape[0] == 4
        # W 通道 ≈ 原始信号（允许浮点舍入误差）
        np.testing.assert_array_almost_equal(result[0], mono[0], decimal=2)
