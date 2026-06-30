"""w2v-SELD 音频 I/O 工具函数."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
import soundfile as sf

from seld.exceptions import (
    AudioFormatError,
    ChannelMismatchError,
    InvalidAudioError,
    SampleRateError,
    SilentAudioError,
)
from seld.logging_utils import get_logger

logger = get_logger(__name__)

# 静音检测阈值：低于此标准差的音频被视为静音
SILENCE_STD_THRESHOLD = 1e-6


def validate_audio(
    audio: np.ndarray,
    *,
    expected_channels: int = 4,
    expected_sample_rate: Optional[int] = None,
    allow_silence: bool = True,
) -> None:
    """
    校验输入音频数组。

    Args:
        audio: 音频数组，shape (channels, samples)。
        expected_channels: 期望的通道数。
        expected_sample_rate: 期望的采样率（仅用于错误信息，不做实际校验）。
        allow_silence: False 时，静音输入将抛出 SilentAudioError。

    Raises:
        InvalidAudioError: 校验失败。
    """
    if not isinstance(audio, np.ndarray):
        raise InvalidAudioError(
            f"Expected np.ndarray, got {type(audio).__name__}"
        )

    if audio.ndim != 2:
        raise InvalidAudioError(
            f"Expected 2D array (channels, samples), got {audio.ndim}D"
        )

    num_channels = audio.shape[0]
    if num_channels != expected_channels:
        raise ChannelMismatchError(
            f"Expected {expected_channels} channels, got {num_channels}"
        )

    if np.any(np.isnan(audio)):
        raise InvalidAudioError("Audio contains NaN values")

    if np.any(np.isinf(audio)):
        raise InvalidAudioError("Audio contains infinite values")

    if audio.size == 0:
        raise InvalidAudioError("Audio array is empty")

    if not allow_silence and audio.std() < SILENCE_STD_THRESHOLD:
        raise SilentAudioError(
            f"Audio is silent (std={audio.std():.2e}, threshold={SILENCE_STD_THRESHOLD:.2e})"
        )

    if not allow_silence:
        logger.debug(
            "Audio validated: shape=%s, dtype=%s, std=%.6f, max=%.6f",
            audio.shape,
            audio.dtype,
            audio.std(),
            audio.max(),
        )


def load_audio(
    path: str | Path,
    target_sample_rate: int = 16000,
    target_channels: int = 4,
) -> np.ndarray:
    """
    加载音频文件并转换为目标格式。

    Args:
        path: 音频文件路径。
        target_sample_rate: 目标采样率（Hz）。
        target_channels: 目标通道数。

    Returns:
        shape (target_channels, samples) 的 float32 数组。

    Raises:
        AudioFormatError: 文件无法解析。
        ChannelMismatchError: 通道数不匹配。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        audio, sr = torchaudio.load(str(path))
    except Exception:
        # torchaudio 失败时回退到 soundfile（torchaudio 2.9+ 需要 torchcodec）
        try:
            array, sr = sf.read(str(path), dtype="float32")
            audio = torch.from_numpy(array.T if array.ndim == 2 else array[np.newaxis, :])
        except Exception as e:
            raise AudioFormatError(
                f"Failed to load audio file '{path}' (torchaudio + soundfile both failed)"
            ) from e

    # 重采样
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
        logger.debug("Resampled from %d Hz to %d Hz", sr, target_sample_rate)

    # 通道调整
    if audio.shape[0] == 1 and target_channels > 1:
        audio = mono_to_ambisonics(audio, num_channels=target_channels)
        logger.debug("Up-mixed mono to %d-channel pseudo Ambisonics", target_channels)
    elif audio.shape[0] != target_channels:
        raise ChannelMismatchError(
            f"Audio has {audio.shape[0]} channels, expected {target_channels} "
            f"(set target_channels={audio.shape[0]} if this is intentional)"
        )

    return audio.numpy().astype(np.float32)


def resample_audio(
    audio: np.ndarray,
    orig_sample_rate: int,
    target_sample_rate: int = 16000,
) -> np.ndarray:
    """
    重采样音频数据。

    Args:
        audio: 输入音频 (channels, samples)。
        orig_sample_rate: 原始采样率。
        target_sample_rate: 目标采样率。

    Returns:
        重采样后的音频 (channels, new_samples)。
    """
    if orig_sample_rate == target_sample_rate:
        return audio

    tensor = torch.from_numpy(audio).float()
    resampler = torchaudio.transforms.Resample(orig_sample_rate, target_sample_rate)
    return resampler(tensor).numpy().astype(np.float32)


def mono_to_ambisonics(
    audio: np.ndarray | torch.Tensor,
    num_channels: int = 4,
) -> torch.Tensor:
    """
    将单通道音频转换为伪 Ambisonics B 格式。

    策略：W 通道保留原始信号，X/Y/Z 通道为零（零阶 Ambisonics 近似）。
    这不会产生真实的方位信息，但可保证模型输入格式兼容。

    Args:
        audio: 单通道音频，shape (1, samples) 或 (samples,)。
        num_channels: 目标通道数（至少为 1）。

    Returns:
        shape (num_channels, samples) 的 Tensor。
    """
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)

    if audio.shape[0] == num_channels:
        return audio

    if audio.shape[0] != 1:
        raise ChannelMismatchError(
            f"Cannot up-mix: expected 1 channel, got {audio.shape[0]}"
        )

    # W 通道 = 原始信号，其他通道 = 零
    zeros = torch.zeros(num_channels - 1, audio.shape[1], dtype=audio.dtype)
    return torch.cat([audio, zeros], dim=0)


def audio_from_bytes(
    data: bytes,
    target_sample_rate: int = 16000,
    target_channels: int = 4,
) -> np.ndarray:
    """
    从字节流解码音频（用于 API 上传文件处理）。

    Args:
        data: 原始音频字节（WAV/FLAC/OGG 等）。
        target_sample_rate: 目标采样率。
        target_channels: 目标通道数。

    Returns:
        shape (target_channels, samples) 的 float32 数组。
    """
    bio = io.BytesIO(data)
    try:
        audio, sr = torchaudio.load(bio)
    except Exception as e:
        # 尝试用 soundfile 作为后备解码器
        try:
            bio.seek(0)
            array, sr = sf.read(bio, dtype="float32")
            audio = torch.from_numpy(array.T if array.ndim == 2 else array[np.newaxis, :])
        except Exception:
            raise AudioFormatError(
                f"Cannot decode audio data (torchaudio + soundfile both failed). "
                f"torchaudio error: {e}"
            ) from e

    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)

    if audio.shape[0] == 1 and target_channels > 1:
        audio = mono_to_ambisonics(audio, num_channels=target_channels)
    elif audio.shape[0] != target_channels:
        raise ChannelMismatchError(
            f"Decoded audio has {audio.shape[0]} channels, expected {target_channels}"
        )

    return audio.numpy().astype(np.float32)
