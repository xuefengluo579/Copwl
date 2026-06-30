"""合成测试音频生成."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def generate_synthetic_audio(
    sample_rate: int = 16000,
    num_channels: int = 4,
    duration_sec: float = 2.0,
    frequencies: Optional[List[float]] = None,
    amplitude: float = 0.8,
    noise_std: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """
    生成四通道伪 Ambisonics 测试音频。

    通道映射（B 格式 ACN）:
        ch0: W  (omni — 全向)
        ch1: X  (front-back)
        ch2: Y  (left-right)
        ch3: Z  (up-down)

    通过在不同通道放置不同频率的正弦波来模拟方位信息。

    Args:
        sample_rate: 采样率（Hz）。
        num_channels: 通道数（通常为 4）。
        duration_sec: 音频时长（秒）。
        frequencies: 各通道的正弦波频率（Hz）。默认 [440, 550, 660, 770]。
        amplitude: 信号幅度 (0~1)。
        noise_std: 高斯噪声标准差。
        seed: 随机种子。

    Returns:
        shape (num_channels, samples) 的 float32 数组。
    """
    rng = np.random.RandomState(seed)

    if frequencies is None:
        frequencies = [440.0, 550.0, 660.0, 770.0]
    if len(frequencies) < num_channels:
        frequencies = frequencies * num_channels

    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    audio = np.zeros((num_channels, num_samples), dtype=np.float32)

    for ch in range(num_channels):
        freq = frequencies[ch % len(frequencies)]
        # W 通道（ch0）混合所有频率，其他通道为各自频率
        if ch == 0:
            signal = sum(
                amplitude * np.sin(2 * np.pi * f * t)
                for f in frequencies[:num_channels]
            ) / num_channels
        else:
            signal = amplitude * np.sin(2 * np.pi * freq * t)

        # 添加声道间微小延迟（模拟方位）
        delay_samples = ch * 2
        if delay_samples > 0:
            signal = np.roll(signal, delay_samples)

        audio[ch] = signal.astype(np.float32)

    # 添加低噪底
    audio += noise_std * rng.randn(*audio.shape).astype(np.float32)

    return audio


def generate_event_audio(
    sample_rate: int = 16000,
    num_channels: int = 4,
    duration_sec: float = 2.0,
    events: Optional[List[Tuple[float, float, int]]] = None,
    seed: int = 42,
) -> np.ndarray:
    """
    生成包含可辨识声音事件的测试音频。

    每个事件在其持续时间内叠加一个特定频率的正弦脉冲
    （带衰减包络），方便验证后处理逻辑中的事件检测。

    Args:
        sample_rate: 采样率。
        num_channels: 通道数。
        duration_sec: 总时长。
        events: (start_sec, end_sec, class_index) 列表。
                默认: [(0.3, 0.8, 0), (1.2, 1.6, 3)]
        seed: 随机种子。

    Returns:
        shape (num_channels, samples) 的 float32 数组。
    """
    rng = np.random.RandomState(seed)

    # 默认事件: speech@0.3s, knock@1.2s
    if events is None:
        events = [
            (0.3, 0.8, 0),  # speech
            (1.2, 1.6, 3),  # knock
        ]

    num_samples = int(sample_rate * duration_sec)

    # 底噪
    audio = (0.005 * rng.randn(num_channels, num_samples)).astype(np.float32)

    # 为不同事件类型分配特征频率
    event_frequencies = [
        300.0,   # 0: speech
        1000.0,  # 1: fall (低频冲击)
        2000.0,  # 2: glass_break (高频)
        500.0,   # 3: knock
        400.0,   # 4: object_drop
        200.0,   # 5: footstep
        2500.0,  # 6: door_slam
    ]

    for start_sec, end_sec, class_idx in events:
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)
        event_len = end_sample - start_sample

        t_event = np.linspace(0, (end_sec - start_sec), event_len, endpoint=False)
        freq = event_frequencies[class_idx % len(event_frequencies)]

        # 带衰减包络的脉冲
        envelope = np.exp(-3.0 * t_event / (end_sec - start_sec + 0.01))
        signal = 0.6 * envelope * np.sin(2 * np.pi * freq * t_event)

        # 写入 W 通道 (ch0)
        audio[0, start_sample:end_sample] += signal.astype(np.float32)

        # 方位模拟：X 通道 (ch1) 放置偏向一侧的信号
        azimuth_gain = 0.3 * (1.0 if class_idx % 2 == 0 else -0.7)
        audio[1, start_sample:end_sample] += (
            azimuth_gain * signal
        ).astype(np.float32)

    return audio


def generate_silent_audio(
    sample_rate: int = 16000,
    num_channels: int = 4,
    duration_sec: float = 1.0,
) -> np.ndarray:
    """生成全零音频（用于边界测试）."""
    num_samples = int(sample_rate * duration_sec)
    return np.zeros((num_channels, num_samples), dtype=np.float32)
