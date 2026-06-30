"""w2v-SELD 流式推理适配器."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from seld.config import SELDConfig, StreamingConfig
from seld.engine import W2vSELDEngine
from seld.logging_utils import get_logger
from seld.models import DetectedEvent

logger = get_logger(__name__)


class StreamingSELDEngine(W2vSELDEngine):
    """
    支持流式输入的 SED 引擎。

    维护一个滑动窗口缓冲区，当缓冲区填满时触发推理。
    适用于实时音频流场景（麦克风阵列持续输入）。

    Usage:
        config = SELDConfig.from_yaml("config/default.yaml")
        engine = StreamingSELDEngine(config, window_seconds=2.0, hop_seconds=0.5)
        while streaming:
            chunk = microphone.read()  # shape (4, chunk_samples)
            events = engine.feed(chunk)
            if events:
                handle_events(events)
    """

    def __init__(
        self,
        config: SELDConfig,
        *,
        window_seconds: Optional[float] = None,
        hop_seconds: Optional[float] = None,
    ):
        """
        初始化流式引擎。

        Args:
            config: SELDConfig 配置实例。
            window_seconds: 滑动窗口长度（秒）。为 None 时使用配置值。
            hop_seconds: 窗口步进（秒）。为 None 时使用配置值。
        """
        super().__init__(config)

        stream_cfg: StreamingConfig = config.streaming
        self.window_seconds = window_seconds or stream_cfg.window_seconds
        self.hop_seconds = hop_seconds or stream_cfg.hop_seconds

        # 计算采样点数
        self.window_size = int(self.window_seconds * self.audio_cfg.sample_rate)
        self.hop_size = int(self.hop_seconds * self.audio_cfg.sample_rate)

        # 初始化循环缓冲区
        self.buffer = np.zeros(
            (self.audio_cfg.num_channels, self.window_size),
            dtype=np.float32,
        )
        self._samples_buffered: int = 0
        self._total_samples_processed: int = 0

        logger.info(
            "Streaming engine: window=%.1fs (%d samples), hop=%.1fs (%d samples)",
            self.window_seconds,
            self.window_size,
            self.hop_seconds,
            self.hop_size,
        )

    def feed(self, chunk: np.ndarray) -> Optional[List[DetectedEvent]]:
        """
        接收音频块，当缓冲区满时返回检测结果。

        Args:
            chunk: 音频块，shape (channels, chunk_samples)。

        Returns:
            检测到的事件列表，缓冲区未满时返回 None。

        Raises:
            InvalidAudioError: 输入校验失败。
        """
        if chunk.ndim != 2:
            raise ValueError(
                f"Expected 2D array (channels, samples), got {chunk.ndim}D"
            )
        if chunk.shape[0] != self.audio_cfg.num_channels:
            raise ValueError(
                f"Expected {self.audio_cfg.num_channels} channels, got {chunk.shape[0]}"
            )

        chunk_len = chunk.shape[1]
        self._total_samples_processed += chunk_len

        # 追加到缓冲区（滚动）
        if chunk_len >= self.window_size:
            # 块大于窗口 → 取最后 window_size 个样本
            self.buffer = chunk[:, -self.window_size:].astype(np.float32)
            self._samples_buffered = self.window_size
        else:
            # 滚动缓冲区：左移 chunk_len，右侧填入新数据
            self.buffer = np.roll(self.buffer, -chunk_len, axis=1)
            self.buffer[:, -chunk_len:] = chunk.astype(np.float32)
            self._samples_buffered = min(
                self._samples_buffered + chunk_len,
                self.window_size,
            )

        # 缓冲区未满，不触发推理
        if self._samples_buffered < self.window_size:
            logger.debug(
                "Buffer: %d/%d samples (%.1f%%)",
                self._samples_buffered,
                self.window_size,
                100.0 * self._samples_buffered / self.window_size,
            )
            return None

        # 缓冲区满，触发推理
        events, elapsed = self.infer(self.buffer)
        logger.info(
            "Streaming inference: %.1fms, %d events (buffer %.1fs)",
            elapsed,
            len(events),
            self.window_seconds,
        )

        # 步进：左移 hop_size，腾出空间
        self.buffer = np.roll(self.buffer, -self.hop_size, axis=1)
        self.buffer[:, -self.hop_size:] = 0.0
        self._samples_buffered = max(0, self._samples_buffered - self.hop_size)

        return events if events else None

    def reset(self) -> None:
        """重置缓冲区状态."""
        self.buffer.fill(0.0)
        self._samples_buffered = 0
        self._total_samples_processed = 0
        logger.info("Streaming buffer reset")

    @property
    def buffer_fill_percent(self) -> float:
        """缓冲区填充百分比."""
        return 100.0 * self._samples_buffered / self.window_size

    @property
    def total_seconds_processed(self) -> float:
        """累计处理的音频时长（秒）."""
        return self._total_samples_processed / self.audio_cfg.sample_rate

    def __repr__(self) -> str:
        return (
            f"StreamingSELDEngine("
            f"device={self.device}, "
            f"window={self.window_seconds}s, "
            f"hop={self.hop_seconds}s, "
            f"buffer={self._samples_buffered}/{self.window_size})"
        )
