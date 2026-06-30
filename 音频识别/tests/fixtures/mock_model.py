"""Mock 模型 — 用于 CI 无 GPU/无权重环境下测试推理逻辑."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class MockW2vSELDModel(nn.Module):
    """
    模拟 w2v-SELD 模型行为。

    返回固定 shape 的随机输出，使预处理/后处理逻辑可完全测试
    而无需下载 400MB+ 的真实模型权重。
    """

    def __init__(
        self,
        num_frames: int = 20,
        num_classes: int = 7,
        doa_channels: int = 3,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.doa_channels = doa_channels

    def forward(
        self,
        source: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        模拟 w2v-SELD forward 调用。

        Args:
            source: 输入音频特征 (batch, channels, samples) 或 (batch, samples)。

        Returns:
            (sed_logits, doa_logits) — 模拟 SED 和 DOA 输出。
        """
        batch_size = source.shape[0] if source.ndim >= 2 else 1

        # 帧数：根据输入长度估算（每 160 个样本 ≈ 1 帧，对应 10ms@16kHz）
        if source.ndim >= 3:
            approx_frames = max(1, source.shape[2] // 160)
        else:
            approx_frames = self.num_frames

        sed_logits = torch.randn(batch_size, approx_frames, self.num_classes)
        doa_logits = torch.randn(batch_size, approx_frames, self.doa_channels)

        return sed_logits, doa_logits


class MockW2vSELDEngine:
    """
    轻量级 Mock 引擎，不依赖 fairseq/torchaudio。

    用于单元测试中验证数据流、异常处理和配置加载，
    避免引入重型 ML 依赖。
    """

    def __init__(self, config=None):
        from seld.config import SELDConfig

        self.config = config or SELDConfig()
        self.device = torch.device("cpu")
        self.threshold = self.config.inference.event_threshold
        self.min_duration_ms = self.config.inference.min_event_duration_ms
        self.frame_ms = self.config.audio.frame_duration_ms
        self.model = MockW2vSELDModel()
        self._inference_count = 0

    @property
    def model_loaded(self) -> bool:
        return True

    @property
    def inference_count(self) -> int:
        return self._inference_count
