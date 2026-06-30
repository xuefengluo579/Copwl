"""测试 W2vSELDEngine 核心引擎."""

import numpy as np
import pytest
import torch

from seld.config import SELDConfig
from seld.engine import W2vSELDEngine
from seld.exceptions import InvalidAudioError, ChannelMismatchError
from seld.models import EventType, DetectedEvent
from tests.fixtures.mock_model import MockW2vSELDModel


class TestDeviceResolution:
    """测试设备解析."""

    def test_explicit_cpu(self):
        device = W2vSELDEngine._resolve_device("cpu")
        assert device.type == "cpu"

    def test_auto(self):
        device = W2vSELDEngine._resolve_device("auto")
        assert device.type in ("cpu", "cuda", "mps")


class TestPreprocess:
    """测试预处理流水线."""

    def test_output_shape(self, engine, synthetic_audio):
        result = engine.preprocess(synthetic_audio)
        assert result.ndim == 3  # (batch, channels, samples)
        assert result.shape[0] == 1
        assert result.shape[1] == 4

    def test_normalization_zero_mean(self, engine, synthetic_audio):
        result = engine.preprocess(synthetic_audio)
        # 逐通道均值应接近 0
        means = result[0].mean(dim=1)
        assert torch.all(torch.abs(means) < 1e-4)

    def test_normalization_unit_variance(self, engine, synthetic_audio):
        result = engine.preprocess(synthetic_audio)
        # 逐通道标准差应接近 1
        stds = result[0].std(dim=1)
        assert torch.all(torch.abs(stds - 1.0) < 0.1)

    def test_rejects_wrong_channels(self, engine):
        audio = np.zeros((2, 8000), dtype=np.float32)
        with pytest.raises(ChannelMismatchError):
            engine.preprocess(audio)

    def test_device_placement(self, engine, synthetic_audio):
        result = engine.preprocess(synthetic_audio)
        assert result.device.type == "cpu"


class TestPostprocess:
    """测试后处理流水线."""

    def test_returns_list(self, engine):
        logits = torch.randn(1, 20, 7)
        events = engine.postprocess(logits)
        assert isinstance(events, list)

    def test_all_items_are_detected_events(self, engine):
        logits = torch.randn(1, 20, 7)
        events = engine.postprocess(logits)
        for e in events:
            assert isinstance(e, DetectedEvent)

    def test_high_threshold_no_events(self, engine):
        """阈值足够高时不产生事件."""
        engine.threshold = 0.99
        logits = torch.randn(1, 20, 7)  # ~0.5 概率，远低于 0.99
        events = engine.postprocess(logits)
        assert len(events) == 0

    def test_low_threshold_produces_events(self, engine):
        """阈值足够低时产生事件."""
        engine.threshold = 0.01
        logits = torch.randn(1, 20, 7)  # ~0.5 概率，高于 0.01
        events = engine.postprocess(logits)
        assert len(events) > 0

    def test_single_activation(self, engine):
        """单帧激活产生一个事件."""
        engine.threshold = 0.5
        # 将 class 0 的第 5 帧设为高概率
        logits = torch.full((1, 20, 7), -10.0)
        logits[0, 5, 0] = 10.0  # class 0 在 frame 5 高激活
        events = engine.postprocess(logits)
        # 应该至少有 1 个 speech 事件
        speech_events = [e for e in events if e.event == EventType.SPEECH]
        assert len(speech_events) >= 1

    def test_contiguous_activation_merged(self, engine):
        """连续激活应合并为一个事件."""
        engine.threshold = 0.5
        logits = torch.full((1, 30, 7), -10.0)
        # class 1 (fall) frames 5-15 高激活
        logits[0, 5:16, 1] = 10.0
        events = engine.postprocess(logits)
        fall_events = [e for e in events if e.event == EventType.FALL]
        assert len(fall_events) == 1
        assert fall_events[0].t_start_ms == 500  # frame 5 * 100ms
        assert fall_events[0].t_end_ms == 1600  # frame 16 * 100ms

    def test_event_sorted_by_time(self, engine):
        """事件应按时间排序."""
        engine.threshold = 0.3
        logits = torch.full((1, 30, 7), -10.0)
        # class 0 在 frame 15, class 3 在 frame 2
        logits[0, 15, 0] = 10.0
        logits[0, 2, 3] = 10.0
        events = engine.postprocess(logits)
        # 验证时间顺序
        for i in range(len(events) - 1):
            assert events[i].t_start_ms <= events[i + 1].t_start_ms

    def test_min_duration_filter(self, engine):
        """过短事件应被过滤."""
        engine.threshold = 0.5
        engine.min_duration_ms = 200  # 至少 200ms
        logits = torch.full((1, 20, 7), -10.0)
        logits[0, 5, 0] = 10.0  # 单帧 = 100ms，应被过滤
        events = engine.postprocess(logits)
        # 不应有事件（单帧 = 100ms < 200ms）
        assert len(events) == 0

    def test_confidence_range(self, engine):
        """置信度应在 [0, 1] 范围内."""
        engine.threshold = 0.3
        logits = torch.randn(1, 20, 7) * 2  # 扩大 logits
        events = engine.postprocess(logits)
        for e in events:
            assert 0.0 <= e.confidence <= 1.0


class TestInferPipeline:
    """测试完整推理流水线."""

    def test_returns_events_and_time(self, engine, synthetic_audio):
        events, elapsed = engine.infer(synthetic_audio)
        assert isinstance(events, list)
        assert isinstance(elapsed, float)
        assert elapsed > 0

    def test_inference_count_increments(self, engine, synthetic_audio):
        before = engine.inference_count
        engine.infer(synthetic_audio)
        assert engine.inference_count == before + 1

    def test_consecutive_inferences(self, engine, synthetic_audio):
        engine.infer(synthetic_audio)
        engine.infer(synthetic_audio)
        assert engine.inference_count == 2

    def test_model_loaded_property(self, engine):
        assert engine.model_loaded is True

    def test_repr(self, engine):
        r = repr(engine)
        assert "W2vSELDEngine" in r
        assert "cpu" in r
