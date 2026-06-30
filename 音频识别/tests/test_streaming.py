"""测试 StreamingSELDEngine 流式引擎."""

import numpy as np
import pytest


class TestBufferManagement:
    """测试缓冲区管理."""

    def test_initial_buffer_empty(self, streaming_engine):
        assert streaming_engine.buffer_fill_percent == 0.0

    def test_buffer_fills_gradually(self, streaming_engine):
        chunk = np.random.randn(4, 2000).astype(np.float32)
        streaming_engine.feed(chunk)
        assert streaming_engine.buffer_fill_percent > 0

    def test_feed_returns_none_when_not_full(self, streaming_engine):
        chunk = np.random.randn(4, 1000).astype(np.float32)
        # window_size=8000, feed 1000 → 未满
        result = streaming_engine.feed(chunk)
        assert result is None

    def test_feed_large_enough_triggers(self, streaming_engine):
        # 一次性填充整个窗口
        chunk = np.random.randn(4, streaming_engine.window_size).astype(np.float32)
        result = streaming_engine.feed(chunk)
        # 可能返回 list（有事件）或 None（无事件）
        assert result is None or isinstance(result, list)

    def test_chunk_larger_than_window(self, streaming_engine):
        """块大于窗口时，取最后 window_size 个样本并触发推理."""
        chunk = np.random.randn(4, streaming_engine.window_size + 1000).astype(np.float32)
        # 注入一个强激活以触发事件（避免全零导致无事件返回）
        # 由于 mock 模型返回随机值，这里只验证不会崩溃
        result = streaming_engine.feed(chunk)
        # feed 应返回结果或 None，不应抛出异常
        assert result is None or isinstance(result, list)
        # 推理后缓冲区因 hop 步进可能非满，验证引擎正常运转
        assert streaming_engine._total_samples_processed > 0

    def test_reset_clears_buffer(self, streaming_engine):
        chunk = np.random.randn(4, 4000).astype(np.float32)
        streaming_engine.feed(chunk)
        assert streaming_engine.buffer_fill_percent > 0
        streaming_engine.reset()
        assert streaming_engine.buffer_fill_percent == 0.0
        assert streaming_engine.total_seconds_processed == 0.0

    def test_total_seconds_tracks(self, streaming_engine):
        chunk = np.random.randn(4, 4000).astype(np.float32)
        streaming_engine.feed(chunk)
        assert streaming_engine.total_seconds_processed == pytest.approx(0.25)  # 4000/16000

    def test_repr(self, streaming_engine):
        r = repr(streaming_engine)
        assert "StreamingSELDEngine" in r
        assert "window=" in r


class TestStreamingInputValidation:
    """测试流式输入校验."""

    def test_rejects_1d_chunk(self, streaming_engine):
        chunk = np.random.randn(8000).astype(np.float32)
        with pytest.raises(ValueError, match="2D"):
            streaming_engine.feed(chunk)

    def test_rejects_wrong_channels(self, streaming_engine):
        chunk = np.random.randn(2, 1000).astype(np.float32)
        with pytest.raises(ValueError, match="channels"):
            streaming_engine.feed(chunk)
