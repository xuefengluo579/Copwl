"""pytest 共享 fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 确保 src/ 在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from seld.config import SELDConfig
from seld.engine import W2vSELDEngine
from seld.streaming import StreamingSELDEngine
from tests.fixtures.mock_model import MockW2vSELDModel
from tests.fixtures.synthetic import (
    generate_synthetic_audio,
    generate_event_audio,
    generate_silent_audio,
)


@pytest.fixture
def test_config() -> SELDConfig:
    """返回 CPU 推理用的最小配置."""
    config = SELDConfig()
    config.model.device = "cpu"
    config.model.use_fairseq = False
    config.inference.event_threshold = 0.5
    config.inference.min_event_duration_ms = 100
    return config


@pytest.fixture
def mock_model() -> MockW2vSELDModel:
    """返回 Mock 模型实例."""
    return MockW2vSELDModel(num_frames=20, num_classes=7)


@pytest.fixture
def synthetic_audio() -> np.ndarray:
    """生成 2 秒四通道测试音频."""
    return generate_synthetic_audio(duration_sec=2.0, seed=42)


@pytest.fixture
def event_audio() -> np.ndarray:
    """生成包含标注事件的测试音频."""
    return generate_event_audio(duration_sec=2.0, seed=42)


@pytest.fixture
def silent_audio() -> np.ndarray:
    """生成全零音频."""
    return generate_silent_audio(duration_sec=1.0)


@pytest.fixture
def engine(test_config, mock_model) -> W2vSELDEngine:
    """返回预初始化的 W2vSELDEngine（使用 mock 模型）."""
    eng = W2vSELDEngine.__new__(W2vSELDEngine)
    eng.config = test_config
    eng.model_cfg = test_config.model
    eng.audio_cfg = test_config.audio
    eng.infer_cfg = test_config.inference
    eng.device = "cpu"
    eng.model = mock_model
    eng.threshold = test_config.inference.event_threshold
    eng.min_duration_ms = test_config.inference.min_event_duration_ms
    eng.frame_ms = test_config.audio.frame_duration_ms
    eng._inference_count = 0
    return eng


@pytest.fixture
def streaming_engine(test_config, mock_model) -> StreamingSELDEngine:
    """返回预初始化的 StreamingSELDEngine（使用 mock 模型）."""
    eng = StreamingSELDEngine.__new__(StreamingSELDEngine)
    eng.config = test_config
    eng.model_cfg = test_config.model
    eng.audio_cfg = test_config.audio
    eng.infer_cfg = test_config.inference
    eng.device = "cpu"
    eng.model = mock_model
    eng.threshold = test_config.inference.event_threshold
    eng.min_duration_ms = test_config.inference.min_event_duration_ms
    eng.frame_ms = test_config.audio.frame_duration_ms
    eng._inference_count = 0
    eng.window_seconds = 0.5
    eng.hop_seconds = 2.0  # 故意设大以避免 buffer 满时步进
    eng.window_size = int(0.5 * test_config.audio.sample_rate)
    eng.hop_size = 8000
    eng.buffer = np.zeros(
        (eng.audio_cfg.num_channels, eng.window_size),
        dtype=np.float32,
    )
    eng._samples_buffered = 0
    eng._total_samples_processed = 0
    return eng


@pytest.fixture
def test_client():
    """返回 FastAPI TestClient."""
    from fastapi.testclient import TestClient
    from seld.api import app

    return TestClient(app)
