"""测试 EmotionEngine 模块（无需 modelscope，用 mock pipeline）."""

import numpy as np
import pytest

from emotion.config import EmotionConfig
from emotion.models import EmotionType
from seld.exceptions import ConfigurationError


class TestEmotionType:

    def test_all_labels_exist(self):
        assert EmotionType.num_classes() == 8

    def test_from_str_english(self):
        assert EmotionType.from_str("happy") == EmotionType.HAPPY
        assert EmotionType.from_str("SAD") == EmotionType.SAD

    def test_from_str_chinese(self):
        assert EmotionType.from_str("开心") == EmotionType.HAPPY
        assert EmotionType.from_str("恐惧") == EmotionType.FEAR
        assert EmotionType.from_str("痛苦") == EmotionType.PAIN

    def test_from_str_unknown(self):
        with pytest.raises(ValueError):
            EmotionType.from_str("unknown_emotion")

    def test_cn_labels(self):
        labels = EmotionType.cn_labels()
        assert len(labels) == 8
        assert "开心" in labels
        assert "痛苦" in labels

    def test_distress_weights(self):
        assert EmotionType.distress_weight("pain") == 0.95
        assert EmotionType.distress_weight("happy") == -0.2
        assert EmotionType.distress_weight("neutral") == 0.0
        assert EmotionType.distress_weight("unknown") == 0.0

    def test_negative_set(self):
        neg = EmotionType.negative_set()
        assert "angry" in neg
        assert "pain" in neg
        assert "happy" not in neg
        assert "neutral" not in neg


class TestEmotionConfig:

    def test_defaults(self):
        config = EmotionConfig()
        assert config.model_name == "iic/emotion2vec_base"
        assert config.device == "auto"
        assert config.confidence_threshold == 0.3
        assert config.allow_cpu_fallback is True

    def test_validate_ok(self):
        EmotionConfig().validate()

    def test_validate_bad_device(self):
        config = EmotionConfig(device="invalid")
        with pytest.raises(ConfigurationError):
            config.validate()

    def test_validate_bad_threshold(self):
        config = EmotionConfig(confidence_threshold=1.5)
        with pytest.raises(ConfigurationError):
            config.validate()

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("EMOTION_DEVICE", "cpu")
        monkeypatch.setenv("EMOTION_CONFIDENCE_THRESHOLD", "0.5")
        config = EmotionConfig.from_env()
        assert config.device == "cpu"
        assert config.confidence_threshold == 0.5


class TestEngineMock:
    """用 mock pipeline 测试引擎逻辑（无需 modelscope）."""

    @pytest.fixture
    def mock_pipeline(self):
        """返回一个假的 pipeline 函数."""

        def _mock(audio):
            return {
                "scores": [0.85, 0.08, 0.04, 0.02, 0.01, 0.0, 0.0, 0.0],
                "labels": ["开心", "悲伤", "生气", "中性", "恐惧", "惊讶", "厌恶", "痛苦"],
            }

        return _mock

    @pytest.fixture
    def engine(self, mock_pipeline, monkeypatch):
        """创建一个绕过 modelscope 加载的 EmotionEngine."""
        from emotion.engine import EmotionEngine, EmotionConfig

        config = EmotionConfig()
        eng = EmotionEngine.__new__(EmotionEngine)
        eng.config = config
        eng.device = __import__("torch").device("cpu")
        eng.device_str = "cpu"
        eng._pipeline = mock_pipeline
        eng._inference_count = 0
        return eng

    def test_infer_returns_label_and_confidence(self, engine):
        audio = np.random.randn(4, 1600).astype(np.float32) * 0.1
        label, conf = engine.infer(audio)
        assert isinstance(label, str)
        assert 0.0 <= conf <= 1.0

    def test_infer_happy_from_chinese_label(self, engine):
        audio = np.random.randn(1, 3200).astype(np.float32) * 0.1
        label, conf = engine.infer(audio)
        assert label == "happy"
        assert conf == 0.85

    def test_silent_input_returns_none(self, engine):
        audio = np.zeros((4, 1600), dtype=np.float32)
        label, conf = engine.infer(audio)
        assert label is None
        assert conf == 0.0

    def test_rejects_bad_input(self, engine):
        with pytest.raises(Exception):
            engine.infer(np.array([1.0]))  # 1D

    def test_inference_count_increments(self, engine):
        audio = np.random.randn(4, 1600).astype(np.float32) * 0.1
        engine.infer(audio)
        assert engine.inference_count >= 1

    def test_model_loaded(self, engine):
        assert engine.model_loaded is True
