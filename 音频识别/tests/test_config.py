"""测试配置管理系统."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from seld.config import (
    SELDConfig,
    ModelConfig,
    AudioConfig,
    InferenceConfig,
    StreamingConfig,
)
from seld.exceptions import ConfigurationError


class TestDefaultConfig:
    """测试默认配置."""

    def test_default_values(self):
        config = SELDConfig()
        assert config.model.device == "auto"
        assert config.audio.sample_rate == 16000
        assert config.audio.num_channels == 4
        assert config.inference.event_threshold == 0.5

    def test_validate_ok(self):
        config = SELDConfig()
        config.validate()  # 不抛异常

    def test_validate_bad_device(self):
        config = SELDConfig()
        config.model.device = "invalid"
        with pytest.raises(ConfigurationError, match="device"):
            config.validate()

    def test_validate_bad_threshold_negative(self):
        config = SELDConfig()
        config.inference.event_threshold = 0.0
        with pytest.raises(ConfigurationError, match="threshold"):
            config.validate()

    def test_validate_bad_threshold_over_one(self):
        config = SELDConfig()
        config.inference.event_threshold = 1.5
        with pytest.raises(ConfigurationError, match="threshold"):
            config.validate()

    def test_validate_bad_channels(self):
        config = SELDConfig()
        config.audio.num_channels = 0
        with pytest.raises(ConfigurationError, match="num_channels"):
            config.validate()

    def test_validate_negative_min_duration(self):
        config = SELDConfig()
        config.inference.min_event_duration_ms = -1
        with pytest.raises(ConfigurationError, match="min_event_duration"):
            config.validate()


class TestYAMLLoading:
    """测试 YAML 配置加载."""

    def test_from_yaml_file(self, tmp_path):
        data = {
            "model": {"device": "cpu", "checkpoint_path": "/tmp/test.pt"},
            "audio": {"sample_rate": 16000},
            "inference": {"event_threshold": 0.7},
        }
        yaml_path = tmp_path / "test_config.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(data, f)

        config = SELDConfig.from_yaml(yaml_path)
        assert config.model.device == "cpu"
        assert config.inference.event_threshold == 0.7
        # 未指定的字段保持默认值
        assert config.audio.num_channels == 4

    def test_from_nonexistent_file(self):
        """不存在的文件返回默认配置."""
        config = SELDConfig.from_yaml("/nonexistent/config.yaml")
        assert config.model.device == "auto"

    def test_empty_yaml(self, tmp_path):
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")
        config = SELDConfig.from_yaml(yaml_path)
        assert config.model.device == "auto"


class TestEnvOverrides:
    """测试环境变量覆盖."""

    def test_env_device_override(self):
        os.environ["SELD_DEVICE"] = "cpu"
        try:
            config = SELDConfig()
            config = SELDConfig._apply_env_overrides(config)
            assert config.model.device == "cpu"
        finally:
            del os.environ["SELD_DEVICE"]

    def test_env_threshold_override(self):
        os.environ["SELD_EVENT_THRESHOLD"] = "0.75"
        try:
            config = SELDConfig()
            config = SELDConfig._apply_env_overrides(config)
            assert config.inference.event_threshold == 0.75
        finally:
            del os.environ["SELD_EVENT_THRESHOLD"]

    def test_env_int_override(self):
        os.environ["SELD_SAMPLE_RATE"] = "8000"
        try:
            config = SELDConfig()
            config = SELDConfig._apply_env_overrides(config)
            assert config.audio.sample_rate == 8000
        finally:
            del os.environ["SELD_SAMPLE_RATE"]

    def test_multiple_env_overrides(self):
        os.environ["SELD_DEVICE"] = "cpu"
        os.environ["SELD_BATCH_SIZE"] = "4"
        os.environ["SELD_WINDOW_SECONDS"] = "3.0"
        try:
            config = SELDConfig()
            config = SELDConfig._apply_env_overrides(config)
            assert config.model.device == "cpu"
            assert config.inference.batch_size == 4
            assert config.streaming.window_seconds == 3.0
        finally:
            for k in ["SELD_DEVICE", "SELD_BATCH_SIZE", "SELD_WINDOW_SECONDS"]:
                os.environ.pop(k, None)


class TestSerialization:
    """测试配置序列化."""

    def test_to_dict(self):
        config = SELDConfig()
        d = config.to_dict()
        assert "model" in d
        assert "audio" in d
        assert d["model"]["device"] == "auto"
        assert d["audio"]["sample_rate"] == 16000

    def test_to_yaml(self, tmp_path):
        config = SELDConfig()
        config.model.device = "cuda"
        yaml_path = tmp_path / "output.yaml"
        config.to_yaml(yaml_path)
        assert yaml_path.exists()

        # 验证可重新加载
        reloaded = SELDConfig.from_yaml(yaml_path)
        assert reloaded.model.device == "cuda"


class TestSubConfigs:
    """测试子配置类."""

    def test_model_config_defaults(self):
        mc = ModelConfig()
        assert mc.device == "auto"
        assert mc.use_fairseq is True

    def test_audio_config_defaults(self):
        ac = AudioConfig()
        assert ac.sample_rate == 16000
        assert ac.num_channels == 4
        assert ac.frame_duration_ms == 100

    def test_streaming_config_defaults(self):
        sc = StreamingConfig()
        assert sc.window_seconds == 2.0
        assert sc.hop_seconds == 0.5
