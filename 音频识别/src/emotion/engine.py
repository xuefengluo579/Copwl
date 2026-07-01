"""EmotionEngine — 基于 emotion2vec 的语音情绪识别推理引擎.

与 W2vSELDEngine 共享相同的输入格式: (channels, samples) float32 @ 16kHz.
输出标准化情绪标签和置信度，供 AudioPerception 消费。
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import torch

from emotion.config import EmotionConfig
from emotion.models import EmotionType
from seld.exceptions import (
    InferenceError,
    InvalidAudioError,
    ModelLoadError,
)
from seld.logging_utils import get_logger

logger = get_logger(__name__)

# 可选依赖
try:
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    HAS_MODELSCOPE = True
except ImportError:
    HAS_MODELSCOPE = False


# ── 全局单例 ──────────────────────────────────────────────

_global_engine: Optional["EmotionEngine"] = None


def get_emotion_engine(config: Optional[EmotionConfig] = None) -> "EmotionEngine":
    """获取全局 EmotionEngine 单例."""
    global _global_engine
    if _global_engine is None:
        _global_engine = EmotionEngine(config or EmotionConfig())
    return _global_engine


# ── 引擎 ──────────────────────────────────────────────────


class EmotionEngine:
    """
    基于 emotion2vec 的语音情绪识别推理引擎。

    Usage:
        config = EmotionConfig()
        engine = EmotionEngine(config)
        label, conf = engine.infer(audio)  # audio: (4, 1600) float32
    """

    def __init__(self, config: EmotionConfig):
        config.validate()
        self.config = config
        self.device, self.device_str = self._resolve_device(config.device)
        self._pipeline = None
        self._inference_count: int = 0
        self._load_model()

    # ── 设备 ─────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: str) -> Tuple[torch.device, str]:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda"), "cuda:0"
            logger.warning("No GPU detected, EmotionEngine running on CPU")
            return torch.device("cpu"), "cpu"
        return torch.device(device), device

    # ── 模型加载 ─────────────────────────────────────────

    def _load_model(self) -> None:
        if not HAS_MODELSCOPE:
            raise ModelLoadError(
                "modelscope is not installed. Run: pip install modelscope"
            )

        errors = []

        # 策略 1: pipeline API
        try:
            logger.info("Loading emotion2vec via pipeline: %s", self.config.model_name)
            self._pipeline = pipeline(
                task=Tasks.emotion_recognition,
                model=self.config.model_name,
                device=self.device_str,
            )
            # warm-up
            _ = self._pipeline(np.zeros((16000,), dtype=np.float32))
            logger.info("Emotion2vec loaded (pipeline mode)")
            return
        except Exception as e:
            errors.append(f"pipeline: {e}")

        # 策略 2: 原生 Model API
        try:
            from modelscope.models.base import Model as ScopeModel
            from modelscope.preprocessors import Preprocessor

            logger.info("Falling back to native Model: %s", self.config.model_name)
            self._model = ScopeModel.from_pretrained(
                self.config.model_name, device=self.device_str
            )
            self._preprocessor = Preprocessor.from_pretrained(self.config.model_name)
            self._use_pipeline = False
            logger.info("Emotion2vec loaded (native model mode)")
            return
        except Exception as e:
            errors.append(f"native model: {e}")

        raise ModelLoadError(
            f"Could not load emotion2vec '{self.config.model_name}'.\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    # ── 预处理 ───────────────────────────────────────────

    @staticmethod
    def _validate(audio: np.ndarray) -> None:
        if not isinstance(audio, np.ndarray):
            raise InvalidAudioError(f"Expected np.ndarray, got {type(audio).__name__}")
        if audio.ndim != 2:
            raise InvalidAudioError(
                f"Expected 2D array (channels, samples), got {audio.ndim}D"
            )
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            raise InvalidAudioError("Audio contains NaN or inf")
        if audio.size == 0:
            raise InvalidAudioError("Audio array is empty")

    def preprocess(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """校验 → 多通道降混 → 静音检测 → 返回一维单声道."""
        self._validate(audio)
        if audio.shape[0] > 1:
            mono = np.mean(audio, axis=0, dtype=np.float32)
        else:
            mono = audio[0].astype(np.float32)
        rms = float(np.sqrt(np.mean(mono**2)))
        if rms < 1e-6:
            return None
        return mono

    # ── 后处理 ───────────────────────────────────────────

    def postprocess(
        self, raw_output: dict, top_k: int = 1
    ) -> Tuple[Optional[str], float]:
        """emotion2vec 输出 → 标准情绪标签 + 置信度."""
        if raw_output is None:
            return None, 0.0

        scores = raw_output.get("scores", [])
        labels = raw_output.get("labels", [])
        if not scores or not labels:
            return None, 0.0

        k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[-k:][::-1]
        best_score = float(scores[top_indices[0]])
        best_label = labels[top_indices[0]]

        # 中文 → 英文
        label_en = self._cn_to_en(best_label)

        if best_score < self.config.confidence_threshold:
            return EmotionType.NEUTRAL.value, round(best_score, 4)

        return label_en, round(best_score, 4)

    @staticmethod
    def _cn_to_en(label: str) -> str:
        mapping = {
            "开心": "happy", "高兴": "happy",
            "悲伤": "sad", "伤心": "sad",
            "生气": "angry", "愤怒": "angry",
            "中性": "neutral", "平静": "neutral",
            "恐惧": "fear", "害怕": "fear",
            "惊讶": "surprise", "吃惊": "surprise",
            "厌恶": "disgust",
            "痛苦": "pain",
        }
        return mapping.get(label, label.lower())

    # ── 推理 ─────────────────────────────────────────────

    def infer(
        self, audio: np.ndarray, top_k: int = 1
    ) -> Tuple[Optional[str], float]:
        """
        执行情绪识别推理.

        Args:
            audio: (channels, samples) float32 @ 16kHz.
            top_k: 保留 top-k 结果（默认 1）。

        Returns:
            (emotion_label, confidence): 英文标签 + 置信度 0~1。
        """
        t0 = time.perf_counter()
        try:
            mono = self.preprocess(audio)
            if mono is None:
                elapsed = (time.perf_counter() - t0) * 1000
                self._inference_count += 1
                return None, 0.0

            with torch.no_grad():
                raw = self._pipeline(mono)

            label, conf = self.postprocess(raw, top_k=top_k)
            elapsed = (time.perf_counter() - t0) * 1000
            self._inference_count += 1

            logger.debug(
                "Emotion #%d: %.1fms, %s (%.3f)",
                self._inference_count, elapsed, label or "none", conf,
            )
            return label, conf

        except (InvalidAudioError, ModelLoadError):
            raise
        except Exception as e:
            raise InferenceError(f"Emotion inference failed: {e}") from e

    @property
    def model_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def inference_count(self) -> int:
        return self._inference_count
