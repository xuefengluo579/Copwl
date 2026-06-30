"""w2v-SELD 核心推理引擎."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchaudio

from seld.config import SELDConfig
from seld.exceptions import (
    ArchitectureError,
    InferenceError,
    InvalidAudioError,
    ModelLoadError,
    WeightNotFoundError,
)
from seld.logging_utils import get_logger
from seld.audio import validate_audio
from seld.models import DetectedEvent, DOAVector, EventType

logger = get_logger(__name__)

# 尝试导入 fairseq（可选依赖）
try:
    from fairseq.models.wav2vec import Wav2Vec2Model

    HAS_FAIRSEQ = True
except ImportError:
    HAS_FAIRSEQ = False
    Wav2Vec2Model = None
    logger.warning("fairseq not installed; fairseq-based model loading will fail")


class W2vSELDEngine:
    """
    w2v-SELD 声音事件检测推理引擎。

    封装模型加载、音频预处理、推理和后处理流水线。
    支持 GPU (CUDA)、Apple Silicon (MPS) 和 CPU 推理。

    Usage:
        config = SELDConfig.from_yaml("config/default.yaml")
        engine = W2vSELDEngine(config)
        events = engine.infer(audio_array)  # shape (4, samples)
    """

    def __init__(self, config: SELDConfig):
        """
        初始化推理引擎。

        Args:
            config: SELDConfig 配置实例。

        Raises:
            ModelLoadError: 模型加载失败。
            ConfigurationError: 配置不合法。
        """
        config.validate()
        self.config = config
        self.model_cfg = config.model
        self.audio_cfg = config.audio
        self.infer_cfg = config.inference

        # 设备选择
        self.device = self._resolve_device(self.model_cfg.device)
        logger.info("Using device: %s", self.device)

        # 模型加载
        self.model: Optional[torch.nn.Module] = None
        self._load_model()
        self.model.eval()

        # 后处理参数
        self.threshold = self.infer_cfg.event_threshold
        self.min_duration_ms = self.infer_cfg.min_event_duration_ms
        self.frame_ms = self.audio_cfg.frame_duration_ms

        # 统计
        self._inference_count: int = 0

    # ── 设备管理 ───────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """
        解析设备字符串，支持 "auto" 自动选择。

        优先级: CUDA > MPS > CPU
        """
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    # ── 模型加载 ───────────────────────────────────────────

    def _load_model(self) -> None:
        """
        加载模型权重。

        策略 1: fairseq Wav2Vec2Model.from_pretrained（标准路径）
        策略 2: torch.load state_dict 直接加载（兼容自定义 checkpoint）

        Raises:
            ModelLoadError: 所有策略均失败。
        """
        errors: list[str] = []
        checkpoint = self.model_cfg.checkpoint_path
        model_path = self.model_cfg.model_path

        # 检查 checkpoint 文件
        if not os.path.exists(checkpoint):
            raise WeightNotFoundError(
                f"Checkpoint not found: {checkpoint}\n"
                f"Run 'python scripts/download_weights.py' to download pre-trained weights."
            )

        # 策略 1: fairseq from_pretrained
        if self.model_cfg.use_fairseq and HAS_FAIRSEQ:
            try:
                logger.info("Loading model via fairseq from_pretrained: %s", model_path)
                self.model = Wav2Vec2Model.from_pretrained(
                    model_path,
                    checkpoint_file=checkpoint,
                ).to(self.device)
                logger.info(
                    "Model loaded successfully (fairseq). Params: %s",
                    f"{sum(p.numel() for p in self.model.parameters()):,}",
                )
                return
            except Exception as e:
                errors.append(f"fairseq API: {e}")
                logger.warning("fairseq loading failed: %s", e)

        # 策略 2: 直接 state_dict 加载
        if HAS_FAIRSEQ:
            try:
                logger.info("Loading model via direct state_dict: %s", checkpoint)
                self.model = Wav2Vec2Model.from_pretrained(model_path).to(self.device)
                state = torch.load(checkpoint, map_location=self.device, weights_only=True)
                missing, unexpected = self.model.load_state_dict(state, strict=False)
                if missing:
                    logger.warning("Missing keys in checkpoint: %s", missing[:5])
                if unexpected:
                    logger.warning("Unexpected keys in checkpoint: %s", unexpected[:5])
                logger.info(
                    "Model loaded (direct state_dict, strict=False). "
                    "Params: %s",
                    f"{sum(p.numel() for p in self.model.parameters()):,}",
                )
                return
            except Exception as e:
                errors.append(f"direct state_dict: {e}")
                logger.warning("Direct state_dict loading failed: %s", e)

        raise ModelLoadError(
            f"Could not load model from {checkpoint}.\n"
            f"Errors:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    # ── 预处理 ─────────────────────────────────────────────

    def preprocess(self, audio: np.ndarray) -> torch.Tensor:
        """
        音频预处理流水线：校验 → 标准化 → 转 Tensor。

        Args:
            audio: 输入音频，shape (channels, samples)，float32。

        Returns:
            shape (1, channels, samples) 的标准化 Tensor，已在目标设备上。
        """
        validate_audio(
            audio,
            expected_channels=self.audio_cfg.num_channels,
            allow_silence=True,  # 静音允许通过，由后处理决定是否检出
        )

        tensor = torch.from_numpy(audio).float()

        # 标准化：零均值，单位方差（逐通道）
        mean = tensor.mean(dim=1, keepdim=True)
        std = tensor.std(dim=1, keepdim=True)
        tensor = (tensor - mean) / (std + 1e-8)

        # 添加 batch 维度
        return tensor.unsqueeze(0).to(self.device)

    # ── 后处理 ─────────────────────────────────────────────

    def postprocess(
        self,
        sed_logits: torch.Tensor,
        doa_output: Optional[torch.Tensor] = None,
    ) -> List[DetectedEvent]:
        """
        解析模型输出为结构化事件列表。

        Args:
            sed_logits: SED 分支输出，shape (batch, frames, num_classes)。
            doa_output: DOA 分支输出，shape (batch, frames, 3)，可选。

        Returns:
            检测到的事件列表。
        """
        probs = torch.sigmoid(sed_logits)  # → 概率

        events: List[DetectedEvent] = []

        for b_idx in range(probs.shape[0]):
            frame_probs = probs[b_idx]  # (frames, num_classes)

            for class_idx in range(frame_probs.shape[1]):
                prob_seq = frame_probs[:, class_idx]  # (frames,)
                activated = (prob_seq > self.threshold).cpu().numpy()

                # 寻找连续激活区间
                start: Optional[int] = None
                for i, active in enumerate(activated):
                    if active and start is None:
                        start = i
                    if (not active) and start is not None:
                        self._add_event(
                            events, class_idx, prob_seq, start, i,
                            doa_output, b_idx,
                        )
                        start = None

                # 处理帧末仍在激活的事件
                if start is not None:
                    self._add_event(
                        events, class_idx, prob_seq, start, len(activated),
                        doa_output, b_idx,
                    )

        # 按时间排序并过滤过短事件
        events.sort(key=lambda e: e.t_start_ms)
        events = [
            e for e in events
            if e.duration_ms >= self.min_duration_ms
        ]

        return events

    def _add_event(
        self,
        events: List[DetectedEvent],
        class_idx: int,
        prob_seq: torch.Tensor,
        frame_start: int,
        frame_end: int,
        doa_output: Optional[torch.Tensor],
        batch_idx: int,
    ) -> None:
        """构建单个 DetectedEvent 并加入列表."""
        t_start = frame_start * self.frame_ms
        t_end = frame_end * self.frame_ms
        max_conf = float(prob_seq[frame_start:frame_end].max().cpu())

        event_type = EventType.from_index(class_idx)

        # DOA：取激活区间的均值，并 clamp 至 [-1, 1]
        doa = None
        if doa_output is not None:
            doa_frames = doa_output[batch_idx, frame_start:frame_end]  # (n_frames, 3)
            doa_mean = doa_frames.mean(dim=0).clamp(-1.0, 1.0).cpu()  # (3,)
            doa = DOAVector(
                x=float(doa_mean[0]),
                y=float(doa_mean[1]),
                z=float(doa_mean[2]),
            )

        events.append(DetectedEvent(
            event=event_type,
            confidence=round(max_conf, 4),
            t_start_ms=t_start,
            t_end_ms=t_end,
            doa=doa,
        ))

    # ── 推理 ───────────────────────────────────────────────

    def infer(self, audio: np.ndarray) -> Tuple[List[DetectedEvent], float]:
        """
        执行完整推理流水线。

        Args:
            audio: 输入音频，shape (channels, samples)，float32。

        Returns:
            (events, elapsed_ms): 检测到的事件列表和推理耗时（毫秒）。

        Raises:
            InvalidAudioError: 输入校验失败。
            InferenceError: 推理运行时错误。
        """
        t0 = time.perf_counter()

        try:
            # 1. 预处理
            input_tensor = self.preprocess(audio)

            # 2. 模型推理
            with torch.no_grad():
                # w2v-SELD 模型的 forward 返回 (sed_logits, doa_logits)
                # 实际调用签名取决于具体实现
                outputs = self.model(source=input_tensor)
                if isinstance(outputs, dict):
                    sed_output = outputs.get("sed", outputs.get("logits"))
                    doa_output = outputs.get("doa")
                elif isinstance(outputs, (tuple, list)):
                    sed_output = outputs[0]
                    doa_output = outputs[1] if len(outputs) > 1 else None
                else:
                    sed_output = outputs
                    doa_output = None

            # 3. 后处理
            events = self.postprocess(sed_output, doa_output)

            elapsed = (time.perf_counter() - t0) * 1000
            self._inference_count += 1

            logger.debug(
                "Inference #%d: %.1f ms, %d events detected (input: %s)",
                self._inference_count,
                elapsed,
                len(events),
                input_tensor.shape,
            )

            if events:
                for e in events:
                    logger.info(
                        "  Event: %-15s conf=%.3f t=[%5d..%5d]ms",
                        e.event.value,
                        e.confidence,
                        e.t_start_ms,
                        e.t_end_ms,
                    )

            return events, elapsed

        except (InvalidAudioError, ModelLoadError):
            raise
        except Exception as e:
            raise InferenceError(
                f"Inference failed: {e}",
                details={"audio_shape": audio.shape, "audio_dtype": str(audio.dtype)},
            ) from e

    # ── 属性 ───────────────────────────────────────────────

    @property
    def model_loaded(self) -> bool:
        """模型是否已加载."""
        return self.model is not None

    @property
    def inference_count(self) -> int:
        """累计推理次数."""
        return self._inference_count

    def __repr__(self) -> str:
        return (
            f"W2vSELDEngine("
            f"device={self.device}, "
            f"threshold={self.threshold}, "
            f"inferences={self._inference_count})"
        )
