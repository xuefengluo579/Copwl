"""w2v-SELD FastAPI 服务.

启动:
    seld-api                    # 通过 pip 安装后的 CLI 入口
    python -m seld.api           # 直接运行模块
    uvicorn seld.api:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from seld.config import SELDConfig
from seld.engine import W2vSELDEngine
from seld.exceptions import (
    SELDError,
    InvalidAudioError,
    ModelLoadError,
    WeightNotFoundError,
    ChannelMismatchError,
    InferenceError,
    ConfigurationError,
)
from seld.logging_utils import setup_logging, get_logger
from seld.models import (
    DetectionResult,
    DetectedEvent,
    HealthStatus,
)
from seld.audio import audio_from_bytes

# ── 应用初始化 ────────────────────────────────────────────

logger = get_logger(__name__)

# 全局引擎实例
engine: W2vSELDEngine | None = None
_startup_time: float = 0.0

# 配置文件路径（可通过环境变量覆盖）
CONFIG_PATH = os.environ.get("SELD_CONFIG_PATH", "config/default.yaml")
LOGGING_CONFIG_PATH = os.environ.get("SELD_LOGGING_CONFIG_PATH", "config/logging.yaml")


# ── 生命周期管理 ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时加载模型，关闭时释放资源."""
    global engine, _startup_time

    # 启动
    setup_logging(
        config_path=LOGGING_CONFIG_PATH if os.path.exists(LOGGING_CONFIG_PATH) else None,
        level=os.environ.get("SELD_LOG_LEVEL", "INFO"),  # type: ignore
    )
    logger.info("=== w2v-SELD API Server starting ===")
    logger.info("Config: %s", CONFIG_PATH)

    try:
        config = SELDConfig.from_yaml(CONFIG_PATH)
        engine = W2vSELDEngine(config)
        logger.info("Engine initialized successfully: %s", engine)
    except WeightNotFoundError:
        logger.error("Model weights not found. Run: python scripts/download_weights.py")
        engine = None
    except Exception as e:
        logger.error("Failed to initialize engine: %s", e)
        engine = None

    _startup_time = time.time()
    yield

    # 关闭
    engine = None
    logger.info("=== w2v-SELD API Server stopped ===")


app = FastAPI(
    title="w2v-SELD Audio Event Detection",
    description="基于 wav2vec 2.0 的声音事件检测与定位 API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ── 全局异常处理 ──────────────────────────────────────────

@app.exception_handler(SELDError)
async def seld_error_handler(request, exc: SELDError):
    """将模块自定义异常映射为 HTTP 响应."""
    status_code = 422 if isinstance(exc, InvalidAudioError) else 500
    logger.error("[%d] %s: %s", status_code, exc.__class__.__name__, exc)
    return JSONResponse(
        status_code=status_code,
        content={
            "error": exc.__class__.__name__,
            "message": str(exc),
            "details": exc.details,
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    """捕获 Pydantic 验证错误."""
    logger.warning("Validation error: %s", exc)
    return JSONResponse(
        status_code=422,
        content={"error": "ValidationError", "message": str(exc)},
    )


# ── API 端点 ──────────────────────────────────────────────

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """服务健康检查."""
    global engine, _startup_time

    if engine is None:
        return HealthStatus(
            status="degraded",
            model_loaded=False,
            device="unknown",
            version=__import__("seld").__version__,
            uptime_seconds=round(time.time() - _startup_time, 1),
            model_name=None,
        )

    try:
        return HealthStatus(
            status="ok",
            model_loaded=engine.model_loaded,
            device=str(engine.device),
            version=__import__("seld").__version__,
            uptime_seconds=round(time.time() - _startup_time, 1),
            model_name=os.path.basename(engine.model_cfg.checkpoint_path),
        )
    except Exception as e:
        return HealthStatus(
            status="error",
            model_loaded=False,
            device="unknown",
            version=__import__("seld").__version__,
            uptime_seconds=round(time.time() - _startup_time, 1),
            model_name=str(e),
        )


@app.post(
    "/v1/audio/events",
    response_model=DetectionResult,
    summary="检测音频事件",
    description="上传音频文件（WAV/FLAC/OGG），返回检测到的声音事件列表。",
)
async def detect_events(file: UploadFile = File(..., description="音频文件")):
    """
    音频事件检测端点。

    支持格式: WAV, FLAC, OGG
    要求: 采样率 16kHz（自动重采样），支持单通道~四通道

    Returns:
        DetectionResult: 包含检测到的事件列表、推理耗时等信息。
    """
    global engine

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Engine not initialized. Model weights may be missing.",
        )

    # 验证文件
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    if not file.content_type or "audio" not in file.content_type:
        # 不严格拒绝，尝试解码
        logger.debug("Non-audio content_type: %s, attempting decode", file.content_type)

    # 读取并解码
    try:
        audio_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if not audio_bytes or len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        audio = audio_from_bytes(
            audio_bytes,
            target_sample_rate=engine.audio_cfg.sample_rate,
            target_channels=engine.audio_cfg.num_channels,
        )
    except ChannelMismatchError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Audio channel error: {e}. Required: {engine.audio_cfg.num_channels} channels.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to decode audio: {e}. Supported formats: WAV, FLAC, OGG.",
        )

    # 大小限制检查（防止超大文件 OOM）
    max_samples = 30 * engine.audio_cfg.sample_rate  # 30 秒
    if audio.shape[1] > max_samples:
        raise HTTPException(
            status_code=413,
            detail=f"Audio too long: {audio.shape[1] / engine.audio_cfg.sample_rate:.1f}s. "
            f"Maximum: 30 seconds.",
        )

    # 推理
    try:
        events, elapsed_ms = engine.infer(audio)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {e}",
        )

    audio_duration_ms = int(audio.shape[1] / engine.audio_cfg.sample_rate * 1000)

    return DetectionResult(
        events=events,
        inference_time_ms=round(elapsed_ms, 2),
        audio_duration_ms=audio_duration_ms,
    )


# ── CLI 入口 ──────────────────────────────────────────────

def main():
    """CLI 入口: seld-api 或 python -m seld.api."""
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="w2v-SELD API Server")
    parser.add_argument(
        "--host", default="0.0.0.0", help="绑定地址 (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="绑定端口 (default: 8080)"
    )
    parser.add_argument(
        "--reload", action="store_true", help="启用热重载（开发模式）"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="工作进程数 (default: 1)"
    )
    args = parser.parse_args()

    uvicorn.run(
        "seld.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
