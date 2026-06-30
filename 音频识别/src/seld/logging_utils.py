"""w2v-SELD 日志配置工具."""

import logging
import logging.config
import os
from pathlib import Path

import yaml

DEFAULT_LOGGER_NAME = "seld"
_logger: logging.Logger | None = None


def get_logger(name: str = DEFAULT_LOGGER_NAME) -> logging.Logger:
    """获取模块日志器实例."""
    return logging.getLogger(name)


def setup_logging(
    config_path: str | Path | None = None,
    *,
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    log_dir: str | Path = "logs",
) -> logging.Logger:
    """
    配置模块日志系统。

    优先级:
        1. YAML 配置文件（如果提供且存在）
        2. 编程方式配置（log_file + level）

    Args:
        config_path: logging YAML 配置文件路径。
        level: 默认日志级别。
        log_file: 日志文件路径。为 None 时仅输出到控制台。
        log_dir: 日志目录（创建文件日志时使用）。

    Returns:
        配置好的 logger 实例。
    """
    global _logger

    if config_path and os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        log_dir_ = config_dict.get("log_dir", str(log_dir))
        os.makedirs(log_dir_, exist_ok=True)
        logging.config.dictConfig(config_dict)
    else:
        os.makedirs(log_dir, exist_ok=True)

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        root = logging.getLogger(DEFAULT_LOGGER_NAME)
        root.setLevel(level)

        # 控制台 handler
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(formatter)
        root.addHandler(console)

        # 文件 handler（如果指定了 log_file）
        if log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

    _logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    _logger.info("Logging system initialized (level=%s)", logging.getLevelName(level))
    return _logger
