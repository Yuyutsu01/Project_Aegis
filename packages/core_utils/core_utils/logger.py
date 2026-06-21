import logging
import os
import sys
from datetime import datetime

from pythonjsonlogger import jsonlogger


def get_logger(name: str = "ProjectAegis", log_level: str = "INFO") -> logging.Logger:
    """Enterprise JSON logger configuration"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level.upper()))

    # Check if we are running in docker or local console
    is_k8s_or_docker = os.environ.get("IS_CONTAINER", "false").lower() in ("true", "1")

    stream_handler = logging.StreamHandler(sys.stdout)

    if is_k8s_or_docker:
        # Enforce JSON logs in containers
        formatter: logging.Formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
        )
    else:
        # User-friendly console logs for local dev
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Also log to a persistent file
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    file_log_path = os.path.join(log_dir, f"aegis_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(file_log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
