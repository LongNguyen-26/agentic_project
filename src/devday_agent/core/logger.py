import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from devday_agent.config import config


def setup_logging(level: Optional[str] = None) -> None:
    """Configure root logging once for console and rotating file output."""
    root = logging.getLogger()
    if root.handlers:
        return

    resolved_level = (level or config.LOG_LEVEL or "INFO").upper()
    root.setLevel(getattr(logging, resolved_level, logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    log_path = Path(config.LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
