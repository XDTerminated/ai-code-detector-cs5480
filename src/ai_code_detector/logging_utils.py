"""Centralized logging setup.

Scripts call :func:`configure_logging` once at start-up so every module uses the
same format and level. The default format embeds a timestamp and logger name,
which is invaluable when stitching together logs from multiple pipeline stages.
"""

from __future__ import annotations

import logging
import sys

_DEFAULT_FORMAT: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATEFMT: str = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a consistent, readable format.

    Idempotent: repeated calls will not stack handlers.

    Args:
        level: Minimum log level to emit. Defaults to INFO.
    """
    root = logging.getLogger()
    if root.handlers:
        # Already configured (e.g., by a parent script). Align the level and
        # leave existing handlers in place.
        root.setLevel(level)
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
    root.addHandler(handler)
    root.setLevel(level)

    # Silence noisy third-party loggers that otherwise flood the console.
    for noisy in ("urllib3", "filelock", "fsspec", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
