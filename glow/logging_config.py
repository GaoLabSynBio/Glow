"""Central logging configuration for Glow.

Provides ``get_logger(name)`` returning a logger under the ``glow`` namespace.
Logs go to a rotating file in the user data directory (so packaged desktop apps,
which have no console, can still produce a log to attach to bug reports) and to
stderr for developers running from a terminal.
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

_CONFIGURED = False


def configure(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    logger = logging.getLogger("glow")
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    # File handler (best effort -- never let logging setup crash the app).
    try:
        from .paths import logs_dir

        file_handler = RotatingFileHandler(
            logs_dir() / "glow.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except Exception:
        pass

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    logger.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring handlers on first use."""
    configure()
    return logging.getLogger(f"glow.{name}")
