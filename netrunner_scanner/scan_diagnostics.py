"""Lightweight runtime diagnostics for scan pipeline debugging.

The intent is to capture why automatic scanning did or did not run without
requiring a debugger.  This file is deliberately dependency-light and safe to
call from worker threads.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGER = None
_LOG_PATH = None
_INIT_LOCK = threading.Lock()
_LAST = {}


def _default_log_path() -> Path:
    env_path = os.environ.get("NETREADYEYES_LOG")
    if env_path:
        return Path(env_path)
    return Path.cwd() / "logs" / "netreadyeyes_scan_debug.log"


def init_diagnostics() -> Path:
    global _LOGGER, _LOG_PATH
    with _INIT_LOCK:
        if _LOGGER is not None:
            return _LOG_PATH

        _LOG_PATH = _default_log_path()
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger("netreadyeyes.scan")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.handlers.clear()

        handler = RotatingFileHandler(
            _LOG_PATH,
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(levelname)s] [%(threadName)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)

        _LOGGER = logger
        atexit.register(close_diagnostics)
        log("diagnostics_started", log_path=str(_LOG_PATH), cwd=str(Path.cwd()), python=sys.version.split()[0])
        return _LOG_PATH


def log(event: str, **fields) -> None:
    logger = _LOGGER
    if logger is None:
        init_diagnostics()
        logger = _LOGGER
    try:
        payload = {"event": event, **fields}
        logger.info(json.dumps(payload, default=str, sort_keys=True))
    except Exception:
        # Diagnostics must never break scanning.
        pass


def log_exception(event: str, exc: BaseException | None = None, **fields) -> None:
    logger = _LOGGER
    if logger is None:
        init_diagnostics()
        logger = _LOGGER
    try:
        payload = {"event": event, **fields}
        if exc is not None:
            payload["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception(json.dumps(payload, default=str, sort_keys=True))
    except Exception:
        pass


def log_throttled(key: str, seconds: float, event: str, **fields) -> None:
    now = time.monotonic()
    last = float(_LAST.get(key, 0.0))
    if now - last < seconds:
        return
    _LAST[key] = now
    log(event, **fields)


def close_diagnostics() -> None:
    logger = _LOGGER
    if logger is None:
        return
    try:
        log("diagnostics_stopping")
        for handler in list(logger.handlers):
            handler.flush()
            handler.close()
            logger.removeHandler(handler)
    except Exception:
        pass
