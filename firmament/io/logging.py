"""Structured JSONL logging. Every record: run_id, tick, sim_time, component, level, msg.

The tick/sim_time are injected from a context object the scheduler advances, so any
component logging anywhere is greppable by run_id and tick.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import time
from pathlib import Path

_STD = {"name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
        "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
        "relativeCreated", "thread", "threadName", "processName", "process", "taskName",
        "message"}


class SimClock:
    """Mutable clock context shared with the log formatter."""

    def __init__(self) -> None:
        self.run_id = "?"
        self.tick = 0
        self.sim_seconds = 0.0

    def sim_time(self) -> str:
        s = int(self.sim_seconds)
        y, s = divmod(s, 365 * 86400)
        d, s = divmod(s, 86400)
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        return f"{y}y {d}d {h:02d}:{m:02d}:{s:02d}"


CLOCK = SimClock()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "run_id": CLOCK.run_id,
            "tick": CLOCK.tick,
            "sim_time": CLOCK.sim_time(),
            "component": record.name.removeprefix("firmament."),
            "level": record.levelname,
            "msg": record.getMessage(),
            "wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        for k, v in record.__dict__.items():
            if k not in _STD and not k.startswith("_"):
                out[k] = v
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def setup(log_dir: str | Path, level: str = "INFO") -> logging.Logger:
    """Configure the root 'firmament' logger: rotating sim.log + stderr, JSONL."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("firmament")
    root.setLevel(level)
    root.handlers.clear()
    fh = logging.handlers.RotatingFileHandler(
        Path(log_dir) / "sim.log", maxBytes=64 * 2**20, backupCount=10)
    fh.setFormatter(JsonFormatter())
    root.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(JsonFormatter())
    root.addHandler(sh)
    return root


def set_level(level: str) -> None:
    """Runtime-adjustable level (also exposed via the API)."""
    logging.getLogger("firmament").setLevel(level)


def get(component: str) -> logging.Logger:
    return logging.getLogger(f"firmament.{component}")
