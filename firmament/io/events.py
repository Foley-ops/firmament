"""Append-only JSONL logs.

Two separate logs with separate counters (docs/DECISIONS.md, 2026-09-25):
- events.jsonl   — CAUSAL history (seed, natural events, developer EDITs). Together with
                   the frozen config and seed it fully determines the world.
- analysis.jsonl — DERIVED observations (milestones, novelty). Never read by physics.
"""
from __future__ import annotations

import json
from pathlib import Path

from firmament.io.logging import CLOCK, get

log = get("events")


class EventLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.touch(exist_ok=True)
        self.count = sum(1 for x in open(self.path) if x.strip())
        self.f = open(self.path, "a", buffering=1)

    def append(self, event: str, tick: int, component: str = "operator", n: int | None = None,
               **fields) -> dict:
        n = self.count if n is None else n
        rec = {"run_id": CLOCK.run_id, "tick": tick, "sim_time": CLOCK.sim_time(),
               "component": component, "level": "INFO", "event": event, "n": n, **fields}
        self.f.write(json.dumps(rec, default=str) + "\n")
        self.f.flush()
        self.count += 1
        log.info(f"event {event}", extra={"event_n": n, "log": self.path.name})
        return rec

    def read_all(self) -> list[dict]:
        return [json.loads(line) for line in open(self.path) if line.strip()]

    def flush(self) -> None:
        self.f.flush()

    def tail(self, n: int = 50) -> list[dict]:
        return self.read_all()[-n:]

    def truncate_after(self, tick: int) -> int:
        """Drop records newer than `tick` (used on resume: the resumed process regenerates
        them deterministically, so keeping them would duplicate history)."""
        keep = [r for r in self.read_all() if r["tick"] <= tick]
        dropped = self.count - len(keep)
        self.f.close()
        with open(self.path, "w") as f:
            for r in keep:
                f.write(json.dumps(r, default=str) + "\n")
        self.count = len(keep)
        self.f = open(self.path, "a", buffering=1)
        return dropped


def replay_events(sched, events_path: Path) -> int:
    """Queue the causal history not yet contained in the current state for replay."""
    from firmament.operator import causal
    return causal.load_replay(sched, events_path)
