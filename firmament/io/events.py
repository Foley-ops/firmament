"""Append-only JSONL event log: seed placement (#0), operator actions, milestones, EDITs.

Events are part of a run's causal definition: (config, seed, event log) fully
determines the world. Replay reads the log and re-applies each event at its tick.
"""
from __future__ import annotations

import json
from pathlib import Path

from firmament.io.logging import CLOCK, get

log = get("events")


class EventLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.f = open(self.path, "a", buffering=1)
        self.count = sum(1 for _ in open(self.path)) if self.path.exists() else 0

    def append(self, event: str, tick: int, component: str = "operator", **fields) -> dict:
        rec = {"run_id": CLOCK.run_id, "tick": tick, "sim_time": CLOCK.sim_time(),
               "component": component, "level": "INFO", "event": event, "n": self.count,
               **fields}
        self.f.write(json.dumps(rec, default=str) + "\n")
        self.f.flush()
        self.count += 1
        log.info(f"event {event}", extra=fields | {"event_n": rec["n"]})
        return rec

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in open(self.path) if line.strip()]

    def flush(self) -> None:
        self.f.flush()

    def tail(self, n: int = 50) -> list[dict]:
        return self.read_all()[-n:]


def replay_events(sched, events_path: Path) -> None:
    """Queue logged operator events for re-application at their original ticks."""
    from firmament.operator import console

    pending = []
    for rec in EventLog(events_path).read_all():
        if rec.get("component") != "operator":
            continue                       # milestones re-emerge on their own
        if rec["tick"] < sched.state.tick:
            continue                       # already inside the snapshot
        pending.append(rec)
    pending.sort(key=lambda r: (r["tick"], r["n"]))
    sched.replay_queue = pending
    if pending:
        orig = sched.tick_once

        def tick_with_replay():
            while sched.replay_queue and sched.replay_queue[0]["tick"] == sched.state.tick:
                rec = sched.replay_queue.pop(0)
                console.apply_event(sched, rec, replaying=True)
            orig()
        sched.tick_once = tick_with_replay
    log.info("replay queue loaded", extra={"n_events": len(pending)})
