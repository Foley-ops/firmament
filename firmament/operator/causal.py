"""Causal history: the ONLY path by which anything outside physics changes the world.

Commands (natural events, seed placement, developer edits) are submitted to a mailbox
and COMMITTED by the sim thread at a tick boundary: each gets the next causal sequence
number (state.causal_n), is written to events.jsonl with tick = state.tick at
application, and is applied. Snapshots store causal_n, so resume/replay/fork re-apply
exactly the logged records with n >= causal_n when the world reaches their tick.

Analysis output (milestones, novelty) lives in analysis.jsonl with its own counter and
never feeds any RNG key, so observation can never become causal.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from firmament.io.logging import get

log = get("causal")

NATURAL = ("rain", "drought", "flood", "earthquake", "volcano", "meteor",
           "climate", "solar", "add_land", "add_species")
COMPONENT = {"seed_placed": "operator", "EDIT": "developer"} | {k: "operator" for k in NATURAL}

_lock = threading.Lock()


def submit(sched, kind: str, params: dict) -> None:
    """Queue a command from any thread; it is committed at the next tick boundary."""
    if kind not in COMPONENT:
        raise ValueError(f"unknown causal command: {kind}")
    with _lock:
        sched.mailbox.append((kind, dict(params)))


def commit_now(sched, kind: str, params: dict) -> dict:
    """Log and apply a command immediately. Only legal at a tick boundary (sim thread
    between ticks, or offline tools such as `firmament seed`)."""
    if kind not in COMPONENT:
        raise ValueError(f"unknown causal command: {kind}")
    if getattr(sched, "replay_queue", None):
        raise RuntimeError("cannot commit a new command while logged history is replaying")
    s = sched.state
    if sched.events.count != s.causal_n:
        raise RuntimeError(f"causal log has {sched.events.count} records but state has applied "
                           f"{s.causal_n} — refusing to fork history silently")
    rec = sched.events.append(kind, s.tick, component=COMPONENT[kind], n=s.causal_n,
                              params=params)
    _apply(sched, rec, live=True)
    return rec


def load_replay(sched, events_path: Path) -> int:
    """Queue every logged command not yet contained in the current state."""
    recs = []
    if Path(events_path).exists():
        recs = [json.loads(x) for x in open(events_path) if x.strip()]
    recs = sorted((r for r in recs if r.get("n", -1) >= sched.state.causal_n), key=lambda r: r["n"])
    for want, r in enumerate(recs, start=sched.state.causal_n):
        if r["n"] != want:
            raise RuntimeError(f"causal log has a gap: expected n={want}, found {r['n']}")
    sched.replay_queue = recs
    log.info("replay queue loaded", extra={"n_events": len(recs), "from_n": sched.state.causal_n})
    return len(recs)


def apply_due(sched) -> None:
    """Called by the scheduler at every tick boundary: replayed history first, then
    live commands (held until replay has caught up, so n order == application order)."""
    s = sched.state
    q = getattr(sched, "replay_queue", None)
    while q and q[0]["tick"] == s.tick:
        _apply(sched, q.pop(0), live=False)
    if q:
        if q[0]["tick"] < s.tick:
            raise RuntimeError(f"replay missed causal event n={q[0]['n']} at tick {q[0]['tick']}")
        return
    with _lock:
        pending, sched.mailbox[:] = list(sched.mailbox), []
    for kind, params in pending:
        commit_now(sched, kind, params)


def _apply(sched, rec: dict, live: bool) -> None:
    from firmament.operator import console, developer
    kind, params = rec["event"], rec["params"]
    if kind == "seed_placed":
        console.apply_seed(sched, params)
    elif kind == "EDIT":
        developer.apply_edit(sched, params)
        if live:
            developer.mark_touched(sched, params)
    else:
        console.apply_natural(sched, kind, params, rec["n"], rec["tick"])
    sched.state.causal_n += 1
