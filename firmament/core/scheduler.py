"""Tick loop, time modes, module ordering. Phase 0: no-op physics; modules attach later."""
from __future__ import annotations

import time

from firmament.io.logging import CLOCK, get

log = get("scheduler")

MODES = ("unbounded", "throttled", "paused", "step")


class Scheduler:
    """Runs modules in fixed order each tick: radiation → thermal → fluid → chemistry
    → polymers → audit. Modules are callables f(state, tick) appended by each phase."""

    def __init__(self, state, cfg, run) -> None:
        self.state = state
        self.cfg = cfg
        self.run = run
        self.modules: list = []
        self.instruments: list = []          # read-only observers, may not touch state
        self.pending_events: list = []       # operator events applied on tick boundary
        self.mode = "unbounded"
        self.target_tps = 30.0
        self.step_budget = 0
        self.dt = cfg.run.dt_seconds
        self.stop_flag = False

    def add(self, fn) -> None:
        self.modules.append(fn)

    def tick_once(self) -> None:
        s = self.state
        while self.pending_events:                      # events apply on tick boundary
            self.pending_events.pop(0)(s, s.tick)
        for fn in self.modules:
            fn(s, s.tick)
        s.tick += 1
        CLOCK.tick = s.tick
        CLOCK.sim_seconds = s.tick * self.dt
        for fn in self.instruments:
            fn(s, s.tick)

    def loop(self, max_ticks: int | None = None) -> None:
        t0, tick0 = time.perf_counter(), self.state.tick
        last_report = t0
        while not self.stop_flag:
            if max_ticks is not None and self.state.tick - tick0 >= max_ticks:
                break
            if self.mode == "paused":
                time.sleep(0.05)
                continue
            if self.mode == "step":
                if self.step_budget <= 0:
                    self.mode = "paused"
                    continue
                self.step_budget -= 1
            self.tick_once()
            if self.mode == "throttled":
                elapsed = time.perf_counter() - t0
                ahead = (self.state.tick - tick0) / self.target_tps - elapsed
                if ahead > 0:
                    time.sleep(min(ahead, 0.1))
            now = time.perf_counter()
            if now - last_report >= 10.0:
                self.tps = (self.state.tick - tick0) / (now - t0)
                log.info("throughput", extra={"ticks_per_sec": round(self.tps, 2)})
                last_report = now
        tps = (self.state.tick - tick0) / max(time.perf_counter() - t0, 1e-9)
        log.info("loop done", extra={"ticks": self.state.tick - tick0, "ticks_per_sec": round(tps, 2)})
