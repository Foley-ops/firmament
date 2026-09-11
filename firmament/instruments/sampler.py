"""Instrument sampler: runs at most once per metrics_every_ticks, on read-only host
copies of state (instruments never receive device arrays — enforced by construction:
they get a plain dict of numpy arrays, so they physically cannot write sim state).
"""
from __future__ import annotations

import numpy as np

from firmament.instruments import diversity, ecology, milestones, novelty, population
from firmament.io.logging import get
from firmament.io.metrics import MetricsWriter

log = get("instruments")


class ReadOnlyView(dict):
    """Host-side numpy copies; arrays are flagged non-writeable."""

    @classmethod
    def from_state(cls, s, poly=None):
        v = cls()
        for name in ("water_depth", "vapor", "temp", "light", "light_water", "species",
                     "compartment_id", "membrane_store", "p_id", "p_parent", "p_cell",
                     "p_len", "p_state", "p_partner", "p_motifs", "p_mut", "p_born",
                     "p_seq", "elevation", "sediment"):
            a = getattr(s, name, None)
            if a is not None:
                arr = a.numpy()
                arr.setflags(write=False)
                v[name] = arr
        v["tick"] = s.tick
        v["shape"] = s.shape
        v["recent_copies"] = list(poly.recent_events) if poly is not None else []
        return v


class Sampler:
    def __init__(self, sched, cfg, run_dir) -> None:
        self.sched = sched
        self.cfg = cfg
        self.every = cfg.run.metrics_every_ticks
        self.writer = MetricsWriter(run_dir / "metrics")
        sched.metrics = self.writer
        self.mile = milestones.Milestones(sched)
        self.nov = novelty.Novelty(sched)
        self.seed_hash = None
        self.first_copy_seen = False

    def __call__(self, s, tick: int) -> None:
        if tick % self.every != 0:
            return
        poly = getattr(self.sched, "poly", None)
        view = ReadOnlyView.from_state(s, poly)
        self.sched.latest_view = view                      # GUI reads this cache
        row = {"tick": tick}
        row |= population.sample(view)
        row |= diversity.sample(view)
        row |= ecology.sample(view, self.cfg)
        row |= self.nov.sample(view)
        row |= _environment(view)
        self.writer.add(row)
        self.mile.check(view, row)
        if tick % (self.every * 100) == 0:
            self.writer.flush()


def _environment(v) -> dict:
    t = v["temp"]
    wet = v["water_depth"] > 1e-3
    return {
        "temp_air_mean": float(t[0].mean()), "temp_surf_mean": float(t[1].mean()),
        "temp_surf_max": float(t[1].max()), "temp_surf_min": float(t[1].min()),
        "water_total_m3": float(v["water_depth"].sum()), "vapor_total_kg": float(v["vapor"].sum()),
        "wet_fraction": float(wet.mean()), "light_mean": float(v["light"].mean()),
    }


def attach_instruments(sched, cfg, run_dir) -> None:
    sched.instruments.append(Sampler(sched, cfg, run_dir))
