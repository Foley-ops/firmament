"""Milestone detectors: emit events, never act. Each fires once per run.

Detectors are pure functions of the read-only view (plus small detector state),
so they can be tested on synthetic fixtures (Phase 6 tests).
"""
from __future__ import annotations

import numpy as np

from firmament.instruments import diversity
from firmament.io.logging import get

log = get("milestones")


class Milestones:
    def __init__(self, sched) -> None:
        self.sched = sched
        self.fired: set[str] = set()
        self.prev_bound: dict[int, int] = {}      # polymer id -> partner id
        self.prev_comp_share = 0.0
        self.signal_hits = 0
        self.nonmet_hits = 0

    def _emit(self, tick: int, name: str, **fields) -> None:
        if name in self.fired:
            return
        self.fired.add(name)
        ev = getattr(self.sched, "events", None)
        if ev:
            ev.append(name, tick, component="milestones", **fields)
        log.info(f"MILESTONE {name}", extra=fields)

    def check(self, v, row: dict) -> None:
        tick = v["tick"]
        alive = np.nonzero(v["p_state"] > 0)[0]
        n = len(alive)

        if v["recent_copies"]:
            e = v["recent_copies"][0]
            self._emit(tick, "first_copy", child_id=int(e[0]), parent_id=int(e[1]),
                       cell=int(e[2]), length=int(e[3]))

        if n:
            # fixation: one variant >90% of a real population
            hashes = diversity.seq_hashes(v)
            uniq, counts = np.unique(hashes, return_counts=True)
            if n >= 50 and counts.max() / n > 0.9 and len(uniq) > 1:
                self._emit(tick, "fixation", share=float(counts.max() / n), population=n)

            # mutant lineage surviving ~100 generations (proxy: mutated polymer with
            # lineage depth >= 100; depth from the lineage DB)
            if "mutant_100_generations" not in self.fired:
                muts = alive[(v["p_mut"][alive] > 0)]
                lin = getattr(self.sched, "lineage", None)
                if lin is not None and len(muts):
                    pid = int(v["p_id"][muts[0]])
                    if lin.depth_of(pid) >= 100:
                        self._emit(tick, "mutant_100_generations", polymer_id=pid)

            comp = v.get("compartment_id")
            if comp is not None and (comp > 0).any():
                cell = np.argwhere(comp > 0)[0]
                self._emit(tick, "first_compartment", cell=[int(cell[0]), int(cell[1])])

            # compartment lineage outcompeting free polymers
            in_c = row.get("poly_in_compartments", 0)
            share = in_c / n if n else 0.0
            if n >= 100 and share > 0.5 and share > self.prev_comp_share:
                self._emit(tick, "compartment_outcompetes", share=share)
            self.prev_comp_share = share

            # predation-like: a bound partner that is now dead
            bound_now: dict[int, int] = {}
            partners = v["p_partner"]
            ids = v["p_id"]
            states = v["p_state"]
            for s in alive:
                pr = partners[s]
                if pr >= 0 and pr != s:
                    bound_now[int(ids[s])] = int(pr)
            for pid, pslot in self.prev_bound.items():
                if pslot < len(states) and states[pslot] == 0 and pid in set(int(ids[a]) for a in alive):
                    self._emit(tick, "predation_like", survivor_id=pid)
                    break
            self.prev_bound = bound_now

            # aggregate of >= 2 lineages: bound pair with different parents
            for s in alive:
                pr = partners[s]
                if pr >= 0 and pr != s and states[pr] > 0:
                    if v["p_parent"][s] != v["p_parent"][pr]:
                        self._emit(tick, "aggregate_two_lineages",
                                   ids=[int(ids[s]), int(ids[pr])])
                        break

            # persistent signal correlation: [A] vs polymer density across wet cells
            if "signal_correlation" not in self.fired and n >= 50:
                try:
                    from firmament.core.chemistry import Chemistry  # index via sched
                    ia = self.sched.chem.index["A"]
                    a_field = v["species"][ia].ravel().astype(np.float64)
                    dens = np.bincount(v["p_cell"][alive], minlength=a_field.size).astype(np.float64)
                    if a_field.std() > 0 and dens.std() > 0:
                        r = float(np.corrcoef(a_field, dens)[0, 1])
                        self.signal_hits = self.signal_hits + 1 if r > 0.5 else 0
                        if self.signal_hits >= 3:
                            self._emit(tick, "signal_correlation", r=r)
                except Exception:
                    pass

            # persistent non-metabolic energy flow ("they built something" — vague on purpose)
            mem = row.get("membrane_L_total", 0)
            self.nonmet_hits = self.nonmet_hits + 1 if mem > 1000 else 0
            if self.nonmet_hits >= 5:
                self._emit(tick, "non_metabolic_energy_flow", membrane_L=mem)
