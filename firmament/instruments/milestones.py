"""Milestone detectors: emit ANALYSIS events, never act. Each fires once per run.

Names follow what the detector actually measures. Facts keep plain names
(first_copy, first_compartment); anything that implies a causal biological phenomenon
is a `candidate_*` alarm that needs a separate confirmatory analysis
(docs/DECISIONS.md, 2026-09-25). Detector state is saved in snapshots
(state_dict/load_state) so resume never re-fires a milestone.

Only COMPLETED polymers count (p_state 1..3); children under construction (5) are not
individuals yet.
"""
from __future__ import annotations

import numpy as np

from firmament.instruments import diversity
from firmament.io.logging import get

log = get("milestones")

P_FREE = 1
TOP_K = 20            # hashes tracked for sweep detection (bounded state)


def completed(v) -> np.ndarray:
    st = v["p_state"]
    return np.nonzero((st >= 1) & (st <= 3))[0]


class Milestones:
    def __init__(self, sched) -> None:
        self.sched = sched
        self.fired: set[str] = set()
        self.first_share: dict[str, float] = {}   # hash -> share when first in top-K
        self.prev_binder_pairs: dict[int, int] = {}
        self.aggregate_streak: dict[str, int] = {}
        self.comp_streak = 0
        self.signal_hits = 0
        self.nonmet_hits = 0
        names = getattr(getattr(sched, "poly", None), "motif_names", [])
        self.binder_bit = 1 << names.index("binder") if "binder" in names else 0

    # ---- persistence (saved inside snapshots) ----
    def state_dict(self) -> dict:
        return {"fired": sorted(self.fired), "first_share": self.first_share,
                "prev_binder_pairs": {str(k): v for k, v in self.prev_binder_pairs.items()},
                "aggregate_streak": self.aggregate_streak, "comp_streak": self.comp_streak,
                "signal_hits": self.signal_hits, "nonmet_hits": self.nonmet_hits}

    def load_state(self, d: dict) -> None:
        self.fired = set(d.get("fired", []))
        self.first_share = dict(d.get("first_share", {}))
        self.prev_binder_pairs = {int(k): v for k, v in d.get("prev_binder_pairs", {}).items()}
        self.aggregate_streak = dict(d.get("aggregate_streak", {}))
        self.comp_streak = d.get("comp_streak", 0)
        self.signal_hits = d.get("signal_hits", 0)
        self.nonmet_hits = d.get("nonmet_hits", 0)

    def _emit(self, tick: int, name: str, **fields) -> None:
        if name in self.fired:
            return
        self.fired.add(name)
        out = getattr(self.sched, "analysis", None)
        if out:
            out.append(name, tick, component="milestones", **fields)
        log.info(f"MILESTONE {name}", extra=fields)

    def _binder_pairs(self, v, alive) -> dict[int, int]:
        """Mutual partners that are both FREE and both carry the binder motif — this
        excludes copier/template pairs, which share the same p_partner field."""
        if not self.binder_bit:
            return {}
        partner, st, mot = v["p_partner"], v["p_state"], v["p_motifs"]
        pairs = {}
        for s in alive:
            pr = int(partner[s])
            if (pr >= 0 and pr != s and st[s] == P_FREE and st[pr] == P_FREE
                    and partner[pr] == s and mot[s] & self.binder_bit and mot[pr] & self.binder_bit):
                pairs[int(s)] = pr
        return pairs

    def check(self, v, row: dict) -> None:
        tick = v["tick"]
        if v["recent_copies"]:
            e = v["recent_copies"][0]
            self._emit(tick, "first_copy", child_id=int(e[0]), parent_id=int(e[1]),
                       cell=int(e[2]), length=int(e[3]))
        alive = completed(v)
        n = len(alive)
        if not n:
            return
        ids = v["p_id"]

        # candidate_fixation: a variant first seen as a minority (<=10%) sweeps to >90%
        hashes = diversity.seq_hashes(v)
        uniq, counts = np.unique(hashes, return_counts=True)
        if n >= 50:
            top = np.argsort(counts)[::-1][:TOP_K]
            for t in top:
                self.first_share.setdefault(str(int(uniq[t])), float(counts[t] / n))
            d = int(np.argmax(counts))
            share = float(counts[d] / n)
            if share > 0.9 and self.first_share.get(str(int(uniq[d])), 1.0) <= 0.10:
                self._emit(tick, "candidate_fixation", share=share, population=n,
                           first_share=self.first_share[str(int(uniq[d]))])

        if "candidate_mutant_depth100" not in self.fired:
            muts = alive[v["p_mut"][alive] > 0]
            lin = getattr(self.sched, "lineage", None)
            if lin is not None and len(muts):
                pid = int(ids[muts[0]])
                if lin.depth_of(pid) >= 100:
                    self._emit(tick, "candidate_mutant_depth100", polymer_id=pid)

        comp = v.get("compartment_id")
        if comp is not None and (comp > 0).any():
            cell = np.argwhere(comp > 0)[0]
            self._emit(tick, "first_compartment", cell=[int(cell[1]), int(cell[0])])

        in_c = row.get("poly_in_compartments", 0)
        self.comp_streak = self.comp_streak + 1 if (n >= 100 and in_c / n > 0.5) else 0
        if self.comp_streak >= 5:
            self._emit(tick, "candidate_compartment_advantage", share=in_c / n)

        # binder interactions (never template pairs)
        pairs = self._binder_pairs(v, alive)
        st = v["p_state"]
        for s, pr in self.prev_binder_pairs.items():
            if pr < len(st) and st[pr] == 0 and st[s] in (1, 2, 3):
                self._emit(tick, "candidate_predation", survivor_id=int(ids[s]))
                break
        self.prev_binder_pairs = pairs
        streak = {}
        for s, pr in pairs.items():
            if v["p_parent"][s] != v["p_parent"][pr] and hashes_of(v, s) != hashes_of(v, pr):
                key = f"{min(ids[s], ids[pr])}-{max(ids[s], ids[pr])}"
                streak[key] = self.aggregate_streak.get(key, 0) + 1
                if streak[key] >= 3:
                    self._emit(tick, "candidate_aggregate", ids=[int(ids[s]), int(ids[pr])])
        self.aggregate_streak = streak

        if "candidate_signal_correlation" not in self.fired and n >= 50:
            chem = getattr(self.sched, "chem", None)
            if chem is not None and "A" in chem.index:
                a_field = v["species"][chem.index["A"]].ravel().astype(np.float64)
                dens = np.bincount(v["p_cell"][alive], minlength=a_field.size).astype(np.float64)
                if a_field.std() > 0 and dens.std() > 0:
                    r = float(np.corrcoef(a_field, dens)[0, 1])
                    self.signal_hits = self.signal_hits + 1 if r > 0.5 else 0
                    if self.signal_hits >= 3:
                        self._emit(tick, "candidate_signal_correlation", r=r)

        mem = row.get("membrane_L_total", 0)
        self.nonmet_hits = self.nonmet_hits + 1 if mem > 1000 else 0
        if self.nonmet_hits >= 5:
            self._emit(tick, "candidate_nonmetabolic_flow", membrane_L=mem)


def hashes_of(v, slot: int) -> bytes:
    ln = int(v["p_len"][slot])
    return v["p_seq"][slot, :ln].tobytes()
