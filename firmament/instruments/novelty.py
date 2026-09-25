"""Novelty detector: compressed-size jumps and never-seen motif combinations."""
from __future__ import annotations

from firmament.instruments import diversity
from firmament.io.logging import get

log = get("novelty")

JUMP_FACTOR = 1.5


class Novelty:
    def __init__(self, sched) -> None:
        self.sched = sched
        self.prev_compressed = None
        self.seen_combos: set[int] = set()

    def state_dict(self) -> dict:
        return {"prev_compressed": self.prev_compressed, "seen_combos": sorted(self.seen_combos)}

    def load_state(self, d: dict) -> None:
        self.prev_compressed = d.get("prev_compressed")
        self.seen_combos = set(d.get("seen_combos", []))

    def sample(self, v) -> dict:
        import numpy as np
        alive = np.nonzero(((v["p_state"] >= 1) & (v["p_state"] <= 3)))[0]
        row = {}
        if len(alive) == 0:
            return row
        comp = None
        # compressed size jump
        d = diversity.sample(v)
        comp = d["compressed_bytes"]
        if self.prev_compressed and comp > self.prev_compressed * JUMP_FACTOR and comp > 4096:
            self._emit(v, "novelty_compression_jump", before=self.prev_compressed, after=comp)
        self.prev_compressed = comp
        # new motif combinations
        for mask in set(int(m) for m in v["p_motifs"][alive]):
            if mask and mask not in self.seen_combos:
                self.seen_combos.add(mask)
                if len(self.seen_combos) > 1:      # the seed's own combo is not news
                    self._emit(v, "novelty_motif_combo", combo=mask)
        return row

    def _emit(self, v, name, **fields) -> None:
        ev = getattr(self.sched, "analysis", None)
        if ev:
            ev.append(name, v["tick"], component="novelty", **fields)
