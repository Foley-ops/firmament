"""Zarr snapshots: every field + polymer arrays + counters + config hash.

A snapshot is the complete sim state: loading one and replaying the event log must
reproduce any later snapshot byte-for-byte (the replay test, Phase 4).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import zarr

from firmament.io.logging import get

log = get("snapshot")


class SnapshotManager:
    def __init__(self, snap_dir: Path, cfg) -> None:
        self.dir = Path(snap_dir)
        self.cfg = cfg

    def path(self, tick: int) -> Path:
        return self.dir / f"tick_{tick:012d}.zarr"

    def list(self) -> list[int]:
        return sorted(int(p.name[5:-5]) for p in self.dir.glob("tick_*.zarr"))

    def save(self, state, sched=None) -> Path:
        p = self.path(state.tick)
        if p.exists():
            shutil.rmtree(p)
        root = zarr.open_group(str(p), mode="w")
        for name, arr in state.to_numpy().items():
            root.create_array(name, data=arr, chunks="auto")
        meta = {
            "tick": state.tick,
            "next_poly_id": state.next_poly_id,
            "solar_mult": getattr(state, "solar_mult", 1.0),
            "config_hash": self.cfg.hash(),
        }
        if sched is not None and hasattr(sched, "audit"):
            a = sched.audit
            meta["audit"] = {
                "baseline_elements": None if a.baseline_elements is None else a.baseline_elements.tolist(),
                "baseline_water": a.baseline_water,
                "baseline_energy": a.baseline_energy,
                "injected_elements": a.injected_elements.tolist(),
                "injected_water": a.injected_water,
                "injected_energy": a.injected_energy,
            }
        root.attrs["meta"] = meta
        log.info("snapshot saved", extra={"path": str(p), "tick": state.tick})
        return p

    def load(self, state, sched=None, tick: int | None = None) -> None:
        ticks = self.list()
        if not ticks:
            raise FileNotFoundError(f"no snapshots in {self.dir}")
        t = tick if tick is not None else ticks[-1]
        root = zarr.open_group(str(self.path(int(t))), mode="r")
        meta = root.attrs["meta"]
        if meta["config_hash"] != self.cfg.hash():
            raise RuntimeError("snapshot config hash mismatch — refusing to load")
        arrays = {}
        for name in root.array_keys():
            arrays[name] = root[name][:]
        # fields not allocated until modules bind (e_in, e_chem, catalyst, membrane_store)
        # are restored by the caller after build_sim; State.load_numpy handles known ones.
        state.load_numpy(arrays)
        state.tick = int(meta["tick"])
        state.next_poly_id = int(meta["next_poly_id"])
        state.solar_mult = float(meta.get("solar_mult", 1.0))
        from firmament.io.logging import CLOCK
        CLOCK.tick = state.tick
        if sched is not None and hasattr(sched, "audit") and "audit" in meta:
            a = sched.audit
            am = meta["audit"]
            if am["baseline_elements"] is not None:
                a.baseline_elements = np.array(am["baseline_elements"], dtype=np.int64)
            a.baseline_water = am["baseline_water"]
            a.baseline_energy = am["baseline_energy"]
            a.injected_elements = np.array(am["injected_elements"], dtype=np.int64)
            a.injected_water = am["injected_water"]
            a.injected_energy = am["injected_energy"]
        log.info("snapshot loaded", extra={"tick": state.tick})


def fork_run(parent_dir: Path, snapshot: str | None = None) -> Path:
    """Fork: new run_id, parent pointer, first snapshot equals the parent's."""
    from firmament.config import Config
    from firmament.io import rundir

    cfg = Config.load(parent_dir / "config.yaml")
    meta = rundir.read_meta(parent_dir)
    new_dir = rundir.create(cfg, parent=meta["run_id"])
    if meta.get("touched"):
        rundir.mark_touched(new_dir)          # TOUCHED propagates to forks
    snaps = sorted((parent_dir / "snapshots").glob("tick_*.zarr"))
    if not snaps:
        raise FileNotFoundError("parent has no snapshots to fork from")
    src = snaps[-1] if snapshot is None else parent_dir / "snapshots" / snapshot
    shutil.copytree(src, new_dir / "snapshots" / src.name)
    # events up to the fork point carry over (they are part of the causal history)
    shutil.copyfile(parent_dir / "events.jsonl", new_dir / "events.jsonl")
    (new_dir / "forked_from.json").write_text(json.dumps(
        {"parent": meta["run_id"], "snapshot": src.name}))
    return new_dir
