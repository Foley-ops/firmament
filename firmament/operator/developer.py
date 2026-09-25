"""Developer mode: full edit power, and using it has a visible cost.

Edits are causal commands (firmament/operator/causal.py): logged as EDIT with full
parameters, so replay reproduces them exactly. Committing one live prints a loud
warning and marks the run (and all future forks) TOUCHED. Principle 3 in code.
Only available when the sim was started with --developer (sched.developer_allowed).
"""
from __future__ import annotations

import warp as wp

from firmament.io import rundir
from firmament.io.logging import get

log = get("developer")

_warned = False


def mark_touched(sched, params: dict) -> None:
    global _warned
    if not _warned:
        print("\n" + "!" * 70)
        print("!!  DEVELOPER EDIT — this run is now permanently marked TOUCHED  !!")
        print("!" * 70 + "\n")
        _warned = True
    if getattr(sched, "run", None) is not None and (sched.run / "meta.json").exists():
        rundir.mark_touched(sched.run)
    log.warning("EDIT applied", extra={"params": params})


def _edit(sched, action: str, description: str, **fields) -> dict:
    from firmament.operator import causal
    return causal.commit_now(sched, "EDIT", {"action": action,
                                             "description": description or action, **fields})


def set_field(sched, field: str, x: int, y: int, value: float, description: str = "") -> dict:
    return _edit(sched, "set_field", description, field=field, x=x, y=y, value=value)


def set_species(sched, species: str, x: int, y: int, count: int, description: str = "") -> dict:
    return _edit(sched, "set_species", description, species=species, x=x, y=y, count=count)


def edit_polymer(sched, slot: int, sequence: str | None = None, description: str = "") -> dict:
    return _edit(sched, "edit_polymer", description, slot=slot, sequence=sequence)


def apply_edit(sched, p: dict) -> None:
    s = sched.state
    action = p["action"]
    if action == "set_field":
        arr = getattr(s, p["field"])
        a = arr.numpy().copy()
        if a.ndim == 3:
            a[:, p["y"], p["x"]] = p["value"]
        else:
            a[p["y"], p["x"]] = p["value"]
        setattr(s, p["field"], wp.array(a, dtype=arr.dtype, device=s.device))
    elif action == "set_species":
        a = s.species.numpy().copy()
        a[sched.chem.index[p["species"]], p["y"], p["x"]] = p["count"]
        s.species = wp.array(a, dtype=wp.int32, device=s.device)
    elif action == "edit_polymer":
        from firmament.operator.console import _parse_seq
        slot = p["slot"]
        if p.get("sequence") is not None:
            seq = _parse_seq(p["sequence"])
            sq = s.p_seq.numpy().copy()
            sq[slot, :] = 0
            sq[slot, :len(seq)] = seq
            s.p_seq = wp.array(sq, dtype=wp.uint8, device=s.device)
            ln = s.p_len.numpy().copy()
            ln[slot] = len(seq)
            s.p_len = wp.array(ln, dtype=wp.int32, device=s.device)
            mo = s.p_motifs.numpy().copy()
            mo[slot] = sched.poly.motif_mask_host(seq)
            s.p_motifs = wp.array(mo, dtype=wp.int32, device=s.device)
    else:
        raise ValueError(f"unknown developer action: {action}")
