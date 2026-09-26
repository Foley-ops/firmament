"""Developer mode: full edit power, and using it has a visible cost.

Edits are causal commands (firmament/operator/causal.py): logged as EDIT with full
parameters, so replay reproduces them exactly. Committing one live prints a loud
warning and marks the run (and all future forks) TOUCHED. Principle 3 in code.
Only available when the sim was started with --developer (sched.developer_allowed).
"""
from __future__ import annotations

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


EDITABLE_FIELDS = ("elevation", "sediment", "water_depth", "water_u", "water_v", "vapor", "temp",
                   "light", "light_water", "compartment_id", "membrane_store", "albedo_dust",
                   "vent_flux")


def validate_edit(sched, p: dict) -> None:
    import numbers

    import numpy as np
    s = sched.state
    h, w = s.shape
    action = p.get("action")

    def cell_ok():
        x, y = p.get("x"), p.get("y")
        if not (isinstance(x, numbers.Integral) and isinstance(y, numbers.Integral)
                and 0 <= x < w and 0 <= y < h):
            raise ValueError(f"cell ({x},{y}) outside the {w}x{h} world")
    if action == "set_field":
        if p.get("field") not in EDITABLE_FIELDS:
            raise ValueError(f"field {p.get('field')!r} is not editable")
        cell_ok()
        v = p.get("value")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
            raise ValueError("value must be a finite number")
    elif action == "set_species":
        if p.get("species") not in sched.chem.index:
            raise ValueError(f"unknown species {p.get('species')!r}")
        cell_ok()
        c = p.get("count")
        if not isinstance(c, numbers.Integral) or not 0 <= c < 2**31:
            raise ValueError("count must be an integer in [0, 2^31)")
    elif action == "edit_polymer":
        slot = p.get("slot")
        if not isinstance(slot, numbers.Integral) or not 0 <= slot < s.p_cap:
            raise ValueError(f"slot {slot} out of range")
        if p.get("sequence") is not None:
            from firmament.core.polymers import MIN_LEN
            from firmament.operator.console import _parse_seq
            n = len(_parse_seq(p["sequence"]))
            if not MIN_LEN <= n <= sched.cfg.polymers.max_length:
                raise ValueError(f"sequence length {n} out of range")
    else:
        raise ValueError(f"unknown developer action: {action!r}")


def apply_edit(sched, p: dict) -> None:
    from firmament.operator.console import _assign, _parse_seq
    s = sched.state
    action = p["action"]
    if action == "set_field":
        a = getattr(s, p["field"]).numpy().copy()
        if a.ndim == 3:
            a[:, p["y"], p["x"]] = p["value"]
        else:
            a[p["y"], p["x"]] = p["value"]
        _assign(s, p["field"], a)
    elif action == "set_species":
        a = s.species.numpy().copy()
        a[sched.chem.index[p["species"]], p["y"], p["x"]] = p["count"]
        _assign(s, "species", a)
    elif action == "edit_polymer":
        slot = p["slot"]
        if p.get("sequence") is not None:
            seq = _parse_seq(p["sequence"])
            sq = s.p_seq.numpy().copy()
            sq[slot, :] = 0
            sq[slot, :len(seq)] = seq
            _assign(s, "p_seq", sq)
            ln = s.p_len.numpy().copy()
            ln[slot] = len(seq)
            _assign(s, "p_len", ln)
            mo = s.p_motifs.numpy().copy()
            mo[slot] = sched.poly.motif_mask_host(seq)
            _assign(s, "p_motifs", mo)
    # an EDIT deliberately breaks conservation (the run is TOUCHED): the audit takes a
    # new baseline at its next check instead of halting the run
    aud = getattr(sched, "audit", None)
    if aud is not None:
        aud.baseline_elements = None
