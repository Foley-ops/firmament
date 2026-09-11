"""Developer mode: full edit power, and using it has a visible cost.

First use prints a loud warning, writes an EDIT event, and marks the run (and all
future forks) TOUCHED. Principle 3 in code.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from firmament.io import rundir
from firmament.io.logging import get

log = get("developer")

_warned = False


def _mark(sched, description: str, **fields) -> None:
    global _warned
    if not _warned:
        print("\n" + "!" * 70)
        print("!!  DEVELOPER EDIT — this run is now permanently marked TOUCHED  !!")
        print("!" * 70 + "\n")
        _warned = True
    sched.events.append("EDIT", sched.state.tick, component="developer",
                        description=description, **fields)
    rundir.mark_touched(sched.run)
    log.warning("EDIT applied", extra={"description": description} | fields)


def set_field(sched, field: str, x: int, y: int, value: float, description: str = "") -> None:
    s = sched.state
    arr = getattr(s, field)
    a = arr.numpy()
    if a.ndim == 3:
        a[:, y, x] = value
    else:
        a[y, x] = value
    setattr(s, field, wp.array(a, dtype=arr.dtype, device=s.device))
    _mark(sched, description or f"set {field}[{y},{x}]={value}", field=field, cell=[x, y], value=value)


def set_species(sched, species: str, x: int, y: int, count: int, description: str = "") -> None:
    s = sched.state
    k = sched.chem.index[species]
    a = s.species.numpy()
    a[k, y, x] = count
    s.species = wp.array(a, dtype=wp.int32, device=s.device)
    _mark(sched, description or f"set species {species}[{y},{x}]={count}",
          species=species, cell=[x, y], count=count)


def edit_polymer(sched, slot: int, sequence: str | None = None, description: str = "") -> None:
    s = sched.state
    if sequence is not None:
        sym = {"M1": 1, "M2": 2, "M3": 3, "M4": 4}
        toks = [sequence[i:i + 2] for i in range(0, len(sequence), 2)]
        seq = np.array([sym[t] for t in toks], dtype=np.uint8)
        sq = s.p_seq.numpy()
        sq[slot, :] = 0
        sq[slot, :len(seq)] = seq
        s.p_seq = wp.array(sq, dtype=wp.uint8, device=s.device)
        ln = s.p_len.numpy()
        ln[slot] = len(seq)
        s.p_len = wp.array(ln, dtype=wp.int32, device=s.device)
        # rescan motifs so state stays coherent
        poly = sched.poly
        mask = 0
        for mi, row in enumerate(poly.motif_np):
            k = len(row)
            for pos in range(len(seq) - k + 1):
                if np.array_equal(seq[pos:pos + k], row):
                    mask |= 1 << mi
                    break
        mo = s.p_motifs.numpy()
        mo[slot] = mask
        s.p_motifs = wp.array(mo, dtype=wp.int32, device=s.device)
    _mark(sched, description or f"edit polymer slot {slot}", slot=slot, sequence=sequence)
