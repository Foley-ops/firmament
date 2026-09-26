"""Volcano with an odd mineral_counts: audit books do not close (exact integers).

console.apply_natural('volcano') adds `inj // 2` H2S PER CELL but registers
`inj.sum() // 2` H2S in total. For odd mineral_counts these differ by mask.sum()//2
molecules -> the element audit raises AuditError at its next check.
Control: an even mineral_counts (GUI default 10000) closes exactly.
"""
import sys
sys.path.insert(0, "/home/nick/Life/firmament")
import shutil
import tempfile
from pathlib import Path

import numpy as np

from tests.conftest import build_test_sim, tiny_cfg
from firmament.core.audit import AuditError
from firmament.operator import console

DEV = "cpu"


def trial(mineral_counts: int):
    tmp = Path(tempfile.mkdtemp(prefix="volc_"))
    try:
        cfg = tiny_cfg(n=16, cap=200)
        cfg.run.audit_every_ticks = 10          # audit cadence only; physics unchanged
        s, sched = build_test_sim(cfg, DEV, tmp / "run")
        aud = sched.audit
        while s.tick < 5:
            sched.tick_once()                    # audit baseline taken at tick 0
        params = {"cx": 8, "cy": 8, "radius": 3, "heat_J_m2": 1e6, "cone_m": 1.0,
                  "mineral_counts": mineral_counts}
        # ledger check around the exact apply step (the same path tick_once uses)
        el0 = aud.element_totals(s) - aud.injected_elements
        console.request(sched, "volcano", params)          # same entry the API/CLI use
        from firmament.operator import causal
        causal.apply_due(sched)                            # tick-boundary commit+apply
        el1 = aud.element_totals(s) - aud.injected_elements
        idx = sched.chem.index
        mask = console._region_mask(s.shape, params)
        print(f"\n--- volcano mineral_counts={mineral_counts} (cells in mask={int(mask.sum())})")
        print("elements:", list(sched.chem.elements))
        print("net unregistered element change after apply:", (el1 - el0).tolist())
        err = None
        try:
            while s.tick < 30:
                sched.tick_once()
        except AuditError as e:
            err = e
        print("AuditError:", err)
        return err
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


e_even = trial(10000)     # GUI default
e_odd = trial(10001)      # any odd value typed into the GUI/CLI
e_one = trial(1)
print("\nVERDICT: even ->", "AuditError" if e_even else "books close",
      "| odd ->", "AuditError" if e_odd else "books close",
      "| 1 ->", "AuditError" if e_one else "books close")
