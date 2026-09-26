"""Meteor energy ledger: console.apply_natural('meteor') raises temp[1] (the surface
layer, capacity C_SURF_DRY + CW_VOL*depth) by dtk but registers only dtk*6.5e5 J
(C_SURF_DRY alone) with the audit. Over water the books cannot close.
Uses the exact params of tests/test_phase8_console.py's meteor event, 32x32 test world."""
import sys, os, tempfile, logging
sys.path.insert(0, "/home/nick/Life/firmament")
from pathlib import Path
import numpy as np
from tests.conftest import tiny_cfg, build_test_sim
from firmament.core.audit import AuditError
from firmament.operator import console
SCR = "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/env"
logging.getLogger("firmament.fluid").setLevel(logging.ERROR)

def run(params, label):
    tmp = Path(tempfile.mkdtemp(dir=SCR)); os.chdir(tmp)
    cfg = tiny_cfg(n=32, cap=1000)
    cfg.run.audit_every_ticks = 10
    state, sched = build_test_sim(cfg, "cpu", tmp / "run")
    aud = sched.audit
    for _ in range(5):
        sched.tick_once()                 # baseline audit at tick 0 passes
    # measure the event's own bookkeeping at the tick boundary
    inj0 = aud.injected_energy
    st0 = aud.energy_stored(state)
    wet = state.water_depth.numpy() > 1e-3
    console.request(sched, "meteor", params)
    from firmament.operator import causal
    causal.apply_due(sched)                # what tick_once does first, at the boundary
    st1 = aud.energy_stored(state)
    print(f"[{label}] wet cells {int(wet.sum())}/1024; stored-energy jump {st1-st0:.4e} J, "
          f"registered with audit {aud.injected_energy-inj0:.4e} J, unbooked {st1-st0-(aud.injected_energy-inj0):.4e} J")
    try:
        for _ in range(10):
            sched.tick_once()
        print(f"[{label}] audits through tick {state.tick}: all passed")
    except AuditError as e:
        print(f"[{label}] AuditError: {e}")

run({"cx": 16, "cy": 24, "energy_J": 1e9, "crater_m": 0.5, "dust": 0.2}, "test_phase8 meteor params")
run({"cx": 16, "cy": 24, "energy_J": 1e9, "crater_m": 0.0, "dust": 0.0}, "no crater, no dust")
