"""Meteor heats layer-1 temperature by dtk but registers only dtk * C_SURF_DRY (6.5e5)
with the audit. Layer 1's heat capacity is C_SURF_DRY + CW_VOL*water_depth
(audit.energy_stored), so on wet cells the energy actually added is larger than what
is registered -> the energy books do not close.

Control: the same meteor on a dry world closes exactly.
"""
import sys
sys.path.insert(0, "/home/nick/Life/firmament")
import shutil
import tempfile
from pathlib import Path

import numpy as np

from tests.conftest import build_test_sim, tiny_cfg
from firmament.core.audit import AuditError
from firmament.operator import causal, console

DEV = "cpu"


def trial(water_fraction: float, energy_J: float):
    tmp = Path(tempfile.mkdtemp(prefix="met_"))
    try:
        cfg = tiny_cfg(n=32, cap=200, water=water_fraction)
        cfg.run.audit_every_ticks = 10
        s, sched = build_test_sim(cfg, DEV, tmp / "run")
        aud = sched.audit
        while s.tick < 5:
            sched.tick_once()
        wd = s.water_depth.numpy().copy()
        stored0 = aud.energy_stored(s)
        inj0 = aud.injected_energy
        params = {"cx": 16, "cy": 16, "energy_J": energy_J, "crater_m": 2, "dust": 0.3}
        console.request(sched, "meteor", params)       # same path as API / CLI
        causal.apply_due(sched)                        # tick-boundary commit + apply
        stored1 = aud.energy_stored(s)
        registered = aud.injected_energy - inj0
        actual = stored1 - stored0
        print(f"\n--- meteor energy_J={energy_J:g} world water_fraction={water_fraction} "
              f"(wet cells {int((wd > 1e-3).sum())}/{wd.size}, mean depth on wet "
              f"{wd[wd > 1e-3].mean() if (wd > 1e-3).any() else 0:.3f} m)")
        print(f"stored-energy increase (audit.energy_stored): {actual:.6e} J")
        print(f"energy registered with audit               : {registered:.6e} J")
        print(f"unregistered energy                        : {actual - registered:.6e} J "
              f"(ratio actual/registered = {actual / registered:.4f})")
        err = None
        try:
            while s.tick < 60:
                sched.tick_once()
        except AuditError as e:
            err = e
        print("AuditError:", err)
        return err
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


dry = trial(0.0, 1e9)
wet = trial(0.4, 1e9)          # config default initial_water_fraction
print("\nVERDICT: dry ->", "AuditError" if dry else "books close",
      "| wet ->", "AuditError" if wet else "books close")


def gui_default_on_shakeout_world():
    """Ledger check only (no ticking): configs/world_small.yaml (RUNBOOK shakeout world),
    meteor with gui/app.js EVENT_PARAMS defaults."""
    from firmament.config import Config
    tmp = Path(tempfile.mkdtemp(prefix="met256_"))
    try:
        cfg = Config.load("/home/nick/Life/firmament/configs/world_small.yaml")
        cfg.polymers.capacity = 1000            # memory only; meteor ignores polymers
        s, sched = build_test_sim(cfg, DEV, tmp / "run")
        sched.tick_once()
        aud = sched.audit
        stored0, inj0 = aud.energy_stored(s), aud.injected_energy
        console.request(sched, "meteor", {"cx": 128, "cy": 128, "energy_J": 1e12,
                                          "crater_m": 2, "dust": 0.3})
        causal.apply_due(sched)
        actual, registered = aud.energy_stored(s) - stored0, aud.injected_energy - inj0
        print(f"\n--- world_small.yaml 256x256, GUI-default meteor")
        print(f"stored-energy increase {actual:.6e} J, registered {registered:.6e} J, "
              f"unregistered {actual - registered:.6e} J (x{actual / registered:.3f})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


gui_default_on_shakeout_world()
