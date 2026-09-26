"""Defense probe for meteor-wet-energy-unregistered.

- Uses the repo's OWN meteor params (tests/test_phase8_console.py EVENTS, tick 260)
  and the UNMODIFIED tiny_cfg audit cadence (audit_every_ticks=500): no injected states.
- Checks that the unregistered amount equals exactly sum(CW_VOL * water_depth * dT1).
- Causation control: registering just that missing term makes the audit pass.
- Reports the audit's relative tolerance headroom (scale grows with cumulative e_in).
"""
import sys
sys.path.insert(0, "/home/nick/Life/firmament")
import shutil
import tempfile
from pathlib import Path

import numpy as np

from tests.conftest import build_test_sim, tiny_cfg
from firmament.core import physconst as pc
from firmament.core.audit import AuditError
from firmament.operator import console

DEV = "cpu"
METEOR = {"cx": 16, "cy": 24, "energy_J": 1e9, "crater_m": 0.5, "dust": 0.2}  # test_phase8


def trial(label, water, fix=False, t_event=260, t_end=510):
    tmp = Path(tempfile.mkdtemp(prefix="defmet_"))
    try:
        cfg = tiny_cfg(n=32, cap=500, water=water)          # audit_every_ticks stays 500
        s, sched = build_test_sim(cfg, DEV, tmp / "run")
        aud = sched.audit
        while s.tick + 1 < t_event:
            sched.tick_once()
        console.request(sched, "meteor", METEOR)            # exactly as the phase-8 test
        # capture the state right before the tick boundary that applies it
        d0 = s.water_depth.numpy().copy()
        t0 = s.temp.numpy()[1].copy()
        st0, inj0 = aud.energy_stored(s), aud.injected_energy
        from firmament.operator import causal
        causal.apply_due(sched)                             # the commit the tick would do
        t1 = s.temp.numpy()[1].copy()
        st1, inj1 = aud.energy_stored(s), aud.injected_energy
        actual, reg = st1 - st0, inj1 - inj0
        missing = float((pc.CW_VOL * d0 * (t1 - t0)).sum())
        print(f"\n[{label}] water={water} tick={s.tick} wet cells={(d0 > 1e-3).sum()}")
        print(f"  stored delta {actual:.6e}  registered {reg:.6e}  unregistered {actual - reg:.6e}")
        print(f"  sum(CW_VOL*d*dT1) = {missing:.6e}  (matches unregistered: "
              f"{np.isclose(actual - reg, missing, rtol=1e-9, atol=1e-3)})")
        if fix:
            aud.register_injection(energy=missing)
            print("  [control] registered the missing water term")
        err = None
        try:
            while s.tick < t_end:
                sched.tick_once()
        except AuditError as e:
            err = e
        e_in = float(s.e_in.numpy().sum()) + aud.injected_energy
        print(f"  ran to tick {s.tick}; cumulative e_in {e_in:.3e} J -> 1e-3 tolerance "
              f"{1e-3 * e_in:.3e} J")
        print(f"  AuditError: {err}")
        return err
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


trial("dry control", 0.0)
trial("wet default", 0.4)
trial("wet + missing term registered", 0.4, fix=True)
