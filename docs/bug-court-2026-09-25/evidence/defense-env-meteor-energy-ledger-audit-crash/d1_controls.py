"""Defense controls for meteor-energy-ledger-audit-crash."""
import sys, os, tempfile, logging, time
sys.path.insert(0, "/home/nick/Life/firmament")
from pathlib import Path
import numpy as np
from tests.conftest import tiny_cfg, build_test_sim
from firmament.core.audit import AuditError
from firmament.core import physconst as pc
from firmament.operator import console, causal
SCR = "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/defense-env-meteor-energy-ledger-audit-crash"
logging.getLogger("firmament.fluid").setLevel(logging.ERROR)
P = {"cx": 16, "cy": 24, "energy_J": 1e9, "crater_m": 0.5, "dust": 0.2}   # tests/test_phase8_console.py

def build(water=0.4, every=10):
    tmp = Path(tempfile.mkdtemp(dir=SCR)); os.chdir(tmp)
    cfg = tiny_cfg(n=32, cap=1000, water=water)
    cfg.run.audit_every_ticks = every
    return build_test_sim(cfg, "cpu", tmp / "run")

def run(label, meteor, water=0.4, fix=False, every=10, nticks=15):
    state, sched = build(water, every)
    aud = sched.audit
    orig = console.apply_natural
    if fix:  # book the TRUE layer-1 energy delta instead of dtk*C_SURF_DRY
        def patched(sched_, ev, params, n, at):
            t0 = state.temp.numpy()[1].copy(); d = state.water_depth.numpy()
            inj0 = aud.injected_energy
            orig(sched_, ev, params, n, at)
            dT = state.temp.numpy()[1] - t0
            aud.injected_energy = inj0 + float(((pc.C_SURF_DRY + pc.CW_VOL * d) * dT).sum())
        console.apply_natural = patched
    try:
        for _ in range(5):
            sched.tick_once()
        d = state.water_depth.numpy()
        if meteor:
            st0, inj0 = aud.energy_stored(state), aud.injected_energy
            console.request(sched, "meteor", P)
            causal.apply_due(sched)
            print(f"[{label}] wet(>1mm)={int((d>1e-3).sum())} mean depth wet={d[d>1e-3].mean() if (d>1e-3).any() else 0:.3f} m; "
                  f"stored jump {aud.energy_stored(state)-st0:.4e} J; booked {aud.injected_energy-inj0:.4e} J")
        while state.tick < nticks:
            sched.tick_once()
        print(f"[{label}] audits (every {every}) through tick {state.tick}: PASSED")
    except AuditError as e:
        print(f"[{label}] AuditError: {e}")
    finally:
        console.apply_natural = orig

t = time.time()
run("A control: no meteor", meteor=False)
run("B meteor, as shipped", meteor=True)
run("C meteor, booking corrected to C_SURF_DRY+CW_VOL*d", meteor=True, fix=True)
run("D dry world (water=0), as shipped", meteor=True, water=0.0)
print(f"elapsed {time.time()-t:.1f}s")
