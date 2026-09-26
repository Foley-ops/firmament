"""Dust settling (radiation.k_radiation: dust *= 0.99999 every TICK) is a per-tick law,
not a per-second one. Same world, same 2 sim-days of physical time, dt=60 vs dt=120:
the dust left (and the sunlight absorbed) should not depend on the numerical dt."""
import sys, os, tempfile
sys.path.insert(0, "/home/nick/Life/firmament")
from pathlib import Path
import numpy as np
import warp as wp
from tests.conftest import tiny_cfg, build_test_sim
SCR = "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/env"

def run(dt):
    tmp = Path(tempfile.mkdtemp(dir=SCR)); os.chdir(tmp)
    n = 8
    cfg = tiny_cfg(n=n, dt=dt)
    state, sched = build_test_sim(cfg, "cpu", tmp / "run")
    sched.modules = [sched.modules[0]]          # radiation only
    state.albedo_dust = wp.array(np.full((n, n), 0.5, dtype=np.float32), device="cpu")
    ticks = int(2 * 86400 / dt)
    for _ in range(ticks):
        sched.tick_once()
    d = float(state.albedo_dust.numpy().mean())
    ein = float(state.e_in.numpy().sum())
    print(f"dt={dt:5.0f}s ticks={ticks:5d} sim_time={ticks*dt/86400:.1f} d  dust 0.5 -> {d:.6f}  "
          f"e_in(total absorbed SW+geo) {ein:.6e} J")
    return d

a = run(60.0)
b = run(120.0)
c = run(600.0)
print(f"dust decayed at dt=60: {0.5-a:.6f}; at dt=120: {0.5-b:.6f} (ratio {(0.5-a)/(0.5-b):.3f}); at dt=600: {0.5-c:.6f}")
print(f"implied e-folding time: dt=60 -> {60/1e-5/86400:.1f} sim-days; dt=120 -> {120/1e-5/86400:.1f}; dt=600 -> {600/1e-5/86400:.1f}")
