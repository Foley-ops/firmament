"""Lake at rest: terrain.generate's initial water is a flat free surface. With no
forcing (fluid module only, isothermal, vapor exactly saturated -> no evap/rain),
basic physics says the water must stay at rest. Measure velocities / surface."""
import sys, os, tempfile, math
sys.path.insert(0, "/home/nick/Life/firmament")
from pathlib import Path
import numpy as np
import warp as wp
from tests.conftest import tiny_cfg, build_test_sim

SCR = "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/env"
tmp = Path(tempfile.mkdtemp(dir=SCR)); os.chdir(tmp)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
cfg = tiny_cfg(n=n)
state, sched = build_test_sim(cfg, "cpu", tmp / "run")
sched.modules = [sched.modules[2]]          # fluid only
T = 288.0
state.temp = wp.array(np.full((3, n, n), T), dtype=wp.float64, device="cpu")
tc = T - 273.15
qs = 13.0 * 0.6108 * math.exp(17.27 * tc / (tc + 237.3))
state.vapor = wp.array(np.full((n, n), qs), dtype=wp.float64, device="cpu")

elev = state.elevation.numpy() + state.sediment.numpy()
h0 = state.water_depth.numpy().copy()
wet0 = h0 > 1e-4
eta0 = (elev + h0)[wet0]
print(f"t=0: wet cells {wet0.sum()}, eta range over wet cells {eta0.min():.9f}..{eta0.max():.9f} (flat)")
print(f"t=0: max|u| {np.abs(state.water_u.numpy()).max():.3e}  max|v| {np.abs(state.water_v.numpy()).max():.3e}")
for t in range(1, 121):
    sched.tick_once()
    if t in (1, 2, 5, 10, 30, 60, 120):
        u = state.water_u.numpy(); v = state.water_v.numpy(); h = state.water_depth.numpy()
        eta = (state.elevation.numpy() + state.sediment.numpy() + h)[wet0]
        sp = np.sqrt(u*u + v*v)
        print(f"tick {t:4d}: max speed {sp.max():.4f} m/s  mean speed(wet) {sp[wet0].mean():.4f}  "
              f"eta spread {eta.max()-eta.min():.4f} m  max|dh| {np.abs(h-h0).max():.4f} m  "
              f"vapor-change {abs(state.vapor.numpy()-qs).max():.2e}")
