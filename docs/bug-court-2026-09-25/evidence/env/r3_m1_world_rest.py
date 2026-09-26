"""Same lake-at-rest check on the M1 world's own terrain (256x256, terrain_seed 7,
vents [[75,128],[175,50]], initial_water_fraction 0.4 = configs/world_m1.yaml).
Fluid module only, isothermal, vapor saturated (no evap/rain), 3 ticks."""
import sys, os, tempfile, math, time
sys.path.insert(0, "/home/nick/Life/firmament")
from pathlib import Path
import numpy as np
import warp as wp
from tests.conftest import tiny_cfg, build_test_sim

SCR = "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/env"
tmp = Path(tempfile.mkdtemp(dir=SCR)); os.chdir(tmp)
n = 256
cfg = tiny_cfg(n=n, vents=[[75, 128], [175, 50]], cap=1000)
state, sched = build_test_sim(cfg, "cpu", tmp / "run")
flu_step = sched.modules[2]
flu = flu_step.__self__
sched.modules = [flu_step]
T = 288.0
state.temp = wp.array(np.full((3, n, n), T), dtype=wp.float64, device="cpu")
tc = T - 273.15
qs = 13.0 * 0.6108 * math.exp(17.27 * tc / (tc + 237.3))
state.vapor = wp.array(np.full((n, n), qs), dtype=wp.float64, device="cpu")
h0 = state.water_depth.numpy().copy(); wet0 = h0 > 1e-4
eta0 = (state.elevation.numpy() + state.sediment.numpy() + h0)
sed0 = state.sediment.numpy().copy()
print(f"t=0 wet cells {wet0.sum()}, lake volume {h0.sum():.0f} m^3, max depth {h0.max():.2f} m, eta spread over wet {eta0[wet0].max()-eta0[wet0].min():.2e} m, speed 0")
t0 = time.time()
for t in range(1, 4):
    sched.tick_once()
    u = state.water_u.numpy(); v = state.water_v.numpy(); h = state.water_depth.numpy()
    sp = np.sqrt(u*u + v*v)
    eta = state.elevation.numpy() + state.sediment.numpy() + h
    moved = np.abs(flu.b["fx"].numpy()).sum() + np.abs(flu.b["fy"].numpy()).sum()
    print(f"tick {t}: n_sub {flu.last_nsub}  max speed {sp.max():.3f} m/s  cells >0.5 m/s: {(sp>0.5).sum()}  "
          f"mean speed(wet) {sp[wet0].mean():.4f}  eta spread(wet) {eta[wet0].max()-eta[wet0].min():.4f} m  "
          f"water volume through faces this tick {moved:.1f} m^3  max|dsed| {np.abs(state.sediment.numpy()-sed0).max():.2e} m")
print("cpu sec", round(time.time() - t0, 1))
