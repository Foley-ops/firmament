"""Minimal lake-at-rest: 8x8 flat bed, 1 m of still water, one dry bank column 2 m high
(water surface 1 m BELOW the bank top -> the water cannot touch it). Control: same
pond with no bank. Fluid module only, isothermal, vapor exactly saturated so there
is zero evaporation/rain. Hydrostatics: nothing should move in either case."""
import sys, os, tempfile, math
sys.path.insert(0, "/home/nick/Life/firmament")
from pathlib import Path
import numpy as np
import warp as wp
from tests.conftest import tiny_cfg, build_test_sim

SCR = "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/env"
n = 8

def run(bank: bool):
    tmp = Path(tempfile.mkdtemp(dir=SCR)); os.chdir(tmp)
    cfg = tiny_cfg(n=n)
    state, sched = build_test_sim(cfg, "cpu", tmp / "run")
    sched.modules = [sched.modules[2]]          # fluid only
    elev = np.zeros((n, n)); h = np.full((n, n), 1.0)
    if bank:
        elev[:, n - 1] = 2.0; h[:, n - 1] = 0.0
    state.elevation = wp.array(elev, dtype=wp.float64, device="cpu")
    state.sediment = wp.array(np.zeros((n, n)), dtype=wp.float64, device="cpu")
    state.water_depth = wp.array(h, dtype=wp.float64, device="cpu")
    state.water_u = wp.zeros((n, n), dtype=wp.float64, device="cpu")
    state.water_v = wp.zeros((n, n), dtype=wp.float64, device="cpu")
    T = 288.0
    state.temp = wp.array(np.full((3, n, n), T), dtype=wp.float64, device="cpu")
    tc = T - 273.15
    qs = 13.0 * 0.6108 * math.exp(17.27 * tc / (tc + 237.3))
    state.vapor = wp.array(np.full((n, n), qs), dtype=wp.float64, device="cpu")
    print(f"--- bank={bank}: water surface eta = {1.0} m everywhere wet; bank top = {2.0 if bank else '-'} m")
    for t in range(1, 31):
        sched.tick_once()
        if t in (1, 5, 30):
            u = state.water_u.numpy(); hh = state.water_depth.numpy()
            wet = hh > 1e-4
            eta = (state.elevation.numpy() + hh)
            print(f"tick {t:3d}: max|u| {np.abs(u).max():8.4f} m/s  row0 u[0,:] = {np.round(u[0], 3).tolist()}")
            print(f"          row0 depth = {np.round(hh[0], 3).tolist()}  eta spread(wet) {eta[wet].max()-eta[wet].min():.4f} m")

run(bank=False)
run(bank=True)
