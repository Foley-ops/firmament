"""Causality test (in-memory monkeypatch only; repo untouched).
Replace ONLY the eta used for a DRY neighbour in k_momentum's centred difference:
a dry neighbour whose bed is above this cell's free surface is a wall -> use this cell's eta.
Everything else (friction, CFL, k_height, erosion) is the repo code unchanged.
If the lake then stays at rest, lines 87-88 are the cause."""
import sys, os, tempfile, math
sys.path.insert(0, "/home/nick/Life/firmament")
from pathlib import Path
import numpy as np
import warp as wp
from tests.conftest import tiny_cfg, build_test_sim
import firmament.core.fluid as fl
from firmament.core import physconst as pc

SCR = "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/defense-env-lake-at-rest-phantom-currents"


@wp.func
def nb_eta(hn: wp.float64, en: wp.float64, ec: wp.float64) -> wp.float64:
    if hn < wp.float64(fl.H_MIN) and en > ec:
        return ec
    return en


@wp.kernel
def k_momentum_wall(hw: wp.array2d(dtype=wp.float64), elev: wp.array2d(dtype=wp.float64),
                    sed: wp.array2d(dtype=wp.float64), u: wp.array2d(dtype=wp.float64),
                    v: wp.array2d(dtype=wp.float64), u_new: wp.array2d(dtype=wp.float64),
                    v_new: wp.array2d(dtype=wp.float64), dts: wp.float64, H: int, W: int):
    i, j = wp.tid()
    if hw[i, j] < wp.float64(fl.H_MIN):
        u_new[i, j] = wp.float64(0.0)
        v_new[i, j] = wp.float64(0.0)
        return
    jm = wp.max(j - 1, 0)
    jp = wp.min(j + 1, W - 1)
    im = wp.max(i - 1, 0)
    ip = wp.min(i + 1, H - 1)
    ec = elev[i, j] + sed[i, j] + hw[i, j]
    exp_ = nb_eta(hw[i, jp], elev[i, jp] + sed[i, jp] + hw[i, jp], ec)
    exm = nb_eta(hw[i, jm], elev[i, jm] + sed[i, jm] + hw[i, jm], ec)
    eyp = nb_eta(hw[ip, j], elev[ip, j] + sed[ip, j] + hw[ip, j], ec)
    eym = nb_eta(hw[im, j], elev[im, j] + sed[im, j] + hw[im, j], ec)
    ex = (exp_ - exm) / wp.float64(jp - jm)
    ey = (eyp - eym) / wp.float64(ip - im)
    hh = wp.max(hw[i, j], wp.float64(0.05))
    sp = wp.sqrt(u[i, j] * u[i, j] + v[i, j] * v[i, j])
    mann = wp.float64(pc.G) * wp.float64(fl.MANNING_N) * wp.float64(fl.MANNING_N)
    drag = wp.float64(1.0) / (wp.float64(1.0) + dts * mann * sp / wp.pow(hh, wp.float64(4.0 / 3.0)))
    u_new[i, j] = (u[i, j] - wp.float64(pc.G) * ex * dts) * drag
    v_new[i, j] = (v[i, j] - wp.float64(pc.G) * ey * dts) * drag


def isothermal_saturated(state, n):
    T = 288.0
    state.temp = wp.array(np.full((3, n, n), T), dtype=wp.float64, device="cpu")
    tc = T - 273.15
    qs = 13.0 * 0.6108 * math.exp(17.27 * tc / (tc + 237.3))
    state.vapor = wp.array(np.full((n, n), qs), dtype=wp.float64, device="cpu")


def run(label, n, ticks, patched, vents=None, custom_bank=False):
    orig = fl.k_momentum
    if patched:
        fl.k_momentum = k_momentum_wall
    try:
        tmp = Path(tempfile.mkdtemp(dir=SCR)); os.chdir(tmp)
        cfg = tiny_cfg(n=n, vents=vents, cap=1000)
        state, sched = build_test_sim(cfg, "cpu", tmp / "run")
        flu_step = sched.modules[2]
        flu = flu_step.__self__
        sched.modules = [flu_step]
        if custom_bank:
            elev = np.zeros((n, n)); h = np.full((n, n), 1.0)
            elev[:, n - 1] = 2.0; h[:, n - 1] = 0.0
            state.elevation = wp.array(elev, dtype=wp.float64, device="cpu")
            state.sediment = wp.array(np.zeros((n, n)), dtype=wp.float64, device="cpu")
            state.water_depth = wp.array(h, dtype=wp.float64, device="cpu")
            state.water_u = wp.zeros((n, n), dtype=wp.float64, device="cpu")
            state.water_v = wp.zeros((n, n), dtype=wp.float64, device="cpu")
        isothermal_saturated(state, n)
        h0 = state.water_depth.numpy().copy(); wet0 = h0 > 1e-4
        sed0 = state.sediment.numpy().copy()
        for t in range(1, ticks + 1):
            sched.tick_once()
        u = state.water_u.numpy(); v = state.water_v.numpy(); h = state.water_depth.numpy()
        sp = np.sqrt(u * u + v * v)
        eta = state.elevation.numpy() + state.sediment.numpy() + h
        print(f"{label:38s} patched={patched!s:5s} after {ticks:3d} ticks: n_sub {flu.last_nsub:3d}  "
              f"max speed {sp.max():9.5f} m/s  cells>0.5 {(sp > 0.5).sum():5d}  "
              f"eta spread(wet0) {eta[wet0].max() - eta[wet0].min():.2e} m  "
              f"max|dh| {np.abs(h - h0).max():.2e}  max|dsed| {np.abs(state.sediment.numpy() - sed0).max():.2e}",
              flush=True)
    finally:
        fl.k_momentum = orig


for patched in (False, True):
    run("8x8 pond + 2 m dry bank (r2)", 8, 30, patched, custom_bank=True)
for patched in (False, True):
    run("32x32 terrain.generate (r1 32)", 32, 60, patched)
for patched in (False, True):
    run("M1 terrain 256x256 (r3)", 256, 3, patched, vents=[[75, 128], [175, 50]])
