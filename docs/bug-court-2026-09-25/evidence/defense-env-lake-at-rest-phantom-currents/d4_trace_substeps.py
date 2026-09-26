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



import sys as _s
n = int(_s.argv[1]) if len(_s.argv) > 1 else 32
vents = None if n != 256 else [[75, 128], [175, 50]]
tmp = Path(tempfile.mkdtemp(dir=SCR)); os.chdir(tmp)
cfg = tiny_cfg(n=n, cap=1000, vents=vents)
state, sched = build_test_sim(cfg, "cpu", tmp / "run")
el = state.elevation.numpy(); sd = state.sediment.numpy(); h0 = state.water_depth.numpy().copy()
dev = "cpu"
hw = wp.array(h0, dtype=wp.float64, device=dev); hn = wp.zeros_like(hw)
u = wp.zeros_like(hw); v = wp.zeros_like(hw); un = wp.zeros_like(hw); vn = wp.zeros_like(hw)
t1 = wp.array(np.full((n, n), 288.0), dtype=wp.float64, device=dev); t1n = wp.zeros_like(t1)
fx = wp.zeros_like(hw); fy = wp.zeros_like(hw)
E = state.elevation; S = state.sediment
nsub = 704; dts = wp.float64(60.0 / nsub)
first = None
for k in range(nsub):
    wp.launch(k_momentum_wall, dim=(n, n), inputs=[hw, E, S, u, v, un, vn, dts, n, n], device=dev)
    u, un = un, u; v, vn = vn, v
    wp.launch(fl.k_height, dim=(n, n), inputs=[hw, hn, E, S, u, v, t1, t1n, fx, fy, dts, n, n], device=dev)
    hw, hn = hn, hw; t1, t1n = t1n, t1
    U = u.numpy(); V = v.numpy(); sp = np.sqrt(U*U+V*V)
    if sp.max() > 0 and first is None:
        first = k
        idx = np.argwhere(sp > 0)
        print(f"first nonzero speed at substep {k}: max {sp.max():.3e}, n cells {len(idx)}")
        for (a, b) in idx[:8]:
            print(f"   cell ({a},{b}) h={h0[a,b]:.6f} sp={sp[a,b]:.3e}  neighbours h:",
                  [round(float(h0[x,y]),6) for x,y in ((a,b-1),(a,b+1),(a-1,b),(a+1,b)) if 0<=x<n and 0<=y<n],
                  " eta:", [repr(float(el[x,y]+sd[x,y]+h0[x,y])) for x,y in ((a,b-1),(a,b+1),(a-1,b),(a+1,b)) if 0<=x<n and 0<=y<n])
    if k in (0, 10, 50, 100, 200, 400, 703):
        print(f"substep {k}: max speed {sp.max():.3e}")
