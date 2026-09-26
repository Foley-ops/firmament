"""Defense probe A: isolate the cause. Same still pond as the prosecutor's r7, run under
three diffusion kernels (swapped in by monkeypatching the module global, no repo edit):
  actual : the repo's k_species_diffuse (count-difference exchange)
  none   : diffusion disabled (copy) -> shows advection/other code is not the cause
  fick   : Fickian counterfactual, flux = diff_frac * min(hA,hB) * (nA/hA - nB/hB)
If only 'actual' un-mixes, the diffusion formula alone is responsible."""
import sys, os, tempfile, math
sys.path.insert(0, "/home/nick/Life/firmament")
from pathlib import Path
import numpy as np
import warp as wp
from tests.conftest import tiny_cfg, build_test_sim
import firmament.core.fluid as F
from firmament.core.fluid import stoch_round

SCR = "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/defense-env-diffusion-on-counts-not-concentration"
ACTUAL = F.k_species_diffuse
HMIN = 1e-4


@wp.kernel
def k_nodiff(spec: wp.array3d(dtype=wp.int32), spec_new: wp.array3d(dtype=wp.int32),
             hw: wp.array2d(dtype=wp.float64), seed: wp.int32,
             H: int, W: int, ns: int, diff_frac: wp.float64):
    s, i, j = wp.tid()
    spec_new[s, i, j] = spec[s, i, j]


@wp.func
def fick_out(spec: wp.array3d(dtype=wp.int32), s: int, ci: int, cj: int, face: int,
             hw: wp.array2d(dtype=wp.float64), seed: wp.int32,
             H: int, W: int, ns: int, diff_frac: wp.float64) -> int:
    n = spec[s, ci, cj]
    ha = hw[ci, cj]
    if n <= 0 or ha < wp.float64(1.0e-4):
        return 0
    taken = int(0)
    for k in range(4):
        ni = ci
        nj = cj
        if k == 0:
            nj = cj + 1
        elif k == 1:
            nj = cj - 1
        elif k == 2:
            ni = ci + 1
        else:
            ni = ci - 1
        if ni < 0 or nj < 0 or ni >= H or nj >= W:
            continue
        hb = hw[ni, nj]
        if hb < wp.float64(1.0e-4):
            continue
        dc = wp.float64(n) / ha - wp.float64(spec[s, ni, nj]) / hb
        if dc <= wp.float64(0.0):
            continue
        st = wp.rand_init(seed, wp.int32(((ci * W + cj) * 4 + k) * ns + s))
        amt = stoch_round(dc * wp.min(ha, hb) * diff_frac, st)
        if amt > n - taken:
            amt = n - taken
        if k == face:
            return amt
        taken += amt
    return 0


@wp.kernel
def k_fick(spec: wp.array3d(dtype=wp.int32), spec_new: wp.array3d(dtype=wp.int32),
           hw: wp.array2d(dtype=wp.float64), seed: wp.int32,
           H: int, W: int, ns: int, diff_frac: wp.float64):
    s, i, j = wp.tid()
    n = spec[s, i, j]
    out = int(0)
    for k in range(4):
        out += fick_out(spec, s, i, j, k, hw, seed, H, W, ns, diff_frac)
    inn = int(0)
    if j + 1 < W:
        inn += fick_out(spec, s, i, j + 1, 1, hw, seed, H, W, ns, diff_frac)
    if j - 1 >= 0:
        inn += fick_out(spec, s, i, j - 1, 0, hw, seed, H, W, ns, diff_frac)
    if i + 1 < H:
        inn += fick_out(spec, s, i + 1, j, 3, hw, seed, H, W, ns, diff_frac)
    if i - 1 >= 0:
        inn += fick_out(spec, s, i - 1, j, 2, hw, seed, H, W, ns, diff_frac)
    spec_new[s, i, j] = n - out + inn


def run(label, kernel, ticks=1000):
    F.k_species_diffuse = kernel
    tmp = Path(tempfile.mkdtemp(dir=SCR)); os.chdir(tmp)
    n = 8
    cfg = tiny_cfg(n=n)
    state, sched = build_test_sim(cfg, "cpu", tmp / "run")
    sched.modules = [sched.modules[2]]
    elev = np.zeros((n, n)); elev[:, n // 2:] = 1.5
    h = 2.0 - elev
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
    ip = sched.chem.index["Pi"]
    spec = np.zeros_like(state.species.numpy())
    spec[ip] = np.round(h * 1_000_000).astype(np.int32)
    state.species = wp.array(spec, dtype=wp.int32, device="cpu")
    for t in range(1, ticks + 1):
        sched.tick_once()
    s = state.species.numpy()[ip].astype(np.float64)
    hh = state.water_depth.numpy()
    conc = s * 1e-6 / (hh * 1000.0) * 1000.0
    L = conc[:, : n // 2].mean(); R = conc[:, n // 2:].mean()
    print(f"[{label:6s}] tick {ticks}: deep {L:.4f} mM  shallow {R:.4f} mM  ratio {R/L:.3f}  "
          f"max|u| {np.abs(state.water_u.numpy()).max():.1e}  max|dh| {np.abs(hh-h).max():.1e}  "
          f"total {int(s.sum())}", flush=True)


run("actual", ACTUAL)
run("none", k_nodiff)
run("fick", k_fick)
F.k_species_diffuse = ACTUAL
