"""Species diffusion acts on COUNTS per cell, not on CONCENTRATION.
Still pond, flat free surface at 2.0 m, all cells wet (no shoreline): left half is
2.0 m deep (bed 0), right half 0.5 m deep (bed 1.5 m). Fluid module only, isothermal,
vapor exactly saturated -> no flow, no evaporation, no rain. Tracer Pi is set to a
UNIFORM concentration of 1 mmol/L everywhere (count = 1 umol -> 2e6 counts left,
5e5 counts right). A uniform concentration is diffusive equilibrium (Fick: flux =
-D grad c = 0). Expected: nothing changes."""
import sys, os, tempfile, math
sys.path.insert(0, "/home/nick/Life/firmament")
from pathlib import Path
import numpy as np
import warp as wp
from tests.conftest import tiny_cfg, build_test_sim
SCR = "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/env"
tmp = Path(tempfile.mkdtemp(dir=SCR)); os.chdir(tmp)
n = 8
cfg = tiny_cfg(n=n)
state, sched = build_test_sim(cfg, "cpu", tmp / "run")
flu = sched.modules[2].__self__
sched.modules = [sched.modules[2]]          # fluid only
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
spec[ip] = np.round(h * 1_000_000).astype(np.int32)       # 1 umol/count, 1 mM everywhere
state.species = wp.array(spec, dtype=wp.int32, device="cpu")
count_mol = 1e-6

def report(t):
    s = state.species.numpy()[ip].astype(np.float64)
    hh = state.water_depth.numpy()
    conc = s * count_mol / (hh * 1000.0) * 1000.0     # mmol/L
    L = conc[:, : n // 2].mean(); R = conc[:, n // 2:].mean()
    print(f"tick {t:5d}: [Pi] deep half {L:.4f} mM, shallow half {R:.4f} mM, ratio {R/L:.3f}; "
          f"max|u| {np.abs(state.water_u.numpy()).max():.1e}  max|dh| {np.abs(hh-h).max():.1e}  "
          f"total Pi {int(s.sum())}")

print(f"diff_frac per tick = {flu.diff_frac}")
report(0)
for t in range(1, 3001):
    sched.tick_once()
    if t in (100, 500, 1000, 3000):
        report(t)
