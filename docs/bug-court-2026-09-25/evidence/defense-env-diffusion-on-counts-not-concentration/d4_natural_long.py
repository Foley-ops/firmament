"""Defense probe B: NATURAL world (tiny_cfg n=32, full 6-module stack, nothing injected).
Measure how dissolved Pi concentration relates to water depth over time.
slope = OLS slope of log(conc) on log(depth) over wet cells: -1 => conc ~ 1/depth
(equal counts per cell), 0 => depth-independent concentration.
arg1: 'actual' or 'fick' (Fickian counterfactual kernel monkeypatched, no repo edit)."""
import sys, os, tempfile, logging
sys.path.insert(0, "/home/nick/Life/firmament")
sys.path.insert(0, "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/defense-env-diffusion-on-counts-not-concentration")
from pathlib import Path
import numpy as np
import firmament.core.fluid as F
mode = sys.argv[1]
ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
if mode == "fick":
    import importlib.util
    spec = importlib.util.spec_from_file_location("d1k", "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/defense-env-diffusion-on-counts-not-concentration/fickk.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    F.k_species_diffuse = m.k_fick
from tests.conftest import tiny_cfg, build_test_sim
logging.getLogger().setLevel(logging.ERROR)
for name in ("fluid", "firmament", "firmament.fluid"):
    logging.getLogger(name).setLevel(logging.ERROR)
SCR = "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/defense-env-diffusion-on-counts-not-concentration"
tmp = Path(tempfile.mkdtemp(dir=SCR)); os.chdir(tmp)
n = int(sys.argv[3]) if len(sys.argv) > 3 else 32
state, sched = build_test_sim(tiny_cfg(n=n), "cpu", tmp / "run")
ci = {k: sched.chem.index[k] for k in ("Pi", "PP", "N2")}

def report(t):
    h = state.water_depth.numpy(); sp = state.species.numpy()
    wet = h > 1e-2
    out = [f"[{mode}] tick {t:5d} wet={wet.sum()}"]
    for k in ("Pi", "N2"):
        c = sp[ci[k]][wet].astype(float) * 1e-6 / (h[wet] * 1000) * 1e3   # mM
        ok = c > 0
        slope = np.polyfit(np.log(h[wet][ok]), np.log(c[ok]), 1)[0]
        sh = c[(h[wet] < 0.5)].mean(); dp = c[(h[wet] > 2.0)].mean()
        out.append(f"{k}: slope {slope:+.3f} shallow(<0.5m) {sh:.4f} mM deep(>2m) {dp:.4f} mM ratio {sh/dp:.2f}")
    print(" | ".join(out), flush=True)

report(0)
for t in range(1, ticks + 1):
    sched.tick_once()
    if t % 1000 == 0:
        report(t)
