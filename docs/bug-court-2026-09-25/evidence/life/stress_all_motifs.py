import sys
sys.path.insert(0, "/home/nick/Life/firmament")
import tempfile, time
from pathlib import Path
import numpy as np
import warp as wp
from tests.conftest import tiny_cfg, build_test_sim, REPO
from firmament.core.audit import Audit
import yaml

SCR = "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/life"
code = yaml.safe_load(open(REPO / "configs/genetic_code_v0.yaml"))
allm = "".join("".join(code["motifs"][n]["seq"]) for n in code["motifs"])
seq = allm + "M1M3M2M4" * 2
mode = sys.argv[1] if len(sys.argv) > 1 else "poly"
tmp = Path(tempfile.mkdtemp(dir=SCR))
n = 16
cfg = tiny_cfg(n=n, water=0.9, cap=6000, seed=777)
cfg.polymers.mutation_rate_per_monomer = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
cfg.polymers.hydrolysis_base_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
state, sched = build_test_sim(cfg, "cpu", tmp / "run")
if mode == "poly":
    sched.modules = [sched.modules[4]]
idx = sched.chem.index
spec = state.species.numpy()
y = x = n // 2
state.water_depth = wp.array(np.maximum(state.water_depth.numpy(), 0.5), dtype=wp.float64, device="cpu") if mode=="poly" else state.water_depth
spec[idx["H2O"]] = np.maximum(spec[idx["H2O"]], 1_000_000)
for m in ("M1", "M2", "M3", "M4"):
    spec[idx[m], y-2:y+3, x-2:x+3] += 5000
spec[idx["PP"], y-2:y+3, x-2:x+3] += 50000
spec[idx["L"], y-2:y+3, x-2:x+3] += 5000
spec[idx["HCN"], y-2:y+3, x-2:x+3] += 5000
spec[idx["CH2O"], y-2:y+3, x-2:x+3] += 5000
state.species = wp.array(spec, dtype=wp.int32, device="cpu")
from firmament.operator.console import place_seed
place_seed(sched, state, seq, x, y)
aud = Audit(cfg, sched.chem, tmp)
e0 = aud.element_totals(state)
t0 = time.time()
for t in range(1, 601):
    sched.tick_once()
    e = aud.element_totals(state)
    if not np.array_equal(e, e0):
        print("ELEMENT MISMATCH at tick", t, (e - e0).tolist()); break
    if t % 100 == 0:
        st = state.p_state.numpy()
        print(t, "live", int((st > 0).sum()), "build", int((st == 5).sum()), "mem", int(state.membrane_store.numpy().sum()), f"{time.time()-t0:.1f}s")
print("done")
