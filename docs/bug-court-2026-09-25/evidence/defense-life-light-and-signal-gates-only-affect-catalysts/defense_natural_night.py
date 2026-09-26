"""Natural-path check: FULL module stack (radiation, fluid, chemistry, polymers...), no
array injection except granting resources in the seed cell before event #0. The real
radiation module makes it night at tick 0 (hour angle = -pi). Compare a
photoactive+membrane probe with a plain membrane probe, and photoactive+catalyst with
plain catalyst, over the first night-time ticks."""
import sys
sys.path.insert(0, "/home/nick/Life/firmament")
import tempfile
from pathlib import Path
import numpy as np
import warp as wp
import yaml
from tests.conftest import tiny_cfg, build_test_sim, REPO
from firmament.core.radiation import sun_cos_zenith

SCR = Path("/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/"
           "court/defense-life-light-and-signal-gates-only-affect-catalysts")
code = yaml.safe_load(open(REPO / "configs/genetic_code_v0.yaml"))
M = {k: "".join(v["seq"]) for k, v in code["motifs"].items()}
PAD = "M1M3M2M4"
n = 16
TICKS = 20


def run(seq):
    cfg = tiny_cfg(n=n, water=0.9, cap=200, seed=31)
    state, sched = build_test_sim(cfg, "cpu", Path(tempfile.mkdtemp(dir=SCR)) / "run")
    idx = sched.chem.index
    wd = state.water_depth.numpy()
    ys, xs = np.nonzero(wd > 0.05)
    y, x = int(ys[len(ys) // 2]), int(xs[len(xs) // 2])     # a wet cell
    spec = state.species.numpy()
    for m in ("M1", "M2", "M3", "M4"):
        spec[idx[m], y, x] += 500
    spec[idx["PP"], y, x] += 10_000
    spec[idx["L"], y, x] += 1000
    state.species = wp.array(spec, dtype=wp.int32, device="cpu")
    from firmament.operator.console import place_seed
    place_seed(sched, state, seq, x, y)
    rx = [r["name"] for r in sched.chem.reactions].index("formose")
    lw_max, catsum, cells = 0.0, 0.0, set()
    for _ in range(TICKS):
        sched.tick_once()
        lw_max = max(lw_max, float(state.light_water.numpy().max()))
        catsum += float(state.catalyst.numpy()[rx].sum())
    return dict(modules=[type(m).__name__ for m in sched.modules], lw_max=lw_max,
                mem=int(state.membrane_store.numpy().sum()), cat=catsum,
                alive=int(state.p_state.numpy()[0]), depth=float(wd[y, x]))


print("cos zenith at ticks 0, 10, 20, 360, 720:",
      [round(sun_cos_zenith(t, tiny_cfg(n=8)), 3) for t in (0, 10, 20, 360, 720)])
for name, seq in [("membrane", M["membrane"] + PAD),
                  ("photoactive+membrane", M["photoactive"] + PAD + M["membrane"] + PAD),
                  ("catalyst_formose", M["catalyst_formose"] + PAD),
                  ("photoactive+catalyst_formose", M["photoactive"] + PAD + M["catalyst_formose"] + PAD)]:
    r = run(seq)
    print(f"{name:30s} max light_water over {TICKS} ticks={r['lw_max']:.1f}  "
          f"membrane_store total={r['mem']:3d}  sum(formose catalyst field over ticks)={r['cat']:.1f}  "
          f"alive={r['alive']}  depth={r['depth']:.2f}")
print("modules:", r["modules"])
