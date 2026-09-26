"""Defense probe B0: natural world depth structure (no injection)."""
import sys, os, tempfile, time
sys.path.insert(0, "/home/nick/Life/firmament")
from pathlib import Path
import numpy as np
from tests.conftest import tiny_cfg, build_test_sim
SCR = "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/defense-env-diffusion-on-counts-not-concentration"
tmp = Path(tempfile.mkdtemp(dir=SCR)); os.chdir(tmp)
for n in (16, 32):
    cfg = tiny_cfg(n=n)
    state, sched = build_test_sim(cfg, "cpu", tmp / f"run{n}")
    h = state.water_depth.numpy()
    wet = h > 1e-3
    print(f"n={n}: modules={[getattr(m,'__qualname__',m) for m in sched.modules]}")
    print(f"  wet cells {wet.sum()}/{n*n}; wet depth min {h[wet].min():.4f} median {np.median(h[wet]):.4f} max {h[wet].max():.4f}")
    rat = []
    for i in range(n):
        for j in range(n):
            for di, dj in ((0,1),(1,0)):
                a, b = i+di, j+dj
                if a < n and b < n and h[i,j] > 1e-4 and h[a,b] > 1e-4:
                    rat.append(max(h[i,j],h[a,b])/min(h[i,j],h[a,b]))
    rat = np.array(rat)
    print(f"  wet-wet adjacent depth ratio: median {np.median(rat):.2f}, p90 {np.percentile(rat,90):.2f}, max {rat.max():.1f}")
    ip = sched.chem.index["Pi"]
    s = state.species.numpy()[ip]
    print(f"  init Pi counts on wet cells: unique {np.unique(s[wet])[:5]}")
    t0 = time.time()
    for _ in range(20): sched.tick_once()
    print(f"  20 full ticks: {time.time()-t0:.1f}s")
