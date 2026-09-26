import sys, time
sys.path.insert(0, "/home/nick/Life/firmament")
import numpy as np
import warp as wp
from pathlib import Path
import tempfile
from tests.conftest import tiny_cfg, build_test_sim
from tests.test_phase5_polymers import _life_sim, REPLICASE, PAD

BINDER = "M3M1M3M4M2M4"
dev = "cpu"
tmp = Path(tempfile.mkdtemp(dir="/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/life"))
seq = REPLICASE + BINDER + PAD
cfg, state, sched, (y, x) = _life_sim(tmp, dev, seq=seq, mutation=0.05, pp=500000, monomers=20000)
t0 = time.time()
for t in range(400):
    sched.tick_once()
    if t % 50 == 49:
        st = state.p_state.numpy(); pa = state.p_partner.numpy(); mot = state.p_motifs.numpy()
        alive = np.nonzero((st >= 1) & (st <= 3))[0]
        orphan = [s for s in alive if st[s] == 1 and pa[s] >= 0 and (st[pa[s]] == 0 or pa[pa[s]] != s)]
        print(t+1, "alive", len(alive), "free", int((st==1).sum()), "orphans", len(orphan),
              "nonrepl-orph", sum(1 for s in orphan if not (mot[s] & sched.poly.repl_bit)), f"{time.time()-t0:.1f}s")
