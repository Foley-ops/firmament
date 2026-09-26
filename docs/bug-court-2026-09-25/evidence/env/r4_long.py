import sys, os, tempfile, time
sys.path.insert(0, "/home/nick/Life/firmament")
from pathlib import Path
import numpy as np
from tests.conftest import tiny_cfg, build_test_sim
import logging
SCR = "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/env"
tmp = Path(tempfile.mkdtemp(dir=SCR)); os.chdir(tmp)
n = int(sys.argv[1]); ticks = int(sys.argv[2]); dt = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
cfg = tiny_cfg(n=n, dt=dt)
cfg.run.audit_every_ticks = 50
state, sched = build_test_sim(cfg, "cpu", tmp / "run")
logging.getLogger("firmament.fluid").setLevel(logging.ERROR)
t0 = time.time()
try:
    for k in range(ticks):
        sched.tick_once()
        if k % 500 == 0:
            T = state.temp.numpy()
            print(k, "T range", [round(float(T[l].min()),1) for l in range(3)], [round(float(T[l].max()),1) for l in range(3)],
                  "vapor", round(float(state.vapor.numpy().mean()),3), "sec", round(time.time()-t0), flush=True)
except Exception as e:
    print("EXC at tick", state.tick, type(e).__name__, e)
print("done", state.tick)
