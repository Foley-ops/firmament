import sys, os, tempfile, time
sys.path.insert(0, "/home/nick/Life/firmament")
from pathlib import Path
import numpy as np
from tests.conftest import tiny_cfg, build_test_sim

tmp = Path(tempfile.mkdtemp(dir="/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/env"))
os.chdir(tmp)
cfg = tiny_cfg(n=16)
cfg.run.audit_every_ticks = 10
state, sched = build_test_sim(cfg, "cpu", tmp / "run")
t0 = time.time()
for k in range(300):
    sched.tick_once()
print("ticks", state.tick, "sec", time.time() - t0)
print("water min", state.water_depth.numpy().min(), "max", state.water_depth.numpy().max())
print("temp", state.temp.numpy().min(), state.temp.numpy().max())
print("vapor", state.vapor.numpy().min(), state.vapor.numpy().max())
