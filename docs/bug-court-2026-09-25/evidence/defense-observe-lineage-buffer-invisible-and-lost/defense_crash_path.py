"""Defense's independent check: no SIGKILL, no state injection.

A seeded world runs through the REAL crash handler cli._guarded_loop with a periodic
snapshot hook identical to cli._attach_io's maybe_snapshot. The run dies from a natural
exception (polymer capacity exhausted, the same exception that ended the historical
genesis run 20260911-200738). Ground truth for copy events is captured by wrapping
LineageDB.add_copies to also append each event to a side file (observation only).

Question: after the crash, does lineage.sqlite hold the copy events that happened at or
before the last good snapshot (which resume can never regenerate)?
"""
import sys
sys.path.insert(0, "/home/nick/Life/firmament")
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path

DEV = "cpu"
SNAP_EVERY = 100


def child(workdir: str, cap: int):
    import numpy as np
    import warp as wp
    from tests.conftest import tiny_cfg, build_test_sim
    from tests.test_phase5_polymers import SEED_SEQ
    from firmament import cli
    tmp = Path(workdir)
    n = 16
    cfg = tiny_cfg(n=n, water=0.9, cap=cap, seed=4242)
    state, sched = build_test_sim(cfg, DEV, tmp / "run")
    sched.modules = [sched.modules[4]]
    state.water_depth = wp.array(np.full((n, n), 1.0), dtype=wp.float64, device=DEV)
    state.temp = wp.array(np.full((3, n, n), 288.0), dtype=wp.float64, device=DEV)
    spec = np.zeros_like(state.species.numpy())
    idx = sched.chem.index
    y = x = n // 2
    spec[idx["H2O"], :, :] = 10_000_000
    for m in ("M1", "M2", "M3", "M4"):
        spec[idx[m], y, x] = 2000
    spec[idx["PP"], y, x] = 100000
    state.species = wp.array(spec, dtype=wp.int32, device=DEV)
    from firmament.operator.console import place_seed
    place_seed(sched, state, SEED_SEQ, x, y)

    truth = open(tmp / "run" / "truth.jsonl", "a", buffering=1)
    orig = sched.lineage.add_copies

    def spy(tick, events):
        for e in events:
            truth.write(json.dumps([int(tick)] + [int(v) for v in e]) + "\n")
        truth.flush()
        orig(tick, events)
    sched.lineage.add_copies = spy

    def maybe_snapshot(s, tick):           # same shape as cli._attach_io's hook
        if tick > 0 and tick % SNAP_EVERY == 0:
            sched.snapshots.save(s, sched)
    sched.instruments.append(maybe_snapshot)
    # the real crash path: writes crash.json and re-raises (process then exits)
    cli._guarded_loop(sched, cfg, tmp / "run", state, 5000)


def main():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    for cap in (60, 120):
        w = Path(tempfile.mkdtemp(prefix=f"defcrash{cap}_"))
        try:
            p = subprocess.run([sys.executable, __file__, "child", str(w), str(cap)], env=env,
                               capture_output=True, text=True, timeout=600)
            last_line = [l for l in p.stderr.strip().splitlines() if l.strip()][-1]
            crash = json.loads((w / "run" / "crash.json").read_text())
            snaps = sorted((w / "run" / "snapshots").glob("tick_*.zarr"))
            snap_tick = int(snaps[-1].name[5:-5]) if snaps else None
            truth = [json.loads(l) for l in open(w / "run" / "truth.jsonl")]
            c = sqlite3.connect(w / "run" / "lineage.sqlite")
            rows = c.execute("SELECT COUNT(*) FROM copies").fetchone()[0]
            rows_pre = c.execute("SELECT COUNT(*) FROM copies WHERE tick<=?", (snap_tick,)).fetchone()[0]
            c.close()
            t_pre = sum(1 for t in truth if t[0] <= snap_tick)
            print(f"cap={cap}: child exit code {p.returncode}; last stderr line: {last_line}")
            print(f"  crash.json: tick={crash['tick']} exception={crash['exception']}")
            print(f"  last good snapshot tick={snap_tick}")
            print(f"  copy events that happened (ground truth): total {len(truth)}, "
                  f"at/before snapshot {t_pre}")
            print(f"  rows in lineage.sqlite after crash:        total {rows}, "
                  f"at/before snapshot {rows_pre}")
        finally:
            shutil.rmtree(w, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], int(sys.argv[3]))
    else:
        main()
