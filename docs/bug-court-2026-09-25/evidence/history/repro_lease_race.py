"""Lease TOCTOU probe: several writers start at once on a run whose .lease was left by a
crashed (dead) pid. Single-writer rule (DECISIONS 2026-09-25): at most ONE may win."""
import json
import multiprocessing as mp
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/nick/Life/firmament")
SCR = "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/history"


def worker(d, b1, b2, q):
    from firmament.io import rundir
    b1.wait()
    try:
        rundir.acquire_lease(Path(d))
        q.put(os.getpid())
    except rundir.LeaseError:
        q.put(None)
    b2.wait()                       # everyone stays alive until all have tried


def dead_pid():
    p = mp.get_context("fork").Process(target=lambda: None)
    p.start()
    p.join()
    return p.pid


if __name__ == "__main__":
    ctx = mp.get_context("fork")
    N, TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 12, 150
    doubles = []
    for t in range(TRIALS):
        d = tempfile.mkdtemp(prefix="lease_", dir=SCR)
        Path(d, ".lease").write_text(json.dumps({"pid": dead_pid(), "since": "crash"}))
        b1, b2, q = ctx.Barrier(N), ctx.Barrier(N), ctx.Queue()
        ps = [ctx.Process(target=worker, args=(d, b1, b2, q)) for _ in range(N)]
        for p in ps:
            p.start()
        winners = [w for w in (q.get() for _ in range(N)) if w is not None]
        for p in ps:
            p.join()
        if len(winners) > 1:
            doubles.append((t, winners))
    print(f"{TRIALS} trials x {N} simultaneous writers on a stale lease")
    print(f"trials where MORE THAN ONE writer acquired the single-writer lease: {len(doubles)}")
    for t, w in doubles[:5]:
        print(f"  trial {t}: lease granted to pids {w}")
