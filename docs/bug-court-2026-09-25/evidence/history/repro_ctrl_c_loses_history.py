"""EXHIBIT B: stopping a run the documented way (RUNBOOK 'Stop': Ctrl-C) writes no
shutdown snapshot and permanently loses every buffered metrics row and lineage row up to
the last snapshot. Resume starts AFTER that snapshot, so those rows are never
regenerated: the resumed run's history differs from an uninterrupted run's.

Sources: Writeup table 'Snapshot ... on shutdown'; RUNBOOK 'Stop' + 'Resume ... The
result is indistinguishable from never having stopped'; RUNBOOK 'metrics/*.parquet,
lineage.sqlite — never rotated away'; Build guide Phase 10 'a run has survived a forced
crash and resumed correctly'."""
import signal
import sqlite3
import subprocess
import sys
import time

sys.path.insert(0, "/home/nick/Life/firmament")
sys.path.insert(0, "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/history")
import os  # noqa: E402

from common import PY, REPO, cli, only_run, workdir, write_cfg  # noqa: E402

SNAP = 300          # periodic snapshot every 300 ticks
END = 600
OVR = {m: {"init_wet": 400} for m in ("M1", "M2", "M3", "M4")} | {"PP": {"init_wet": 20000}}


def seeded_run(tag):
    wd = workdir(tag)
    cfg = write_cfg(wd, n=16, cap=2000, snap_ticks=SNAP, metrics_every=5, overrides=OVR)
    cli(wd, "run", "--config", str(cfg), "--ticks", "1", check=True)
    rid = only_run(wd)
    # a wet cell, chosen exactly as tests/test_causal_history.py does
    import numpy as np
    from firmament.config import Config
    from tests.conftest import build_test_sim
    c = Config.load(cfg)
    s, _ = build_test_sim(c, "cpu", wd / "probe")
    y, x = np.argwhere(s.water_depth.numpy() > 0.3)[0]
    from tests.test_phase5_polymers import SEED_SEQ
    cli(wd, "seed", "--run", rid, "--sequence", SEED_SEQ, "--cell", f"{int(x)},{int(y)}", check=True)
    return wd, rid


def metrics_ticks(wd, rid):
    from firmament.io.metrics import MetricsWriter
    t = MetricsWriter(wd / "runs" / rid / "metrics").read_all()
    return [] if t is None else sorted(t.column("tick").to_pylist())


def lineage_rows(wd, rid):
    db = sqlite3.connect(wd / "runs" / rid / "lineage.sqlite")
    return db.execute("SELECT child_id, parent_id, tick FROM copies ORDER BY child_id").fetchall()


def snaps(wd, rid):
    return sorted(p.name for p in (wd / "runs" / rid / "snapshots").iterdir())


# --- reference: uninterrupted run 1 -> END -----------------------------------------
wdA, ridA = seeded_run("B_straight")
cli(wdA, "resume", "--run", ridA, "--ticks", str(END - 1), check=True)

# --- same world, stopped with Ctrl-C after the tick-300 snapshot, then resumed ------
wdB, ridB = seeded_run("B_ctrlc")
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=REPO)
proc = subprocess.Popen([PY, "-m", "firmament.cli", "--device", "cpu", "resume", "--run", ridB],
                        cwd=wdB, env=env, stdout=open(wdB / "child.out", "w"),
                        stderr=open(wdB / "child.err", "w"), text=True,
                        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
target = wdB / "runs" / ridB / "snapshots" / f"tick_{SNAP:012d}.zarr"
t0 = time.time()
while not target.exists() and time.time() - t0 < 120:
    time.sleep(0.01)
time.sleep(0.5)                     # let it get past the snapshot
proc.send_signal(signal.SIGINT)     # the RUNBOOK's "Stop: Ctrl-C"
proc.wait(timeout=60)
err = (wdB / "child.err").read_text()
print("Ctrl-C: exit code", proc.returncode, "| last stderr line:",
      [x for x in err.strip().splitlines() if x.strip()][-1])
import json  # noqa: E402
ticks_logged = [json.loads(x).get("tick", 0) for x in open(wdB / "runs" / ridB / "logs" / "sim.log")
                if x.strip().startswith("{")]
print("sim had reached tick ~%d when stopped" % max(ticks_logged))
print("snapshots after Ctrl-C       :", snaps(wdB, ridB))
print("metrics rows on disk after Ctrl-C:", len(metrics_ticks(wdB, ridB)))
print("lineage rows on disk after Ctrl-C:", len(lineage_rows(wdB, ridB)))

cli(wdB, "resume", "--run", ridB, "--ticks", str(END - SNAP), check=True)

ma, mb = metrics_ticks(wdA, ridA), metrics_ticks(wdB, ridB)
la, lb = lineage_rows(wdA, ridA), lineage_rows(wdB, ridB)
print()
print("final tick snapshot A/B     :", snaps(wdA, ridA)[-1], snaps(wdB, ridB)[-1])
print("metrics rows  uninterrupted :", len(ma), "ticks", ma[:3], "...", ma[-2:])
print("metrics rows  Ctrl-C+resume :", len(mb), "ticks", mb[:3], "...", mb[-2:])
missing = sorted(set(ma) - set(mb))
print("metrics ticks MISSING after resume:", len(missing), "from", missing[:1], "to", missing[-1:])
print("lineage rows  uninterrupted :", len(la))
print("lineage rows  Ctrl-C+resume :", len(lb))
lost = sorted(set(la) - set(lb))
print("lineage copy records MISSING after resume:", len(lost),
      "(ticks %s..%s)" % (lost[0][2], lost[-1][2]) if lost else "")
if lost:
    # a surviving descendant whose ancestry now dead-ends
    lost_ids = {c for c, _, _ in lost}
    orphans = [(c, p) for c, p, _ in lb if p in lost_ids]
    print("records in B whose parent's copy record is gone:", len(orphans), orphans[:3])
# the final world state itself is identical (the physics is fine; the HISTORY is not)
import numpy as np  # noqa: E402
import zarr  # noqa: E402
ga = zarr.open_group(str(wdA / "runs" / ridA / "snapshots" / snaps(wdA, ridA)[-1]), mode="r")
gb = zarr.open_group(str(wdB / "runs" / ridB / "snapshots" / snaps(wdB, ridB)[-1]), mode="r")
print("final state arrays identical:", all(np.array_equal(ga[k][:], gb[k][:]) for k in ga.array_keys()))
