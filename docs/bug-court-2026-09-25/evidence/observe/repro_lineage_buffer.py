"""LineageDB buffers copy events in RAM and writes them only when 10,000 are buffered
(or on clean shutdown). Nothing else flushes it: not snapshots, not the crash path.

(1) Readers query sqlite only, so on a live run the GUI's /api/inspect/polymer
    lineage_path and the candidate_mutant_depth100 detector (lineage.depth_of) see an
    EMPTY tree until 10k copies have happened.
(2) An unclean kill (RUNBOOK: "loses at most snapshot_every_sim_days of progress")
    loses every buffered copy event, including those BEFORE the last snapshot; resume
    only regenerates events after the snapshot, so they are gone for good.
"""
import sys
sys.path.insert(0, "/home/nick/Life/firmament")
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import tempfile
from pathlib import Path

DEV = "cpu"
SNAP_T, END_T = 200, 230


def child(workdir: str, mode: str):
    """mode=kill: seed, run to SNAP_T, snapshot, run to END_T, SIGKILL (unclean kill).
       mode=ref : same but uninterrupted, clean flush at END_T."""
    from tests.test_phase5_polymers import _life_sim
    tmp = Path(workdir)
    cfg, s, sched, _ = _life_sim(tmp, DEV)
    while s.tick < SNAP_T:
        sched.tick_once()
    sched.snapshots.save(s, sched)
    while s.tick < END_T:
        sched.tick_once()
    if mode == "kill":
        print(json.dumps({"buffered_in_ram": len(sched.lineage.buf)}), flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
    sched.lineage.flush()


def resume_after_kill(workdir: str):
    """What `firmament resume` does: load the last snapshot, re-attach the run's
    lineage DB, replay causal history, continue to END_T, shut down cleanly."""
    from tests.conftest import build_test_sim
    from tests.test_phase5_polymers import _life_sim  # noqa: F401 (same cfg builder)
    from tests.conftest import tiny_cfg
    from firmament.io.events import replay_events
    tmp = Path(workdir)
    cfg = tiny_cfg(n=16, water=0.9, cap=3000, seed=4242)
    s, sched = build_test_sim(cfg, DEV, tmp / "run")
    sched.modules = [sched.modules[4]]
    sched.snapshots.load(s, sched)
    replay_events(sched, tmp / "run" / "events.jsonl")
    while s.tick < END_T:
        sched.tick_once()
    sched.lineage.flush()


def rows(db, where=""):
    c = sqlite3.connect(db)
    n = c.execute("SELECT COUNT(*) FROM copies " + where).fetchone()[0]
    c.close()
    return n


def live_readers():
    """(1) On a live run with instruments + API attached."""
    from tests.test_phase5_polymers import _life_sim
    from firmament.instruments.sampler import attach_instruments
    from firmament.server.api import make_app
    import firmament.io.rundir as rd
    import numpy as np
    tmp = Path(tempfile.mkdtemp(prefix="linA_"))
    try:
        cfg, s, sched, _ = _life_sim(tmp, DEV)
        cfg.run.metrics_every_ticks = 10
        attach_instruments(sched, cfg, tmp / "run")
        while s.tick < SNAP_T:
            sched.tick_once()
        (tmp / "run" / "meta.json").write_text(json.dumps({"run_id": "x", "touched": False}))
        app = make_app(sched, tmp / "run")
        ep = {r.path: r.endpoint for r in app.routes if hasattr(r, "endpoint")}
        v = sched.latest_view
        st = v["p_state"]
        alive = np.nonzero((st >= 1) & (st <= 3))[0]
        grandkids = [int(v["p_id"][p]) for p in alive if int(v["p_parent"][p]) > 1]
        kids = [int(v["p_id"][p]) for p in alive if int(v["p_parent"][p]) == 1]
        pid = (grandkids or kids)[-1]
        print(f"[1] tick {s.tick}: completed polymers {len(alive)}, copy events buffered in RAM "
              f"{len(sched.lineage.buf)}, rows in lineage.sqlite {rows(tmp / 'run' / 'lineage.sqlite')}")
        before = ep["/api/inspect/polymer"](id=pid)
        d_before = sched.lineage.depth_of(pid)
        sched.lineage.flush()
        after = ep["/api/inspect/polymer"](id=pid)
        d_after = sched.lineage.depth_of(pid)
        print(f"[1] /api/inspect/polymer?id={pid} (parent #{before['parent']}): "
              f"lineage_path={before['lineage_path']}  depth_of={d_before}")
        print(f"[1] same query after a manual lineage.flush(): "
              f"lineage_path={after['lineage_path']}  depth_of={d_after}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def kill_and_resume():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    ref = Path(tempfile.mkdtemp(prefix="linREF_"))
    kil = Path(tempfile.mkdtemp(prefix="linKILL_"))
    try:
        subprocess.run([sys.executable, __file__, "child", str(ref), "ref"], env=env,
                       check=True, capture_output=True, timeout=600)
        p = subprocess.run([sys.executable, __file__, "child", str(kil), "kill"], env=env,
                           capture_output=True, text=True, timeout=600)
        print(f"[2] killed child: returncode={p.returncode} (-9 = SIGKILL), "
              f"stdout={p.stdout.strip()}")
        db_k = kil / "run" / "lineage.sqlite"
        db_r = ref / "run" / "lineage.sqlite"
        print(f"[2] reference (never stopped) rows: total {rows(db_r)}, "
              f"tick<={SNAP_T}: {rows(db_r, f'WHERE tick<={SNAP_T}')}")
        print(f"[2] after unclean kill rows      : total {rows(db_k)}")
        subprocess.run([sys.executable, __file__, "resume", str(kil)], env=env,
                       check=True, capture_output=True, timeout=600)
        print(f"[2] after resume from tick-{SNAP_T} snapshot to {END_T}: total {rows(db_k)}, "
              f"tick<={SNAP_T}: {rows(db_k, f'WHERE tick<={SNAP_T}')}  "
              f"(tick>{SNAP_T}: {rows(db_k, f'WHERE tick>{SNAP_T}')} vs ref "
              f"{rows(db_r, f'WHERE tick>{SNAP_T}')})")
    finally:
        shutil.rmtree(ref, ignore_errors=True)
        shutil.rmtree(kil, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 1 and sys.argv[1] == "resume":
        resume_after_kill(sys.argv[2])
    else:
        live_readers()
        kill_and_resume()
