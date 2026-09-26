"""Defense repro: exercise each documented way of stopping a run, then resume, and
compare history (metrics parquet + lineage.sqlite) with an uninterrupted run.

Usage: stop_paths.py MODE   where MODE in ref | SIGINT_PG | SIGTERM | SIGKILL | CRASH
  ref       : uninterrupted run tick 1 -> 600 (--ticks bounded, clean exit)
  SIGINT_PG : terminal-style Ctrl-C = SIGINT to the whole foreground process group of
              `uv run python -m firmament.cli resume ...` (RUNBOOK invocation)
  SIGTERM   : `kill <pid>` using the pid the run itself records in meta.json
  SIGKILL   : `kill -9 <pid>` (RUNBOOK's "unclean kill")
  CRASH     : a Python exception inside the loop at tick 316 (Phase 10 "forced crash")
Writes <scratch>/result_MODE.json.
Metrics cadence mirrors real configs: flush period (every*100 = 200 ticks) is shorter
than and misaligned with the snapshot period (300 ticks), as in world_small.yaml
(flush every 10000 ticks vs snapshot every 43200 ticks).
"""
import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = "/home/nick/Life/firmament"
sys.path.insert(0, REPO)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
PY = f"{REPO}/.venv/bin/python"
SNAP, END, STOP_AT = 300, 600, 316
OVR = {m: {"init_wet": 400} for m in ("M1", "M2", "M3", "M4")} | {"PP": {"init_wet": 20000}}
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=REPO)


def cli(wd, *a, check=True):
    r = subprocess.run([PY, "-m", "firmament.cli", "--device", "cpu", *a], cwd=wd, env=ENV,
                       capture_output=True, text=True, timeout=600)
    if check and r.returncode:
        raise SystemExit(f"CLI failed {a}\n{r.stdout}\n{r.stderr[-3000:]}")
    return r


def seeded_run(tag):
    import numpy as np
    import yaml
    from firmament.config import Config
    from tests.conftest import build_test_sim, tiny_cfg
    from tests.test_phase5_polymers import SEED_SEQ
    wd = Path(tempfile.mkdtemp(prefix=f"{tag}_", dir=HERE))
    cfg = tiny_cfg(n=16, cap=2000)
    cfg.run.snapshot_every_sim_days = SNAP * 60.0 / 86400.0
    cfg.run.metrics_every_ticks = 2
    cfg.chemistry.overrides = OVR
    p = wd / "c.yaml"
    p.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    cli(wd, "run", "--config", str(p), "--ticks", "1")
    rid = sorted(x.name for x in (wd / "runs").iterdir() if (x / "meta.json").exists())[-1]
    s, _ = build_test_sim(Config.load(p), "cpu", wd / "probe")
    y, x = np.argwhere(s.water_depth.numpy() > 0.3)[0]
    cli(wd, "seed", "--run", rid, "--sequence", SEED_SEQ, "--cell", f"{int(x)},{int(y)}")
    return wd, rid


def history(wd, rid):
    from firmament.io.metrics import MetricsWriter
    t = MetricsWriter(wd / "runs" / rid / "metrics").read_all()
    m = [] if t is None else sorted(t.column("tick").to_pylist())
    db = sqlite3.connect(wd / "runs" / rid / "lineage.sqlite")
    lin = db.execute("SELECT child_id,parent_id,tick FROM copies ORDER BY child_id").fetchall()
    snaps = sorted(x.name for x in (wd / "runs" / rid / "snapshots").iterdir())
    return m, [list(r) for r in lin], snaps


def last_tick_logged(wd, rid):
    ts = []
    for f in (wd / "runs" / rid / "logs").glob("sim.log*"):
        for ln in open(f):
            if ln.startswith("{"):
                ts.append(json.loads(ln).get("tick", 0))
    return max(ts)


CRASH_DRIVER = f"""
import sys; sys.path.insert(0, {REPO!r})
import firmament.cli as c
orig = c.build_sim
def sab(cfg, rd, dev):
    st, sc = orig(cfg, rd, dev)
    def bomb(s, tick):
        if tick == {STOP_AT}:
            raise RuntimeError("injected fault")
    sc.add(bomb)
    return st, sc
c.build_sim = sab
c.main(sys.argv[1:])
"""


def main(mode):
    wd, rid = seeded_run(mode)
    out = {"mode": mode}
    if mode == "ref":
        cli(wd, "resume", "--run", rid, "--ticks", str(END - 1))
    else:
        target = wd / "runs" / rid / "snapshots" / f"tick_{SNAP:012d}.zarr"
        if mode == "CRASH":
            r = subprocess.run([PY, "-c", CRASH_DRIVER, "--device", "cpu", "resume", "--run", rid],
                               cwd=wd, env=ENV, capture_output=True, text=True, timeout=600)
            out["stop_rc"] = r.returncode
            out["stop_stderr_tail"] = r.stderr.strip().splitlines()[-1]
            out["crash_json"] = json.loads((wd / "runs" / rid / "crash.json").read_text())
            out["crash_json"].pop("traceback", None)
        else:
            if mode == "SIGINT_PG":      # exactly what a terminal Ctrl-C does
                cmd = ["uv", "run", "--project", REPO, "python", "-m", "firmament.cli",
                       "--device", "cpu", "resume", "--run", rid]
            else:
                cmd = [PY, "-m", "firmament.cli", "--device", "cpu", "resume", "--run", rid]
            proc = subprocess.Popen(cmd, cwd=wd, env=ENV, stdout=open(wd / "child.out", "w"),
                                    stderr=open(wd / "child.err", "w"), start_new_session=True,
                                    preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
            t0 = time.time()
            while not target.exists() and time.time() - t0 < 300:
                time.sleep(0.01)
            time.sleep(0.5)
            meta = json.loads((wd / "runs" / rid / "meta.json").read_text())
            if mode == "SIGINT_PG":
                os.killpg(proc.pid, signal.SIGINT)
            else:
                os.kill(meta["pid"], getattr(signal, mode))     # pid recorded by the run
            proc.wait(timeout=120)
            time.sleep(0.5)
            out["stop_rc"] = proc.returncode
            err = [x for x in (wd / "child.err").read_text().splitlines() if x.strip()]
            out["stop_stderr_tail"] = err[-1] if err else ""
        out["tick_reached"] = last_tick_logged(wd, rid)
        m, lin, snaps = history(wd, rid)
        out["after_stop"] = {"snapshots": snaps, "metric_rows": len(m),
                             "metric_ticks_range": [m[0], m[-1]] if m else None,
                             "lineage_rows": len(lin)}
        cli(wd, "resume", "--run", rid, "--ticks", str(END - SNAP))
    m, lin, snaps = history(wd, rid)
    out["final"] = {"snapshots_tail": snaps[-1], "metric_ticks": m, "lineage": lin}
    out["wd"] = str(wd)
    (HERE / f"result_{mode}.json").write_text(json.dumps(out))
    print(mode, "done", json.dumps({k: v for k, v in out.items() if k != "final"}))


if __name__ == "__main__":
    main(sys.argv[1])
