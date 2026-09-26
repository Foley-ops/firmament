"""Defense re-examination of 'offline-event-silently-dropped'.

A) CONTROL: clean stop (shutdown snapshot) -> `firmament event --type drought` works?
B) ACCUSED PATH via plain `kill` (SIGTERM; RUNBOOK: "Ctrl-C (or kill) ... an unclean kill
   loses at most snapshot_every_sim_days of progress") after a GUI rain -> event drought.
   Does not depend on the prosecutor's Ctrl-C claim.
C) In-process: wrap cli._flush to observe sched.mailbox / replay_queue at the moment
   cmd_event is about to exit (direct evidence of 'held then discarded').
CPU only; everything happens in a fresh temp dir under the defense scratch dir.
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = "/home/nick/Life/firmament"
sys.path.insert(0, REPO)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
SCR = Path("/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/"
           "court/defense-history-offline-event-silently-dropped")
PY = f"{REPO}/.venv/bin/python"

import numpy as np  # noqa: E402
import yaml  # noqa: E402
import zarr  # noqa: E402
from tests.conftest import tiny_cfg  # noqa: E402


def cli(wd, *args, check=False):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=REPO)
    r = subprocess.run([PY, "-m", "firmament.cli", "--device", "cpu", *args], cwd=wd, env=env,
                       capture_output=True, text=True, timeout=600)
    if check and r.returncode:
        raise SystemExit(f"CLI failed {args}\n{r.stdout}\n{r.stderr}")
    return r


def setup(tag):
    wd = Path(tempfile.mkdtemp(prefix=tag + "_", dir=SCR))
    cfg = tiny_cfg(n=16, cap=500)          # snapshot_every_sim_days=100000: no periodic snaps
    (wd / "c.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    cli(wd, "run", "--config", str(wd / "c.yaml"), "--ticks", "5", check=True)
    rid = sorted(p.name for p in (wd / "runs").iterdir() if (p / "meta.json").exists())[-1]
    return wd, rid, wd / "runs" / rid


def evs(run):
    return [(e["n"], e["tick"], e["event"]) for e in
            (json.loads(x) for x in open(run / "events.jsonl") if x.strip())]


def snaps(run):
    return sorted(p.name for p in (run / "snapshots").glob("tick_*.zarr"))


def causal_n(p):
    return zarr.open_group(str(p), mode="r").attrs["meta"]["causal_n"]


def vapor(p):
    return float(zarr.open_group(str(p), mode="r")["vapor"][:].sum())


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def gui_rain_then_kill(wd, rid, run, sig):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=REPO)
    proc = subprocess.Popen([PY, "-m", "firmament.cli", "--device", "cpu", "resume", "--run", rid],
                            cwd=wd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    port, t0 = None, time.time()
    while time.time() - t0 < 60 and port is None:
        try:
            m = json.loads((run / "meta.json").read_text())
            if m.get("pid") == proc.pid and m.get("port"):
                port = m["port"]
        except Exception:
            pass
        time.sleep(0.1)
    post(port, "/api/time", {"mode": "throttled", "target_tps": 30})
    tok = post(port, "/api/console/rain", {"intensity_mm": 5.0})["confirm_token"]
    print("   GUI:", post(port, "/api/console/rain", {"intensity_mm": 5.0, "confirm_token": tok}))
    while not evs(run):
        time.sleep(0.05)
    time.sleep(0.3)
    proc.send_signal(sig)
    rc = proc.wait(timeout=60)
    print(f"   sent {sig.name}; child rc={rc}; events.jsonl={evs(run)}; snapshots={snaps(run)}")


# ---------------- A) control ----------------
print("A) CONTROL (clean stop)")
wd, rid, run = setup("A")
print("   after run --ticks 5: snapshots", snaps(run), "events", evs(run))
r = cli(wd, "event", "--run", rid, "--type", "drought", "--params", '{"strength": 0.9}')
print(f"   event drought rc={r.returncode} stdout={r.stdout.strip()!r}")
print("   events.jsonl:", evs(run))
s5, s6 = run / "snapshots" / "tick_000000000005.zarr", run / "snapshots" / "tick_000000000006.zarr"
print(f"   tick6 causal_n={causal_n(s6)}  vapor tick5={vapor(s5):.4f} tick6={vapor(s6):.4f}")

# ---------------- B) accused path via SIGTERM ----------------
print("B) ACCUSED PATH via plain `kill` (SIGTERM) after a GUI rain")
wd, rid, run = setup("B")
gui_rain_then_kill(wd, rid, run, signal.SIGTERM)
r = cli(wd, "event", "--run", rid, "--type", "drought", "--params", '{"strength": 0.9}')
print(f"   event drought rc={r.returncode} stdout={r.stdout.strip()!r} stderr_tail={r.stderr.strip()[-200:]!r}")
print("   events.jsonl:", evs(run))
latest = run / "snapshots" / snaps(run)[-1]
print(f"   latest {latest.name} causal_n={causal_n(latest)}  "
      f"vapor tick5={vapor(run / 'snapshots' / 'tick_000000000005.zarr'):.4f} "
      f"{latest.name}={vapor(latest):.4f}")
r = cli(wd, "resume", "--run", rid, "--ticks", "100", check=True)
print("   after resume --ticks 100 events.jsonl:", evs(run),
      " drought present:", any(e[2] == "drought" for e in evs(run)))

# ---------------- C) in-process: what is in the mailbox when cmd_event exits ----------------
print("C) IN-PROCESS: mailbox / replay_queue at cmd_event exit (SIGKILL variant)")
wd, rid, run = setup("C")
gui_rain_then_kill(wd, rid, run, signal.SIGKILL)
os.chdir(wd)                                         # CLI writes ./runs relative to cwd
from firmament import cli as fcli  # noqa: E402
orig = fcli._flush
seen = {}


def spy(sched):
    seen["mailbox"] = list(sched.mailbox)
    seen["replay_queue"] = [(r["n"], r["tick"], r["event"]) for r in sched.replay_queue]
    seen["tick"] = sched.state.tick
    seen["causal_n"] = sched.state.causal_n
    return orig(sched)


fcli._flush = spy
fcli.main(["--device", "cpu", "event", "--run", rid, "--type", "drought",
           "--params", '{"strength": 0.9}'])
print("   at exit:", seen)
print("   events.jsonl:", evs(run))
