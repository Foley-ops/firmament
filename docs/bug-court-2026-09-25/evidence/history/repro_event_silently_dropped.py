"""EXHIBIT C: `firmament event` reports "<type> applied at tick T" but the command is
never logged or applied when the run still has logged causal history waiting to be
replayed (any event committed after the last snapshot, e.g. a GUI rain followed by the
RUNBOOK's Ctrl-C stop). causal.apply_due HOLDS live commands until replay catches up;
cmd_event runs exactly one tick, saves a snapshot and exits, discarding the mailbox.

Sources: cli.cmd_event docstring "Apply a natural event to a resumable run"; its own
output "applied at tick"; causal.py docstring: commands are COMMITTED at a tick boundary
and written to events.jsonl; RUNBOOK: events.jsonl = "the run's definition"."""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, "/home/nick/Life/firmament")
sys.path.insert(0, "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/history")
from common import PY, REPO, cli, events, last_line, only_run, workdir, write_cfg  # noqa: E402


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


wd = workdir("C_event")
cfg = write_cfg(wd)                                   # no periodic snapshots in this window
cli(wd, "run", "--config", str(cfg), "--ticks", "5", check=True)
rid = only_run(wd)
run = wd / "runs" / rid
print("1) run --ticks 5 -> snapshots", sorted(p.name for p in (run / "snapshots").iterdir()))

# 2) live sim; the Operator issues rain from the GUI console, then stops with Ctrl-C
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=REPO)
proc = subprocess.Popen([PY, "-m", "firmament.cli", "--device", "cpu", "resume", "--run", rid],
                        cwd=wd, env=env, stdout=open(wd / "child.out", "w"),
                        stderr=open(wd / "child.err", "w"),
                        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
port = None
t0 = time.time()
while time.time() - t0 < 60:
    try:
        m = json.loads((run / "meta.json").read_text())
        if m.get("pid") == proc.pid and m.get("port"):
            port = m["port"]
            post(port, "/api/time", {"mode": "throttled", "target_tps": 30})
            break
    except Exception:
        pass
    time.sleep(0.1)
tok = post(port, "/api/console/rain", {"intensity_mm": 5.0})["confirm_token"]
print("2) GUI console:", post(port, "/api/console/rain", {"intensity_mm": 5.0, "confirm_token": tok}))
while not events(wd, rid):
    time.sleep(0.05)
time.sleep(0.3)
proc.send_signal(signal.SIGINT)
proc.wait(timeout=60)
print("   Ctrl-C; events.jsonl:", [(e["n"], e["tick"], e["event"]) for e in events(wd, rid)])
print("   snapshots:", sorted(p.name for p in (run / "snapshots").iterdir()))

# 3) offline natural event through the CLI
r = cli(wd, "event", "--run", rid, "--type", "drought", "--params", '{"strength": 0.9}')
print("3) event --type drought  rc=%d  stdout: %r" % (r.returncode, last_line(r.stdout)))
print("   events.jsonl now:", [(e["n"], e["tick"], e["event"]) for e in events(wd, rid)])
import zarr  # noqa: E402
latest = sorted((run / "snapshots").glob("tick_*.zarr"))[-1]
print("   latest snapshot %s causal_n=%d" % (latest.name,
      zarr.open_group(str(latest), mode="r").attrs["meta"]["causal_n"]))

# 4) continue the run: the drought never happens
r = cli(wd, "resume", "--run", rid, "--ticks", "200", check=True)
evs = events(wd, rid)
print("4) resume --ticks 200 -> final events.jsonl:", [(e["n"], e["tick"], e["event"]) for e in evs])
print("   drought in causal history:", any(e["event"] == "drought" for e in evs))

# physical evidence: a 0.9-strength drought over the whole world removes 90% of vapor
import numpy as np  # noqa: E402
v5 = zarr.open_group(str(run / "snapshots" / "tick_000000000005.zarr"), mode="r")["vapor"][:]
v6 = zarr.open_group(str(run / "snapshots" / "tick_000000000006.zarr"), mode="r")["vapor"][:]
print("   total vapor tick5=%.4f tick6=%.4f (a drought of 0.9 would leave ~%.4f)" % (
    v5.sum(), v6.sum(), 0.1 * v5.sum()))
