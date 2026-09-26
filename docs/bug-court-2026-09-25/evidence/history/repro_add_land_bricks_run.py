"""EXHIBIT A: a refused causal command is logged BEFORE it is applied; when the apply
raises, the refusal stays in events.jsonl forever and every later resume replays it and
crashes again -> the run can never be continued.

RUNBOOK ('Moving to world_default.yaml'): "`add_land` is a documented refusal".
causal.py docstring: resume re-applies exactly the logged records."""
import sys
sys.path.insert(0, "/home/nick/Life/firmament")
sys.path.insert(0, "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/history")
from common import cli, events, last_line, only_run, workdir, write_cfg  # noqa: E402

wd = workdir("A_addland")
cfg = write_cfg(wd)
print("workdir:", wd)

r = cli(wd, "run", "--config", str(cfg), "--ticks", "10", check=True)
rid = only_run(wd)
print("1) run --ticks 10          rc=%d  snapshots=%s" % (
    r.returncode, sorted(p.name for p in (wd / "runs" / rid / "snapshots").iterdir())))
print("   events.jsonl before:", events(wd, rid))

r = cli(wd, "event", "--run", rid, "--type", "add_land", "--params", '{"cols": 4}')
print("2) event --type add_land   rc=%d  -> %s" % (r.returncode, last_line(r.stderr)))
print("   events.jsonl after refusal:",
      [(e["n"], e["tick"], e["event"], e["params"]) for e in events(wd, rid)])

for attempt in (1, 2):
    r = cli(wd, "resume", "--run", rid, "--ticks", "5")
    print("3.%d) resume --ticks 5     rc=%d  -> %s" % (attempt, r.returncode, last_line(r.stderr)))

crash = (wd / "runs" / rid / "crash.json")
if crash.exists():
    import json
    c = json.loads(crash.read_text())
    print("   crash.json:", {"exception": c["exception"], "tick": c["tick"]})

# the same poisoning with an ordinary natural event whose parameter is malformed
wd2 = workdir("A_badrain")
cfg2 = write_cfg(wd2)
cli(wd2, "run", "--config", str(cfg2), "--ticks", "10", check=True)
rid2 = only_run(wd2)
r = cli(wd2, "event", "--run", rid2, "--type", "rain", "--params", '{"intensity_mm": "10mm"}')
print("4) event rain intensity_mm='10mm'  rc=%d -> %s" % (r.returncode, last_line(r.stderr)))
print("   events.jsonl:", [(e["n"], e["event"], e["params"]) for e in events(wd2, rid2)])
r = cli(wd2, "resume", "--run", rid2, "--ticks", "5")
print("5) resume --ticks 5        rc=%d  -> %s" % (r.returncode, last_line(r.stderr)))

# 6) LIVE path: the GUI god console sends `+input.value` for every field (gui/app.js),
#    so a typo like "10mm" arrives as JSON null. The running sim commits (logs) it, then
#    crashes applying it; every resume replays the poisoned record and crashes again.
import os, signal, subprocess, time, urllib.request, json as _j  # noqa: E401,E402
from common import PY, REPO  # noqa: E402


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=_j.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return _j.loads(urllib.request.urlopen(req, timeout=10).read())


wd3 = workdir("A_gui")
cfg3 = write_cfg(wd3)
cli(wd3, "run", "--config", str(cfg3), "--ticks", "10", check=True)
rid3 = only_run(wd3)
run3 = wd3 / "runs" / rid3
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=REPO)
proc = subprocess.Popen([PY, "-m", "firmament.cli", "--device", "cpu", "resume", "--run", rid3,
                         "--auto-restart"], cwd=wd3, env=env,
                        stdout=open(wd3 / "child.out", "w"), stderr=open(wd3 / "child.err", "w"))
port = None
t0 = time.time()
while port is None and time.time() - t0 < 60:
    try:
        m = _j.loads((run3 / "meta.json").read_text())
        if m.get("pid") == proc.pid:
            port = m.get("port")
    except Exception:
        pass
    time.sleep(0.1)
body = {"cx": 8, "cy": 8, "radius": 6, "intensity_mm": None}     # GUI: +"10mm" -> NaN -> null
tok = post(port, "/api/console/rain", dict(body))["confirm_token"]
print("6) GUI console rain (typo) ->", post(port, "/api/console/rain", dict(body, confirm_token=tok)))
rc = proc.wait(timeout=60)
print("   live sim exited rc=%d -> %s" % (rc, last_line((wd3 / "child.err").read_text())))
print("   events.jsonl:", [(e["n"], e["tick"], e["event"], e["params"]) for e in events(wd3, rid3)])
r = cli(wd3, "resume", "--run", rid3, "--ticks", "5", "--auto-restart")
print("7) resume --auto-restart   rc=%d  -> %s" % (r.returncode, last_line(r.stderr)))
r = cli(wd3, "event", "--run", rid3, "--type", "rain", "--params", '{"intensity_mm": 5}')
print("8) event rain (valid)      rc=%d  stdout: %r" % (r.returncode, last_line(r.stdout)))
print("   events.jsonl:", [(e["n"], e["tick"], e["event"], e["params"]) for e in events(wd3, rid3)])
r = cli(wd3, "resume", "--run", rid3, "--ticks", "5")
print("9) resume --ticks 5        rc=%d  -> %s" % (r.returncode, last_line(r.stderr)))
