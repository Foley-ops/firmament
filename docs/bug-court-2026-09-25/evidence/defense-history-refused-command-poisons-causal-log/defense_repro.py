"""Defense's independent re-test of 'refused-command-poisons-causal-log'. CPU only."""
import sys
sys.path.insert(0, "/home/nick/Life/firmament")
import json, os, shutil, subprocess, tempfile, time, urllib.request  # noqa: E401,E402
from pathlib import Path  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = "/home/nick/Life/firmament"
PY = f"{REPO}/.venv/bin/python"
HERE = Path(__file__).resolve().parent
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=REPO)


def last(s):
    ls = [x for x in s.strip().splitlines() if x.strip()]
    return ls[-1][:160] if ls else ""


def cli(wd, *a):
    return subprocess.run([PY, "-m", "firmament.cli", "--device", "cpu", *a], cwd=wd, env=ENV,
                          capture_output=True, text=True, timeout=300)


def evs(p):
    return [(r["n"], r["tick"], r["event"], r["params"]) for r in
            (json.loads(x) for x in open(p) if x.strip())]


# ---------------- Part 1: in-process, the code's OWN invariant is broken ----------------
from tests.conftest import build_test_sim, tiny_cfg  # noqa: E402
from firmament.operator import causal, console  # noqa: E402

tmp = Path(tempfile.mkdtemp(dir=HERE, prefix="p1_"))
cfg = tiny_cfg(n=8, cap=100)
s, sched = build_test_sim(cfg, "cpu", tmp / "run")
for _ in range(3):
    sched.tick_once()
console.request(sched, "rain", {"intensity_mm": None})   # what app.js sends for "10mm"
try:
    sched.tick_once()
except Exception as e:
    print("P1 live apply raised:", type(e).__name__, e)
print("P1 events.jsonl:", evs(tmp / "run" / "events.jsonl"))
print("P1 sched.events.count =", sched.events.count, " state.causal_n =", s.causal_n)
try:
    causal.commit_now(sched, "solar", {"multiplier": 1.0})
except Exception as e:
    print("P1 next commit_now ->", type(e).__name__, e)

# ---------------- Part 2: full CLI + live GUI API path ----------------
wd = Path(tempfile.mkdtemp(dir=HERE, prefix="p2_"))
import yaml  # noqa: E402
c = tiny_cfg(n=16, cap=500)
(wd / "c.yaml").write_text(yaml.safe_dump(c.model_dump(mode="json")))
r = cli(wd, "run", "--config", "c.yaml", "--ticks", "10")
rid = sorted(p.name for p in (wd / "runs").iterdir())[-1]
run = wd / "runs" / rid
print("P2 run --ticks 10 rc=%d snaps=%s" % (r.returncode, sorted(x.name for x in (run / "snapshots").iterdir())))

proc = subprocess.Popen([PY, "-m", "firmament.cli", "--device", "cpu", "resume", "--run", rid,
                         "--ticks", "2000"], cwd=wd, env=ENV,
                        stdout=open(wd / "o.txt", "w"), stderr=open(wd / "e.txt", "w"))
port, t0 = None, time.time()
while port is None and time.time() - t0 < 60:
    try:
        m = json.loads((run / "meta.json").read_text())
        port = m.get("port") if m.get("pid") == proc.pid else None
    except Exception:
        pass
    time.sleep(0.1)


def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/console/rain",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


# exactly app.js: body[k] = +input.value, "10mm" -> NaN -> JSON null
body = {"cx": 8, "cy": 8, "radius": 60, "intensity_mm": None}
tok = post(dict(body))["confirm_token"]
print("P2 GUI confirm ->", post(dict(body, confirm_token=tok)))
rc = proc.wait(timeout=120)
print("P2 live sim rc=%d -> %s" % (rc, last((wd / "e.txt").read_text())))
print("P2 events.jsonl:", evs(run / "events.jsonl"))
print("P2 snapshots:", sorted(x.name for x in (run / "snapshots").iterdir()))

for i in (1, 2):
    r = cli(wd, "resume", "--run", rid, "--ticks", "2000")
    print("P2 resume #%d rc=%d -> %s" % (i, r.returncode, last(r.stderr)))
cj = json.loads((run / "crash.json").read_text())
print("P2 crash.json tick:", cj["tick"], cj["exception"][:90])

# control: identical run dir with ONLY the never-applied record removed resumes fine
wd2 = Path(tempfile.mkdtemp(dir=HERE, prefix="p2ctl_"))
shutil.copytree(wd / "runs", wd2 / "runs")
(wd2 / "runs" / rid / ".lease").unlink(missing_ok=True)
(wd2 / "runs" / rid / "events.jsonl").write_text("")
r = cli(wd2, "resume", "--run", rid, "--ticks", "60")
print("P2 CONTROL (poison record stripped) resume --ticks 60 rc=%d" % r.returncode,
      "snaps:", sorted(x.name for x in (wd2 / "runs" / rid / "snapshots").iterdir()))

# ---------------- Part 3: the documented refusal via CLI ----------------
wd3 = Path(tempfile.mkdtemp(dir=HERE, prefix="p3_"))
(wd3 / "c.yaml").write_text(yaml.safe_dump(c.model_dump(mode="json")))
cli(wd3, "run", "--config", "c.yaml", "--ticks", "10")
rid3 = sorted(p.name for p in (wd3 / "runs").iterdir())[-1]
r = cli(wd3, "event", "--run", rid3, "--type", "add_land", "--params", '{"cols": 4}')
print("P3 event add_land rc=%d -> %s" % (r.returncode, last(r.stderr)))
print("P3 events.jsonl:", evs(wd3 / "runs" / rid3 / "events.jsonl"))
r = cli(wd3, "resume", "--run", rid3, "--ticks", "20")
print("P3 resume rc=%d -> %s" % (r.returncode, last(r.stderr)))
r = cli(wd3, "fork", "--run", rid3)
print("P3 fork rc=%d -> new run %s" % (r.returncode, r.stdout.strip()))
