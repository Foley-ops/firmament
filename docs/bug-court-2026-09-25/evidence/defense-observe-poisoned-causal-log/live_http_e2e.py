"""Defense probe: exercise the REAL live path (a `firmament run` process, uvicorn,
real HTTP POSTs from urllib exactly as gui/app.js would send them), then resume,
then check whether `firmament fork` is an escape hatch.

CPU only; everything happens in a fresh temp cwd (the CLI writes ./runs)."""
import sys
sys.path.insert(0, "/home/nick/Life/firmament")
import json
import os
import shutil
import socket
import subprocess
import tempfile
import textwrap
import time
import urllib.request
from pathlib import Path

REPO = Path("/home/nick/Life/firmament")
DEV = "cpu"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def scenario(tag, event, raw_body):
    tmp = Path(tempfile.mkdtemp(prefix=f"defE2E_{tag}_"))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(REPO))
    try:
        cfgp = tmp / "tiny.yaml"
        cfgp.write_text(textwrap.dedent(f"""
            run: {{name: defe2e, seed: 1, dt_seconds: 60, snapshot_every_sim_days: 0.0139,
                   metrics_every_ticks: 10, audit_every_ticks: 1000}}
            world: {{size: [16, 16], vents: [[4, 8]]}}
            chemistry: {{file: {REPO}/configs/chemistry_v0_2.yaml}}
            polymers: {{capacity: 200, genetic_code: {REPO}/configs/genetic_code_v0.yaml}}
            logging: {{level: WARNING}}
        """))

        def cli(*args, timeout=300):
            p = subprocess.run([sys.executable, "-m", "firmament.cli", "--device", DEV, *args],
                               cwd=tmp, env=env, capture_output=True, text=True, timeout=timeout)
            tail = [l for l in (p.stdout + p.stderr).strip().splitlines()
                    if l and "CFL" not in l and "Warp CUDA error" not in l and "took" not in l][-1:]
            return p.returncode, tail

        port = free_port()
        proc = subprocess.Popen([sys.executable, "-m", "firmament.cli", "--device", DEV, "run",
                                 "--config", str(cfgp), "--ticks", "1000000", "--port", str(port)],
                                cwd=tmp, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        # wait for the API port to be published and at least one snapshot to exist
        run_dir = None
        deadline = time.time() + 120
        while time.time() < deadline:
            runs = list((tmp / "runs").glob("*")) if (tmp / "runs").exists() else []
            if runs:
                run_dir = runs[0]
                m = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").exists() else {}
                if m.get("port") and list((run_dir / "snapshots").glob("tick_*")):
                    port = m["port"]
                    break
            time.sleep(0.5)
        time.sleep(1.0)
        print(f"[{tag}] live run up: pid={proc.pid} port={port} alive={proc.poll() is None} "
              f"snapshots={[p.name for p in sorted((run_dir / 'snapshots').glob('tick_*'))][-2:]}")
        st1, r1 = post(port, f"/api/console/{event}", dict(raw_body))
        print(f"[{tag}] POST #1 -> HTTP {st1} {r1.get('message', r1)}")
        st2, r2 = post(port, f"/api/console/{event}", dict(raw_body, confirm_token=r1["confirm_token"]))
        print(f"[{tag}] POST #2 (confirm) -> HTTP {st2} {r2}")
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            print(f"[{tag}] live process still alive after 60 s (did NOT crash)")
            proc.kill()
            proc.wait()
        out = proc.stdout.read().strip().splitlines()
        out = [l for l in out if "CFL" not in l]
        print(f"[{tag}] live process exit code = {proc.returncode}; last line: {out[-1:] }")
        recs = [json.loads(x) for x in open(run_dir / "events.jsonl") if x.strip()]
        print(f"[{tag}] events.jsonl: {[(r['event'], r['n'], r['tick'], r['params']) for r in recs]}")
        crash = json.loads((run_dir / "crash.json").read_text())
        print(f"[{tag}] crash.json: tick={crash['tick']} exc={crash['exception'][:110]} "
              f"last_good={Path(crash['last_good_snapshot']).name if crash['last_good_snapshot'] else None}")
        for attempt in (1, 2):
            rc, tail = cli("resume", "--run", run_dir.name, "--ticks", "50", "--port", str(free_port()))
            print(f"[{tag}] firmament resume attempt {attempt} -> rc={rc} {tail}")
        crash2 = json.loads((run_dir / "crash.json").read_text())
        print(f"[{tag}] crash.json after resume: tick={crash2['tick']}")
        # defense: is fork an escape hatch?
        rc, tail = cli("fork", "--run", run_dir.name)
        fork_id = tail[0] if tail else None
        print(f"[{tag}] firmament fork -> rc={rc} new run {fork_id}")
        frecs = [json.loads(x) for x in open(tmp / "runs" / fork_id / "events.jsonl") if x.strip()]
        print(f"[{tag}] fork events.jsonl: {[(r['event'], r['n']) for r in frecs]}")
        rc, tail = cli("resume", "--run", fork_id, "--ticks", "50", "--port", str(free_port()))
        print(f"[{tag}] firmament resume <fork> --ticks 50 -> rc={rc} {tail}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# exactly what gui/app.js sends when the user types "10mm" (JSON.stringify(+"10mm") -> null)
scenario("gui_typo_rain", "rain", {"cx": 8, "cy": 8, "radius": 4, "intensity_mm": None})
# add_land via the live API (not in the GUI dropdown; reachable with any HTTP client)
scenario("api_add_land", "add_land", {"cols": 4})
