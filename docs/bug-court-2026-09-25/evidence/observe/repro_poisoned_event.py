"""A natural event whose apply step fails (add_land = documented refusal; or a GUI
param typo, which app.js serialises as JSON null) is ACCEPTED by the API/CLI, written to
the causal log events.jsonl, and only then raises inside the sim thread:
  * the live sim crashes;
  * the log now holds a record the state never applied (events.count = causal_n + 1),
    so every later causal command is refused;
  * `firmament resume` replays the poisoned record and crashes again at the same tick,
    forever (the run cannot be continued without hand-editing the causal log).

Part A: GUI/API path (real FastAPI handlers from make_app, called directly because
        httpx/TestClient is not installed).
Part B: CLI path, exactly as in docs/RUNBOOK.md: run -> event -> resume, in a temp cwd.
"""
import sys
sys.path.insert(0, "/home/nick/Life/firmament")
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

REPO = Path("/home/nick/Life/firmament")
DEV = "cpu"


def part_a():
    from tests.conftest import build_test_sim, tiny_cfg
    import firmament.io.rundir as rd
    from firmament.server.api import make_app
    from firmament.operator import console

    tmp = Path(tempfile.mkdtemp(prefix="poisonA_"))
    old = rd.RUNS
    rd.RUNS = tmp / "runs"
    try:
        cfg = tiny_cfg(n=16, cap=200)
        run_dir = rd.create(cfg)
        s, sched = build_test_sim(cfg, DEV, run_dir)
        for _ in range(3):
            sched.tick_once()
        app = make_app(sched, run_dir)
        ep = {r.path: r.endpoint for r in app.routes if hasattr(r, "endpoint")}
        console_cmd = ep["/api/console/{event}"]
        # body exactly as gui/app.js builds it when the user types "10mm" in intensity_mm:
        # body[k] = +"10mm" -> NaN -> JSON.stringify -> null
        body = {"cx": 8, "cy": 8, "radius": 4, "intensity_mm": None}
        r1 = console_cmd("rain", dict(body))
        r2 = console_cmd("rain", dict(body, confirm_token=r1["confirm_token"]))
        print("[A] API reply 1:", {k: r1[k] for k in ("event", "message")})
        print("[A] API reply 2:", r2)
        try:
            sched.tick_once()
            print("[A] tick ok (unexpected)")
        except Exception as e:
            print(f"[A] sim thread crashed at tick {s.tick}: {type(e).__name__}: {e}")
        recs = [json.loads(x) for x in open(run_dir / "events.jsonl") if x.strip()]
        print(f"[A] events.jsonl records: {[(r['event'], r['n'], r['params']) for r in recs]}")
        print(f"[A] state.causal_n = {s.causal_n}   events.count = {sched.events.count}")
        # a perfectly valid follow-up event is now refused forever
        console.request(sched, "rain", {"cx": 8, "cy": 8, "radius": 4, "intensity_mm": 5.0})
        try:
            sched.tick_once()
            print("[A] valid follow-up event applied")
        except Exception as e:
            print(f"[A] valid follow-up event -> {type(e).__name__}: {e}")
    finally:
        rd.RUNS = old
        shutil.rmtree(tmp, ignore_errors=True)


def part_b():
    tmp = Path(tempfile.mkdtemp(prefix="poisonB_"))
    try:
        cfgp = tmp / "tiny.yaml"
        cfgp.write_text(textwrap.dedent(f"""
            run: {{name: poison, seed: 1, dt_seconds: 60, snapshot_every_sim_days: 100000,
                   metrics_every_ticks: 10, audit_every_ticks: 1000}}
            world: {{size: [16, 16], vents: [[4, 8]]}}
            chemistry: {{file: {REPO}/configs/chemistry_v0_2.yaml}}
            polymers: {{capacity: 200, genetic_code: {REPO}/configs/genetic_code_v0.yaml}}
            logging: {{level: WARNING}}
        """))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(REPO))

        def cli(*args):
            p = subprocess.run([sys.executable, "-m", "firmament.cli", "--device", DEV, *args],
                               cwd=tmp, env=env, capture_output=True, text=True, timeout=300)
            tail = [l for l in (p.stdout + p.stderr).strip().splitlines()
                    if l and "CFL" not in l][-1:]
            return p.returncode, tail

        rc, out = cli("run", "--config", str(cfgp), "--ticks", "5", "--port", "18731")
        run_id = next((tmp / "runs").iterdir()).name
        print(f"[B] firmament run --ticks 5          -> rc={rc}")
        rc, out = cli("event", "--run", run_id, "--type", "add_land", "--params", "{}")
        print(f"[B] firmament event --type add_land  -> rc={rc} {out}")
        recs = [json.loads(x) for x in open(tmp / "runs" / run_id / "events.jsonl") if x.strip()]
        print(f"[B] events.jsonl now: {[(r['event'], r['n'], r['tick']) for r in recs]}")
        for attempt in (1, 2):
            rc, out = cli("resume", "--run", run_id, "--ticks", "5", "--port", "18731")
            print(f"[B] firmament resume (attempt {attempt}) -> rc={rc} {out}")
        crash = json.loads((tmp / "runs" / run_id / "crash.json").read_text())
        print(f"[B] crash.json: tick={crash['tick']} exception={crash['exception']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


part_a()
part_b()
