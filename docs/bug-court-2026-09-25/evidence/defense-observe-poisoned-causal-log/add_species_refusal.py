"""Does the SPEC-MANDATED refusal (Build Guide 275: add-species 'refuses' a change to an
existing rate) also enter the causal log and brick the run? Offline, in-process, CPU."""
import sys
sys.path.insert(0, "/home/nick/Life/firmament")
import json
import shutil
import tempfile
from pathlib import Path

import yaml

from tests.conftest import build_test_sim, tiny_cfg
import firmament.io.rundir as rd
from firmament.operator import console, causal

tmp = Path(tempfile.mkdtemp(prefix="defAddSp_"))
old = rd.RUNS
rd.RUNS = tmp / "runs"
try:
    cfg = tiny_cfg(n=16, cap=200)
    run_dir = rd.create(cfg)
    s, sched = build_test_sim(cfg, "cpu", run_dir)
    for _ in range(2):
        sched.tick_once()
    doc = yaml.safe_load(open("/home/nick/Life/firmament/configs/chemistry_v0_2.yaml"))
    doc["reactions"][0] = dict(doc["reactions"][0], k300=12345.0)   # changes an existing rate
    bad = tmp / "chem_bad.yaml"
    bad.write_text(yaml.safe_dump(doc))
    console.request(sched, "add_species", {"file": str(bad)})
    print("request accepted (no validation at request time)")
    try:
        sched.tick_once()
    except Exception as e:
        print(f"tick {s.tick}: {type(e).__name__}: {e}")
    recs = [json.loads(x) for x in open(run_dir / "events.jsonl") if x.strip()]
    print("events.jsonl:", [(r["event"], r["n"], r["tick"]) for r in recs])
    print(f"state.causal_n={s.causal_n} events.count={sched.events.count}")
    # a fresh process resuming from state with causal_n=0 would replay this record:
    s2, sched2 = build_test_sim(cfg, "cpu", run_dir)
    s2.tick = recs[0]["tick"]
    n = causal.load_replay(sched2, run_dir / "events.jsonl")
    print("replay queue len:", n)
    try:
        sched2.tick_once()
        print("replay ok")
    except Exception as e:
        print(f"replay tick {s2.tick}: {type(e).__name__}: {e}")
finally:
    rd.RUNS = old
    shutil.rmtree(tmp, ignore_errors=True)
