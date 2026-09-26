"""Repro: place_seed ("Validated first, then committed") does not validate length.
(a) a zero-length seed is accepted and becomes a zero-atom, immortal, FREE 'polymer' -
    exactly the v0.1 artifact that DECISIONS 2026-09-25 (MIN_LEN = 2) says v0.2 removed;
(b) a length-1 seed becomes a sub-MIN_LEN 'polymer';
(c) a seed longer than polymers.max_length is COMMITTED to events.jsonl and only then
    crashes in apply_seed, leaving a causal record the state never applied.
(d) a second seed after the first lineage died out is accepted and gets p_id 1 again
    ("the seed must be the first polymer"; ids are the lineage keys)."""
import sys
sys.path.insert(0, "/home/nick/Life/firmament")
import tempfile, json
from pathlib import Path
import numpy as np
import warp as wp
from tests.test_phase5_polymers import _life_sim, SEED_SEQ, _completed
from firmament.operator.console import place_seed

SCR = "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/life"

for label, seq in (("(a) zero-length seed ''", ""), ("(b) length-1 seed 'M1'", "M1")):
    cfg, state, sched, (y, x) = _life_sim(Path(tempfile.mkdtemp(dir=SCR)), "cpu", seq=seq)
    for _ in range(2000):
        sched.tick_once()
    st, ln = state.p_state.numpy(), state.p_len.numpy()
    print(f"{label}: accepted; after 2000 ticks slot0 state={st[0]} (1=FREE) p_len={ln[0]} "
          f"id={state.p_id.numpy()[0]} ; counted as completed polymer: {0 in _completed(state)}")

print("(c) seed longer than max_length:")
cfg, state, sched, (y, x) = _life_sim(Path(tempfile.mkdtemp(dir=SCR)), "cpu")
# fresh sim with no seed yet: reuse helper then clear slot 0 is messy -> build a new one
from tests.conftest import tiny_cfg, build_test_sim
cfg = tiny_cfg(n=16, water=0.9, cap=3000, seed=4242)
rd = Path(tempfile.mkdtemp(dir=SCR)) / "run"
state, sched = build_test_sim(cfg, "cpu", rd)
spec = state.species.numpy(); idx = sched.chem.index
for m in ("M1", "M2", "M3", "M4"):
    spec[idx[m], 8, 8] = 2000
state.species = wp.array(spec, dtype=wp.int32, device="cpu")
LONG = "M1M3M1M2M4M2" + "M1M3M2M4" * 74          # 302 monomers; max_length = 256
try:
    place_seed(sched, state, LONG, 8, 8)
    print("   accepted?!")
except Exception as e:
    print(f"   place_seed raised {type(e).__name__}: {e}")
recs = [json.loads(l) for l in open(rd / "events.jsonl") if l.strip()]
print(f"   events.jsonl records: {[(r['event'], r['n'], len(r['params']['sequence'])//2) for r in recs]}"
      f" ; state.causal_n = {state.causal_n}")

print("(d) second seed after extinction:")
cfg, state, sched, (y, x) = _life_sim(Path(tempfile.mkdtemp(dir=SCR)), "cpu", pp=100000)
t = 0
while True:                       # let the first lineage copy itself a few times
    sched.tick_once(); t += 1
    sched.lineage.flush()
    if sched.lineage.db.execute("SELECT COUNT(*) FROM copies").fetchone()[0] >= 3:
        break
# then it dies out (a hot/hostile spell: faster hydrolysis, same law for every polymer)
cfg.polymers.hydrolysis_base_rate = 1e-3
while int((state.p_state.numpy() > 0).sum()) > 0:
    sched.tick_once(); t += 1
    assert t < 20000
cfg.polymers.hydrolysis_base_rate = 1e-7
rows = sched.lineage.db.execute("SELECT child_id, parent_id FROM copies ORDER BY child_id").fetchall()
print(f"   tick {state.tick}: every polymer dead; lineage has {len(rows)} copies, "
      f"children of id 1: {[c for c, p in rows if p == 1]} ; next_poly_id = {state.next_poly_id}")
spec = state.species.numpy()
for m in ("M1", "M2", "M3", "M4"):
    spec[sched.chem.index[m], y, x] += 100
state.species = wp.array(spec, dtype=wp.int32, device="cpu")
place_seed(sched, state, "M1M3M1M2M4M2" + "M4M2M3M1" * 6, x, y)   # a DIFFERENT sequence
recs = [json.loads(l) for l in open(sched.events.path) if l.strip()]
print(f"   second place_seed accepted; seed_placed records: {sum(r['event']=='seed_placed' for r in recs)} ;"
      f" new seed p_id = {state.p_id.numpy()[0]} (same id as the first seed); next_poly_id = {state.next_poly_id}")
