"""Defense check 2: natural runs only (seed placed by place_seed = CLI seed path), shipped
hydrolysis rate 1e-7 and shipped mutation 0.005. Arm A: 288 K pond (as tests' _life_sim).
Arm B: 330 K warm pool (spec: seed goes into a 'shallow, warm pool near a vent') so
hydrolysis actually kills polymers inside the window -> dead / recycled partner slots.

Tracks non-replicase FREE polymers whose p_partner is stale (target dead, or target no
longer points back) vs a control of non-replicase FREE polymers with p_partner == -1,
and whether each was later used as a template (state BOUND) or had its link reset."""
import sys
sys.path.insert(0, "/home/nick/Life/firmament")
import tempfile
from pathlib import Path
import numpy as np
import warp as wp
from tests.conftest import build_test_sim, tiny_cfg
from tests.test_phase5_polymers import REPLICASE, PAD
from firmament.operator.console import place_seed

BINDER = "M3M1M3M4M2M4"
SCR = Path("/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/defense-life-binder-partner-never-released")


def build(temp_k, monomers, cap):
    tmp = Path(tempfile.mkdtemp(dir=SCR))
    n = 16
    cfg = tiny_cfg(n=n, water=0.9, cap=cap, seed=4242)
    cfg.polymers.mutation_rate_per_monomer = 0.005
    state, sched = build_test_sim(cfg, "cpu", tmp / "run")
    sched.modules = [sched.modules[4]]          # polymers only (as tests' _life_sim)
    state.water_depth = wp.array(np.full((n, n), 1.0), dtype=wp.float64, device="cpu")
    state.temp = wp.array(np.full((3, n, n), temp_k), dtype=wp.float64, device="cpu")
    spec = np.zeros_like(state.species.numpy()); idx = sched.chem.index
    y = x = n // 2
    spec[idx["H2O"]] = 10_000_000
    for m in ("M1", "M2", "M3", "M4"):
        spec[idx[m], y, x] = monomers
    spec[idx["PP"], y, x] = 500000
    state.species = wp.array(spec, dtype=wp.int32, device="cpu")
    place_seed(sched, state, REPLICASE + BINDER + PAD, x, y)
    print(f"  hydrolysis_base_rate={cfg.polymers.hydrolysis_base_rate} mutation={cfg.polymers.mutation_rate_per_monomer} T={temp_k}K cap={cap}")
    return cfg, state, sched


def run(label, temp_k, monomers, cap, T):
    print(label)
    cfg, state, sched = build(temp_k, monomers, cap)
    repl = sched.poly.repl_bit
    stale, ctrl = {}, {}
    kinds = {"dead": 0, "recycled": 0, "oneway": 0}
    last_mutual = {}      # slot -> partner id while mutual
    triangle = 0
    for t in range(1, T + 1):
        try:
            sched.tick_once()
        except RuntimeError as e:
            print("  stopped:", e); T = t - 1; break
        st = state.p_state.numpy(); pa = state.p_partner.numpy(); ids = state.p_id.numpy()
        mot = state.p_motifs.numpy()
        live = np.nonzero((st >= 1) & (st <= 3))[0]
        for s in live:
            pid = int(ids[s]); p = int(pa[s])
            for book in (stale, ctrl):
                r = book.get(pid)
                if r is not None and r["end"] is None:
                    if st[s] == 2:
                        r["bound"] = True
                    if p < 0:
                        r["reset"] = True
            if st[s] != 1 or (mot[s] & repl) or p == s:
                continue
            if p < 0:
                if pid not in ctrl and pid not in stale:
                    ctrl[pid] = dict(first=t, bound=False, reset=False, end=None)
                continue
            if st[p] in (1, 2, 3) and pa[p] == s:
                last_mutual[int(s)] = int(ids[p]); continue
            if st[p] == 0:
                k = "dead"
            elif last_mutual.get(int(s)) is not None and last_mutual[int(s)] != int(ids[p]):
                k = "recycled"
            else:
                k = "oneway"
            q = int(pa[p])
            if q >= 0 and q != p and q != s and st[p] == 1 and pa[q] == p:
                triangle += 1
            r = stale.get(pid)
            if r is None:
                stale[pid] = dict(first=t, bound=False, reset=False, end=None, kinds={k})
                kinds[k] += 1
            else:
                if k not in r["kinds"]:
                    r["kinds"].add(k); kinds[k] += 1
        alive_ids = set(int(ids[s]) for s in live)
        for book in (stale, ctrl):
            for pid, r in book.items():
                if r["end"] is None and pid not in alive_ids:
                    r["end"] = t
    print(f"  ran {T} ticks; live={len(live)}; stale-pointer kinds seen (per polymer): {kinds}")
    for name, book in (("STALE  ", stale), ("CONTROL", ctrl)):
        obs = [r for r in book.values() if ((r['end'] or T) - r['first']) >= 50]
        print(f"  {name}: {len(obs)} non-replicase FREE followed >=50 ticks | later templated: "
              f"{sum(r['bound'] for r in obs)} | link ever reset to -1: {sum(r['reset'] for r in obs) if book is stale else 'n/a'}")
    print(f"  tick-samples where a stale polymer's target is FREE and mutually paired with a THIRD polymer: {triangle}")


pass
run("ARM B (312 K warm pool)", 312.0, 8000, 6000, 1200)
