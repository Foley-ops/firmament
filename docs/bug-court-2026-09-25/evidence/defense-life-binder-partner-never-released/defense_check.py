"""Defense's independent check (no state injection; seed placed via place_seed, the same
function the CLI `seed` command uses). Seed = replicase + binder + pad, shipped mutation.

Per tick we classify every FREE polymer holding p_partner >= 0:
  mutual  : partner is alive and points back (a real binder pair)
  deadptr : partner slot is DEAD (state 0)                      -> stale pointer
  reused  : partner slot is alive but holds a DIFFERENT id than when the link was made
  oneway  : partner alive, same id, but does not point back (e.g. it started copying)
For every polymer first seen stale, we follow it until its own death / end of run and
record: (1) was its p_partner EVER reset to -1 while alive, (2) was it EVER in state
BOUND (=templated) afterwards, (3) did it ever re-enter a mutual pair.
Also compares ecology.bound_pairs with the true number of mutual pairs."""
import sys
sys.path.insert(0, "/home/nick/Life/firmament")
import tempfile
from pathlib import Path
import numpy as np
from tests.test_phase5_polymers import _life_sim, REPLICASE, PAD
from firmament.instruments import ecology

BINDER = "M3M1M3M4M2M4"
SCR = Path("/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/defense-life-binder-partner-never-released")
tmp = Path(tempfile.mkdtemp(dir=SCR))
cfg, state, sched, (y, x) = _life_sim(tmp, "cpu", seq=REPLICASE + BINDER + PAD,
                                      mutation=0.005, pp=500000, monomers=20000)
repl = sched.poly.repl_bit
bbit = 1 << sched.poly.motif_names.index("binder")
T = 500
link_id = {}       # slot -> (partner slot, partner id) at the time the mutual link was seen
stale = {}         # pid -> dict(first=tick, kind=..., reset=False, bound=False, remutual=False, repl=bool, end=None)
kinds = {"deadptr": 0, "reused": 0, "oneway": 0}
bp_err = []
for t in range(1, T + 1):
    sched.tick_once()
    st = state.p_state.numpy(); pa = state.p_partner.numpy(); ids = state.p_id.numpy()
    mot = state.p_motifs.numpy()
    live = np.nonzero((st >= 1) & (st <= 3))[0]
    live_ids = set(int(ids[s]) for s in live)
    mutual_pairs = 0
    for s in live:
        pid = int(ids[s]); p = int(pa[s])
        rec = stale.get(pid)
        if rec is not None and rec["end"] is None:
            if p < 0:
                rec["reset"] = True
            if st[s] == 2:
                rec["bound"] = True
        if st[s] != 1 or p < 0 or p == s:
            continue
        if st[p] in (1, 2, 3) and pa[p] == s:
            mutual_pairs += 1
            if rec is not None and rec["end"] is None and link_id.get(s, (None, None))[1] != int(ids[p]):
                rec["remutual"] = True
            link_id[s] = (p, int(ids[p]))
            continue
        if st[p] == 0:
            k = "deadptr"
        elif s in link_id and link_id[s][0] == p and link_id[s][1] != int(ids[p]):
            k = "reused"
        else:
            k = "oneway"
        if pid not in stale:
            stale[pid] = dict(first=t, kind=k, reset=False, bound=False, remutual=False,
                              repl=bool(mot[s] & repl), end=None, slot=int(s), ptr=p,
                              ptr_id_then=int(ids[p]), ptr_state_then=int(st[p]))
            kinds[k] += 1
        else:
            stale[pid].setdefault("kinds", set()).add(k)
    for pid, rec in stale.items():
        if rec["end"] is None and pid not in live_ids:
            rec["end"] = t
    v = {"p_state": st, "p_partner": pa, "p_cell": state.p_cell.numpy(), "recent_copies": []}
    eco = ecology.sample(v, cfg)["bound_pairs"]
    true_pairs = mutual_pairs // 2
    if eco != true_pairs:
        bp_err.append((t, eco, true_pairs))

print(f"ticks={T} live={int(((st>=1)&(st<=3)).sum())}")
print("first-seen stale-pointer kinds:", kinds)
nr = [r for r in stale.values() if not r["repl"]]
rp = [r for r in stale.values() if r["repl"]]
for name, grp in (("NON-replicase", nr), ("replicase", rp)):
    obs = [r for r in grp if ((r["end"] or T) - r["first"]) >= 50]
    print(f"{name}: {len(grp)} stale polymers; followed >=50 ticks: {len(obs)}; "
          f"ever reset to -1: {sum(r['reset'] for r in obs)}; ever templated (BOUND): "
          f"{sum(r['bound'] for r in obs)}; ever re-paired: {sum(r['remutual'] for r in obs)}")
print("sample of stale non-replicase polymers:")
for pid, r in sorted(stale.items(), key=lambda kv: kv[1]["first"]):
    if r["repl"]:
        continue
    s = r["slot"]; p = r["ptr"]
    now = (f"now p_partner={int(pa[s])} -> slot {p} holds id {int(ids[p])} state {int(st[p])} "
           f"(its p_partner {int(pa[p])})") if r["end"] is None else f"died at tick {r['end']}"
    print(f"  id={pid:5d} stale@{r['first']:3d} kind={r['kind']:7s} ptr slot {p} "
          f"(id {r['ptr_id_then']}, state {r['ptr_state_then']} then) | {now} | reset={r['reset']} "
          f"templated={r['bound']}")
print(f"ecology.bound_pairs != true mutual pairs on {len(bp_err)}/{T} ticks; last: {bp_err[-3:]}")
