"""Repro: a binder partner link is never cleared when the other side is re-purposed
(it starts copying / becomes a template) or dies. The abandoned polymer stays FREE with
p_partner >= 0 forever, and k_cell_pass's template test (`p_partner[p2] < 0`) and binder
test (`p_partner[p] < 0`) then permanently exclude it: it can never be copied again.
Natural run: shipped mutation rate, seed = replicase + binder + neutral pad,
polymers module only (tests' _life_sim helper). No state injection."""
import sys
sys.path.insert(0, "/home/nick/Life/firmament")
import tempfile
from pathlib import Path
import numpy as np
from tests.test_phase5_polymers import _life_sim, REPLICASE, PAD

BINDER = "M3M1M3M4M2M4"          # binder motif from configs/genetic_code_v0.yaml
SCR = "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/life"
tmp = Path(tempfile.mkdtemp(dir=SCR))
cfg, state, sched, (y, x) = _life_sim(tmp, "cpu", seq=REPLICASE + BINDER + PAD,
                                      mutation=0.005, pp=500000, monomers=20000)
print("mutation_rate_per_monomer =", cfg.polymers.mutation_rate_per_monomer, "(shipped default)")
repl = sched.poly.repl_bit
T = 500
orph = {}    # p_id -> [tick first orphaned, ever BOUND afterwards]
ctrl = {}    # p_id -> [tick first seen FREE non-replicase with p_partner == -1, ever BOUND afterwards]
dangling_dead = 0
for t in range(1, T + 1):
    sched.tick_once()
    st = state.p_state.numpy(); pa = state.p_partner.numpy(); ids = state.p_id.numpy()
    mot = state.p_motifs.numpy()
    live = np.nonzero(st > 0)[0]
    for s in live:
        pid = int(ids[s])
        for book in (orph, ctrl):
            if pid in book and t > book[pid][0] and st[s] == 2:
                book[pid][1] = True
        if st[s] != 1 or (mot[s] & repl):
            continue
        p = int(pa[s])
        if p >= 0 and p != s and (st[p] == 0 or pa[p] != s):
            if pid not in orph and pid not in ctrl:
                orph[pid] = [t, False]
        elif p < 0 and pid not in ctrl and pid not in orph:
            ctrl[pid] = [t, False]

st = state.p_state.numpy(); pa = state.p_partner.numpy(); ids = state.p_id.numpy()
alive_ids = {int(ids[s]): s for s in np.nonzero(st > 0)[0]}
def summary(book, name, min_age=100):
    old = {k: v for k, v in book.items() if T - v[0] >= min_age}
    used = sum(1 for v in old.values() if v[1])
    print(f"{name}: {len(old)} non-replicase FREE polymers observed >= {min_age} ticks; "
          f"{used} of them were later bound as a copy template ({(used/max(len(old),1)):.0%})")
    return old
print(f"tick {state.tick}: live polymers {int(((st>=1)&(st<=3)).sum())}")
o = summary(orph, "ORPHANED (p_partner points to a polymer that does not point back)")
c = summary(ctrl, "CONTROL  (p_partner == -1)")
print("orphans, first 8:")
for pid, (t0, used) in sorted(o.items(), key=lambda kv: kv[1][0])[:8]:
    s = alive_ids.get(pid)
    if s is None:
        print(f"  id={pid} orphaned@{t0}: died since"); continue
    p = int(pa[s])
    print(f"  id={pid:5d} orphaned@tick {t0:3d}  now state={int(st[s])} p_partner=slot {p} "
          f"(that slot: state={int(st[p])}, its p_partner={int(pa[p])}, points back: {int(pa[p]) == s})"
          f"  templated since: {used}")
