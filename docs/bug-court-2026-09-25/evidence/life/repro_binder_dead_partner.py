"""Controlled repro: binder pair X<->Y (the mutual p_partner state the binder effect
itself writes). Y dies by ordinary hydrolysis. k_cell_pass's hydrolysis branch only
releases a partner that is P_BOUND, so X (P_FREE) keeps p_partner = Y's old slot forever
- first a dead slot, later a recycled slot holding an unrelated polymer. Because
k_cell_pass accepts templates only with p_partner < 0, X is never copied again.
Counterfactual arm: identical run, but at Y's death X.p_partner is set to -1 (what a
release would do) -> X is copied repeatedly. X/Y are assembled from the cell's own
monomers (M -> unit + H2O); element totals are checked exact."""
import sys
sys.path.insert(0, "/home/nick/Life/firmament")
import tempfile
from pathlib import Path
import numpy as np
import warp as wp
from tests.test_phase5_polymers import _life_sim, REPLICASE, PAD
from firmament.core.audit import Audit

SCR = "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/life"
SYM = {"M1": 1, "M2": 2, "M3": 3, "M4": 4}
def parse(s): return np.array([SYM[s[i:i+2]] for i in range(0, len(s), 2)], np.uint8)
BINDER = "M3M1M3M4M2M4"
X = BINDER + "M1M3"                    # 8-mer, binder only (short -> long-lived)
Y = BINDER + "M1M3M2M4" * 60           # 246-mer, binder only (long -> hydrolyses first)

def run(release: bool):
    tmp = Path(tempfile.mkdtemp(dir=SCR))
    cfg, state, sched, (y, x) = _life_sim(tmp, "cpu", seq=REPLICASE + PAD, pp=400000,
                                          monomers=1500, mutation=0.0)
    cfg.polymers.hydrolysis_base_rate = 3e-6
    poly = sched.poly
    arr = {k: getattr(state, k).numpy().copy() for k in
           ("p_state", "p_len", "p_cell", "p_id", "p_partner", "p_child", "p_seq", "p_motifs", "p_parent")}
    spec = state.species.numpy(); idx = sched.chem.index
    for slot, (pid, s) in zip((1, 2), ((900, X), (901, Y))):
        q = parse(s)
        arr["p_state"][slot], arr["p_len"][slot], arr["p_cell"][slot] = 1, len(q), int(arr["p_cell"][0])
        arr["p_id"][slot], arr["p_partner"][slot], arr["p_child"][slot], arr["p_parent"][slot] = pid, -1, -1, 0
        arr["p_seq"][slot, :] = 0; arr["p_seq"][slot, :len(q)] = q
        arr["p_motifs"][slot] = poly.motif_mask_host(q)
        for m in range(1, 5):
            spec[idx[f"M{m}"], y, x] -= int((q == m).sum())
        spec[idx["H2O"], y, x] += len(q)
    arr["p_partner"][1], arr["p_partner"][2] = 2, 1       # X<->Y aggregate
    state.species = wp.array(spec, dtype=wp.int32, device="cpu")
    for k, a in arr.items():
        setattr(state, k, wp.array(a, dtype=getattr(state, k).dtype, device="cpu"))
    aud = Audit(cfg, sched.chem, tmp); el0 = aud.element_totals(state)
    y_dead, templ, prev, log = None, 0, 1, []
    for t in range(1, 1501):
        sched.tick_once()
        st = state.p_state.numpy(); pa = state.p_partner.numpy(); ids = state.p_id.numpy()
        if y_dead is None and not ((ids == 901) & (st > 0)).any():
            y_dead = t
            log.append(f"  tick {t}: Y hydrolysed (slot 2 state={st[2]}); X state={st[1]} X.p_partner=slot {pa[1]}")
            if release:
                pa = pa.copy(); pa[1] = -1
                state.p_partner = wp.array(pa, dtype=wp.int32, device="cpu")
        assert ids[1] == 900 and st[1] > 0, "X died - pick another seed"
        if y_dead is not None:
            if st[1] == 2 and prev != 2:
                templ += 1
            prev = int(st[1])
            if (t - y_dead) % 300 == 0 or t == 1500:
                p = int(pa[1])
                log.append(f"  tick {t:4d}: X state={st[1]} X.p_partner=slot {p}"
                           + (f" (slot {p} now holds id {ids[p]}, state {st[p]}, its p_partner {pa[p]})" if p >= 0 else "")
                           + f" | X bound as template {templ}x since Y died | live={int(((st>=1)&(st<=3)).sum())}")
    log.append(f"  element totals exact: {bool(np.array_equal(aud.element_totals(state), el0))}")
    return templ, log

for release in (False, True):
    templ, log = run(release)
    print(("COUNTERFACTUAL (X.p_partner released at Y's death)" if release else
           "AS SHIPPED (no release)") + f": X templated {templ} times after Y died")
    print("\n".join(log))
