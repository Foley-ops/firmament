"""Polymers: the life-capable layer. Sequences, templated copying, motif effects.

Determinism design (docs/DECISIONS.md): all resource competition is resolved in a
per-cell SERIAL kernel iterating polymers in id order (index rebuilt host-side each
tick from a device snapshot — same bytes -> same order). Child slots and ids are
pre-reserved via an exclusive scan, so allocation never depends on GPU scheduling.
No designed behavior lives here: polymers do only what their motifs make physical.

Energetics (element-exact):
  copy step:        M + P~P -> chain-unit + 2 Pi           (unit = M - H2O)
  chain hydrolysis: chain-unit + H2O -> M
  effect activity:  P~P + H2O -> 2 Pi  (+heat)             (paying the ATP cost)
"""
from __future__ import annotations

import numpy as np
import warp as wp
import yaml

from firmament.core.rng import SALT_POLY_COPY, SALT_POLY_MOVE, mix, seed32
from firmament.core.state import P_BOUND, P_COPYING, P_DEAD, P_FREE
from firmament.io.logging import get

log = get("polymers")

P_BUILD = 5          # child under construction
EV_CAP = 1 << 16
EFF = {"replicase": 0, "catalyst": 1, "binder": 2, "membrane": 3, "photoactive": 4,
       "motor": 5, "emit": 6, "sense": 7}
KH_EA = 80000.0      # J/mol — hydrolysis is strongly temperature-dependent (hot water kills)
MEM_THRESH = 200     # L units to close a compartment
R_GAS = 8.314


@wp.func
def complement(m: int) -> int:
    if m == 1:
        return 2
    if m == 2:
        return 1
    if m == 3:
        return 4
    return 3


@wp.func
def scan_motifs(seq: wp.array2d(dtype=wp.uint8), p: int, ln: int,
                motif: wp.array2d(dtype=wp.uint8), nm: int, mk: int) -> int:
    mask = int(0)
    for m in range(nm):
        found = int(0)
        for pos in range(ln - mk + 1):
            ok = int(1)
            for x in range(mk):
                if seq[p, pos + x] != motif[m, x]:
                    ok = 0
                    break
            if ok == 1:
                found = 1
                break
        if found == 1:
            mask |= (1 << m)
    return mask


@wp.kernel
def k_cell_pass(
    # polymer arrays
    p_id: wp.array(dtype=wp.int64), p_parent: wp.array(dtype=wp.int64),
    p_cell: wp.array(dtype=wp.int32), p_seq: wp.array2d(dtype=wp.uint8),
    p_len: wp.array(dtype=wp.int32), p_state: wp.array(dtype=wp.uint8),
    p_partner: wp.array(dtype=wp.int32), p_born: wp.array(dtype=wp.int64),
    p_copy_pos: wp.array(dtype=wp.int32), p_child: wp.array(dtype=wp.int32),
    p_motifs: wp.array(dtype=wp.int32), p_mut: wp.array(dtype=wp.int32),
    p_motor: wp.array(dtype=wp.int32),
    # cell index + reservations
    order: wp.array(dtype=wp.int32), cell_start: wp.array(dtype=wp.int32),
    free_slots: wp.array(dtype=wp.int32), cell_base: wp.array(dtype=wp.int32),
    id_base: wp.int64,
    # world
    spec: wp.array3d(dtype=wp.int32), temp: wp.array3d(dtype=wp.float64),
    water: wp.array2d(dtype=wp.float64), light_water: wp.array2d(dtype=wp.float32),
    cat: wp.array3d(dtype=wp.float32), e_chem: wp.array2d(dtype=wp.float64),
    membrane_store: wp.array2d(dtype=wp.int32), pp_pre: wp.array2d(dtype=wp.int32),
    # genetic code
    motif: wp.array2d(dtype=wp.uint8), m_eff: wp.array(dtype=wp.int32),
    m_target: wp.array(dtype=wp.int32), m_strength: wp.array(dtype=wp.float32),
    m_cost: wp.array(dtype=wp.int32), nm: int, mk: int, repl_bit: int,
    # emit reaction stoichiometry (from chemistry tables)
    sub_s: wp.array2d(dtype=wp.int32), sub_c: wp.array2d(dtype=wp.int32),
    prod_s: wp.array2d(dtype=wp.int32), prod_c: wp.array2d(dtype=wp.int32), max_sub: int,
    # species indices & params
    IH2O: int, IPP: int, IPI: int, IM1: int, IL: int,
    kh_base: wp.float32, mut_rate: wp.float32, e_copy: int, l_max: int,
    dt: wp.float64, tick: wp.int64, seed: wp.int32, W: int,
    # lineage event ring
    ev_n: wp.array(dtype=wp.int32), ev_child: wp.array(dtype=wp.int64),
    ev_parent: wp.array(dtype=wp.int64), ev_cell: wp.array(dtype=wp.int32),
    ev_len: wp.array(dtype=wp.int32), ev_mut: wp.array(dtype=wp.int32),
):
    c = wp.tid()
    H = (cell_start.shape[0] - 1) / W
    s0 = cell_start[c]
    s1 = cell_start[c + 1]
    if s0 == s1:
        return
    i = c / W
    j = c % W
    res_cap = cell_base[c + 1] - cell_base[c]
    started = int(0)
    heat = wp.float64(0.0)
    st = wp.rand_init(seed, wp.int32(c))
    t1 = temp[1, i, j]

    for q in range(s0, s1):
        p = order[q]
        stt = int(p_state[p])
        if stt == P_DEAD or stt == P_BUILD:
            continue
        ln = p_len[p]

        # ---- hydrolysis (rate rises with temperature; real: hot water destroys polymers)
        kh = wp.float64(kh_base) * wp.exp(-wp.float64(KH_EA) / wp.float64(R_GAS)
                                          * (wp.float64(1.0) / t1 - wp.float64(1.0) / wp.float64(300.0)))
        pdie = wp.float64(1.0) - wp.exp(-kh * wp.float64(ln) * dt)
        if wp.float64(wp.randf(st)) < pdie:
            need = ln
            child = p_child[p]
            clen = int(0)
            if stt == P_COPYING and child >= 0 and int(p_state[child]) == P_BUILD:
                clen = p_len[child]
                need += clen
            if spec[IH2O, i, j] >= need:
                for x in range(ln):
                    spec[IM1 + int(p_seq[p, x]) - 1, i, j] += 1
                spec[IH2O, i, j] -= ln
                p_state[p] = wp.uint8(P_DEAD)
                tm = p_partner[p]
                if tm >= 0 and tm != p and int(p_state[tm]) == P_BOUND and p_partner[tm] == p:
                    p_state[tm] = wp.uint8(P_FREE)
                    p_partner[tm] = -1
                p_partner[p] = -1
                if clen > 0:
                    for x in range(clen):
                        spec[IM1 + int(p_seq[child, x]) - 1, i, j] += 1
                    spec[IH2O, i, j] -= clen
                    p_state[child] = wp.uint8(P_DEAD)
                p_child[p] = -1
                continue

        # ---- motif effects (every effect pays P~P; gates: photoactive x light, sense x [S])
        motifs = p_motifs[p]
        gate = wp.float64(1.0)
        for m in range(nm):
            if (motifs & (1 << m)) != 0:
                if m_eff[m] == 4:      # photoactive
                    lf = wp.float64(light_water[i, j]) / wp.float64(300.0)
                    if lf > wp.float64(1.0):
                        lf = wp.float64(1.0)
                    gate = gate * lf
                elif m_eff[m] == 7:    # sense[S]
                    ns = wp.float64(spec[m_target[m], i, j])
                    gate = gate * (ns / (ns + wp.float64(1000.0)))
        for m in range(nm):
            if (motifs & (1 << m)) == 0:
                continue
            eff = m_eff[m]
            if eff == 0 or eff == 4 or eff == 7:
                continue
            cost = m_cost[m]
            if spec[IPP, i, j] < cost or spec[IH2O, i, j] < cost:
                continue
            spec[IPP, i, j] -= cost
            spec[IH2O, i, j] -= cost
            spec[IPI, i, j] += 2 * cost
            heat += wp.float64(0.033) * wp.float64(cost)
            if eff == 1:               # catalyst[R]
                add = wp.float32(wp.float64(m_strength[m]) * gate)
                cat[m_target[m], i, j] = cat[m_target[m], i, j] + add
            elif eff == 2:             # binder -> aggregate
                if p_partner[p] < 0:
                    for q2 in range(q + 1, s1):
                        p2 = order[q2]
                        p2ok = int(p_state[p2]) == P_FREE and p_partner[p2] < 0
                        if p2ok and (p_motifs[p2] & (1 << m)) != 0:
                            p_partner[p] = p2
                            p_partner[p2] = p
                            break
            elif eff == 3:             # membrane: accrete L into the cell's bilayer store
                take = int(m_strength[m])
                if take > spec[IL, i, j]:
                    take = spec[IL, i, j]
                spec[IL, i, j] -= take
                membrane_store[i, j] = membrane_store[i, j] + take
            elif eff == 5:             # motor: stage move up the P~P gradient (pre-pass copy)
                best = int(-1)
                bestv = pp_pre[i, j]
                for d in range(4):
                    ni = i
                    nj = j
                    if d == 0:
                        ni = i - 1
                    elif d == 1:
                        ni = i + 1
                    elif d == 2:
                        nj = j - 1
                    else:
                        nj = j + 1
                    if ni >= 0 and nj >= 0 and ni < H and nj < W:
                        if pp_pre[ni, nj] > bestv:
                            bestv = pp_pre[ni, nj]
                            best = ni * W + nj
                p_motor[p] = best
            elif eff == 6:             # emit: force `strength` events of the target reaction
                rx = m_target[m]
                ev = int(m_strength[m])
                for u in range(max_sub):
                    su = sub_s[rx, u]
                    if su < 0:
                        break
                    avail = spec[su, i, j] // sub_c[rx, u]
                    if ev > avail:
                        ev = avail
                if ev > 0:
                    for u in range(max_sub):
                        su = sub_s[rx, u]
                        if su < 0:
                            break
                        spec[su, i, j] -= sub_c[rx, u] * ev
                    for u in range(max_sub):
                        su = prod_s[rx, u]
                        if su < 0:
                            break
                        spec[su, i, j] += prod_c[rx, u] * ev

        # ---- templated copying: progress one monomer
        if stt == P_COPYING:
            tmpl = p_partner[p]
            child = p_child[p]
            tmpl_ok = tmpl == p or (tmpl >= 0 and int(p_state[tmpl]) == P_BOUND and p_partner[tmpl] == p)
            child_ok = child >= 0 and int(p_state[child]) == P_BUILD
            if not tmpl_ok or not child_ok:
                # abort: dissolve the partial child (unit + H2O -> M), release everyone
                if child_ok:
                    clen = p_len[child]
                    if spec[IH2O, i, j] >= clen:
                        for x in range(clen):
                            spec[IM1 + int(p_seq[child, x]) - 1, i, j] += 1
                        spec[IH2O, i, j] -= clen
                        p_state[child] = wp.uint8(P_DEAD)
                        p_state[p] = wp.uint8(P_FREE)
                        p_partner[p] = -1
                        p_child[p] = -1
                else:
                    p_state[p] = wp.uint8(P_FREE)
                    p_partner[p] = -1
                    p_child[p] = -1
            else:
                pos = p_copy_pos[p]
                tl = p_len[tmpl]
                clen = p_len[child]
                if pos < tl and clen < l_max:
                    # antiparallel synthesis (real polymerases read the template
                    # 3'->5'): child = reverse complement, so reverse-complement-
                    # palindromic motifs survive copying with function intact

                    r = wp.float64(wp.randf(st))
                    er = wp.float64(mut_rate)
                    typ = int(0)
                    adv = int(1)
                    grow = int(1)
                    mut = int(0)
                    if r < er * wp.float64(0.1):          # deletion
                        grow = 0
                        mut = 1
                    elif r < er * wp.float64(0.2):        # insertion
                        typ = 1 + wp.min(int(wp.randf(st) * wp.float32(4.0)), 3)
                        adv = 0
                        mut = 1
                    elif r < er:                          # substitution
                        typ = 1 + wp.min(int(wp.randf(st) * wp.float32(4.0)), 3)
                        mut = 1
                    else:
                        typ = complement(int(p_seq[tmpl, tl - 1 - pos]))
                    if grow == 1:
                        im = IM1 + typ - 1
                        if spec[im, i, j] > 0 and spec[IPP, i, j] >= e_copy:
                            spec[im, i, j] -= 1
                            spec[IPP, i, j] -= e_copy
                            spec[IPI, i, j] += 2 * e_copy
                            heat += wp.float64(0.02)
                            p_seq[child, clen] = wp.uint8(typ)
                            p_len[child] = clen + 1
                            p_mut[child] = p_mut[child] + mut
                            p_copy_pos[p] = pos + adv
                        # else: stall (resumes when monomers/energy return)
                    else:
                        p_mut[child] = p_mut[child] + mut
                        p_copy_pos[p] = pos + adv
                if p_copy_pos[p] >= tl or p_len[child] >= l_max:
                    # complete: child becomes a free polymer
                    p_state[child] = wp.uint8(P_FREE)
                    p_motifs[child] = scan_motifs(p_seq, child, p_len[child], motif, nm, mk)
                    p_state[p] = wp.uint8(P_FREE)
                    if tmpl != p:
                        p_state[tmpl] = wp.uint8(P_FREE)
                        p_partner[tmpl] = -1
                    p_partner[p] = -1
                    p_child[p] = -1
                    e = wp.atomic_add(ev_n, 0, 1)
                    if e < EV_CAP:
                        ev_child[e] = p_id[child]
                        ev_parent[e] = p_parent[child]
                        ev_cell[e] = c
                        ev_len[e] = p_len[child]
                        ev_mut[e] = p_mut[child]

        # ---- copy start: a free replicase binds a template (or itself)
        elif stt == P_FREE and (motifs & repl_bit) != 0 and started < res_cap:
            ncand = int(1)                                 # self is always available
            for q2 in range(s0, s1):
                p2 = order[q2]
                if p2 != p and int(p_state[p2]) == P_FREE and p_partner[p2] < 0:
                    ncand += 1
            pick = wp.min(int(wp.randf(st) * wp.float32(ncand)), ncand - 1)
            tmpl = p
            if pick > 0:
                seen = int(0)
                for q2 in range(s0, s1):
                    p2 = order[q2]
                    if p2 != p and int(p_state[p2]) == P_FREE and p_partner[p2] < 0:
                        seen += 1
                        if seen == pick:
                            tmpl = p2
                            break
            res_idx = cell_base[c] + started
            child = free_slots[res_idx]
            started += 1
            p_state[p] = wp.uint8(P_COPYING)
            p_partner[p] = tmpl
            p_copy_pos[p] = 0
            p_child[p] = child
            if tmpl != p:
                p_state[tmpl] = wp.uint8(P_BOUND)
                p_partner[tmpl] = p
            p_id[child] = id_base + wp.int64(res_idx)
            p_parent[child] = p_id[tmpl]
            p_cell[child] = c
            p_len[child] = 0
            p_state[child] = wp.uint8(P_BUILD)
            p_partner[child] = -1
            p_child[child] = -1
            p_motifs[child] = 0
            p_mut[child] = 0
            p_born[child] = tick
            p_copy_pos[child] = 0

    if heat != wp.float64(0.0):
        c1 = wp.float64(6.5e5) + wp.float64(4.186e6) * water[i, j]
        temp[1, i, j] = temp[1, i, j] + heat / c1
        e_chem[i, j] = e_chem[i, j] + heat


@wp.func
def stoch_round_p(x: wp.float64, st: wp.uint32) -> int:
    fl = wp.floor(x)
    n = int(fl)
    if wp.float64(wp.randf(st)) < x - fl:
        n += 1
    return n


@wp.kernel
def k_membrane(store_old: wp.array2d(dtype=wp.int32), store_new: wp.array2d(dtype=wp.int32),
               comp_id: wp.array2d(dtype=wp.int32), spec: wp.array3d(dtype=wp.int32),
               l_pre: wp.array2d(dtype=wp.int32), IL: int, seed: wp.int32, H: int, W: int):
    i, j = wp.tid()
    s = store_old[i, j]
    st = wp.rand_init(seed, wp.int32(i * W + j))
    dec = stoch_round_p(wp.float64(s) * wp.float64(0.002), st)   # bilayer turnover
    # split: an overloaded neighbor sheds a quarter of its store to its L-richest neighbor
    gain = int(0)
    loss = int(0)
    if s > 2 * MEM_THRESH:
        loss = s / 4
    for d in range(4):
        ni = i
        nj = j
        if d == 0:
            ni = i - 1
        elif d == 1:
            ni = i + 1
        elif d == 2:
            nj = j - 1
        else:
            nj = j + 1
        if ni < 0 or nj < 0 or ni >= H or nj >= W:
            continue
        if store_old[ni, nj] > 2 * MEM_THRESH:
            # am I their L-richest neighbor? (pure function of pre-pass L, NESW tie-break)
            best_d = int(-1)
            best_v = int(-1)
            for e in range(4):
                mi = ni
                mj = nj
                if e == 0:
                    mi = ni - 1
                elif e == 1:
                    mi = ni + 1
                elif e == 2:
                    mj = nj - 1
                else:
                    mj = nj + 1
                if mi < 0 or mj < 0 or mi >= H or mj >= W:
                    continue
                if l_pre[mi, mj] > best_v:
                    best_v = l_pre[mi, mj]
                    best_d = e
            tgt_i = ni
            tgt_j = nj
            if best_d == 0:
                tgt_i = ni - 1
            elif best_d == 1:
                tgt_i = ni + 1
            elif best_d == 2:
                tgt_j = nj - 1
            elif best_d == 3:
                tgt_j = nj + 1
            if tgt_i == i and tgt_j == j:
                gain += store_old[ni, nj] / 4
    if loss > 0:
        # verify I actually have a neighbor (edges): recompute my own target existence
        has = int(0)
        for e in range(4):
            mi = i
            mj = j
            if e == 0:
                mi = i - 1
            elif e == 1:
                mi = i + 1
            elif e == 2:
                mj = j - 1
            else:
                mj = j + 1
            if mi >= 0 and mj >= 0 and mi < H and mj < W:
                has = 1
        if has == 0:
            loss = 0
    ns = s - dec - loss + gain
    if ns < 0:
        ns = 0
    store_new[i, j] = ns
    spec[IL, i, j] = spec[IL, i, j] + dec
    if ns > MEM_THRESH:
        comp_id[i, j] = i * W + j + 1
    else:
        comp_id[i, j] = 0


@wp.kernel
def k_transport(p_state: wp.array(dtype=wp.uint8), p_cell: wp.array(dtype=wp.int32),
                p_motor: wp.array(dtype=wp.int32), comp_id: wp.array2d(dtype=wp.int32),
                fx: wp.array2d(dtype=wp.float64), fy: wp.array2d(dtype=wp.float64),
                h0: wp.array2d(dtype=wp.float64), seed: wp.int32, H: int, W: int):
    p = wp.tid()
    if int(p_state[p]) != P_FREE:
        p_motor[p] = -1
        return
    c = p_cell[p]
    i = c / W
    j = c % W
    dest = int(-1)
    if p_motor[p] >= 0:
        dest = p_motor[p]
        p_motor[p] = -1
    else:
        st = wp.rand_init(seed, wp.int32(p))
        r = wp.float64(wp.randf(st))
        hh = h0[i, j]
        acc = wp.float64(0.0)
        for k in range(4):
            f = wp.float64(0.0)
            ni = i
            nj = j
            if k == 0 and j + 1 < W:
                f = fx[i, j]
                nj = j + 1
            elif k == 1 and j - 1 >= 0:
                f = -fx[i, j - 1]
                nj = j - 1
            elif k == 2 and i + 1 < H:
                f = fy[i, j]
                ni = i + 1
            elif k == 3 and i - 1 >= 0:
                f = -fy[i - 1, j]
                ni = i - 1
            frac = wp.float64(0.01)                     # diffusion floor
            if f > wp.float64(0.0) and hh > wp.float64(1e-4):
                frac += wp.min(f / hh, wp.float64(0.45))
            acc += frac
            if r < acc:
                dest = ni * W + nj
                break
    if dest >= 0 and dest != c:
        di = dest / W
        dj = dest % W
        if comp_id[i, j] == comp_id[di, dj] or (comp_id[i, j] == 0 and comp_id[di, dj] == 0):
            p_cell[p] = dest


@wp.kernel
def k_snapshot_plane(spec: wp.array3d(dtype=wp.int32), idx: int, out: wp.array2d(dtype=wp.int32)):
    i, j = wp.tid()
    out[i, j] = spec[idx, i, j]


class Polymers:
    def __init__(self, cfg, chem) -> None:
        self.cfg = cfg
        self.chem = chem
        self.key_copy = mix(cfg.run.seed, SALT_POLY_COPY)
        self.key_move = mix(cfg.run.seed, SALT_POLY_MOVE)
        self.lineage = None
        self.recent_events = []
        with open(cfg.polymers.genetic_code) as f:
            self.code = yaml.safe_load(f)
        self.mk = int(self.code["k"])
        names = list(self.code["motifs"].keys())
        self.motif_names = names
        self.nm = len(names)
        sym = {"M1": 1, "M2": 2, "M3": 3, "M4": 4}
        self.motif_np = np.array([[sym[s] for s in self.code["motifs"][n]["seq"]] for n in names],
                                 dtype=np.uint8)
        self.eff_np = np.array([EFF[self.code["motifs"][n]["effect"]] for n in names], np.int32)
        tgt = []
        for n in names:
            md = self.code["motifs"][n]
            t = md.get("target")
            if md["effect"] == "catalyst" or md["effect"] == "emit":
                tgt.append([r["name"] for r in chem.reactions].index(t))
            elif md["effect"] == "sense":
                tgt.append(chem.index[t])
            else:
                tgt.append(-1)
        self.tgt_np = np.array(tgt, np.int32)
        self.str_np = np.array([self.code["motifs"][n].get("strength", 1.0) for n in names], np.float32)
        self.cost_np = np.array([self.code["motifs"][n].get("cost_pp", 0) for n in names], np.int32)
        self.repl_bit = 1 << names.index("replicase")
        # verify motifs are palindromic under the complement map (heredity-safe)
        comp = {1: 2, 2: 1, 3: 4, 4: 3}
        for row, n in zip(self.motif_np, names):
            rc = [comp[int(x)] for x in row[::-1]]
            if list(row) != rc:
                raise ValueError(f"motif {n} is not palindromic under complement")

    def bind(self, state) -> None:
        s = state
        dev = s.device
        h, w = s.shape
        for m in ("M2", "M3", "M4"):
            assert self.chem.index[m] == self.chem.index["M1"] + int(m[1]) - 1, "M1..M4 must be contiguous"
        s.membrane_store = wp.zeros((h, w), dtype=wp.int32, device=dev)
        s.p_mut = wp.zeros(s.p_cap, dtype=wp.int32, device=dev)
        s.p_motor = wp.full(s.p_cap, -1, dtype=wp.int32, device=dev)
        arr = lambda a, dt: wp.array(a, dtype=dt, device=dev)  # noqa: E731
        self.g_motif = arr(self.motif_np, wp.uint8)
        self.g_eff = arr(self.eff_np, wp.int32)
        self.g_tgt = arr(self.tgt_np, wp.int32)
        self.g_str = arr(self.str_np, wp.float32)
        self.g_cost = arr(self.cost_np, wp.int32)
        self.g_order = wp.zeros(s.p_cap, dtype=wp.int32, device=dev)
        self.g_cell_start = wp.zeros(h * w + 1, dtype=wp.int32, device=dev)
        self.g_free = wp.zeros(s.p_cap, dtype=wp.int32, device=dev)
        self.g_cell_base = wp.zeros(h * w + 1, dtype=wp.int32, device=dev)
        self.g_pp_pre = wp.zeros((h, w), dtype=wp.int32, device=dev)
        self.g_l_pre = wp.zeros((h, w), dtype=wp.int32, device=dev)
        self.g_store_new = wp.zeros((h, w), dtype=wp.int32, device=dev)
        self.g_ev_n = wp.zeros(1, dtype=wp.int32, device=dev)
        self.g_ev_child = wp.zeros(EV_CAP, dtype=wp.int64, device=dev)
        self.g_ev_parent = wp.zeros(EV_CAP, dtype=wp.int64, device=dev)
        self.g_ev_cell = wp.zeros(EV_CAP, dtype=wp.int32, device=dev)
        self.g_ev_len = wp.zeros(EV_CAP, dtype=wp.int32, device=dev)
        self.g_ev_mut = wp.zeros(EV_CAP, dtype=wp.int32, device=dev)
        self.fluid = None   # wired by cli (fx/fy/h0 buffers)
        self.idx = self.chem.index

    def step(self, s, tick: int) -> None:
        h, w = s.shape
        ncell = h * w
        # ---- host: rebuild cell index and reservations from a device snapshot
        st_np = s.p_state.numpy()
        alive = st_np > 0
        n_live = int(alive.sum())
        s.p_count = n_live
        if n_live == 0 and int(s.membrane_store.numpy().sum()) == 0:
            return
        cell_np = s.p_cell.numpy()
        id_np = s.p_id.numpy()
        mot_np = s.p_motifs.numpy()
        slots = np.nonzero(alive)[0]
        order = slots[np.lexsort((id_np[slots], cell_np[slots]))].astype(np.int32)
        counts = np.bincount(cell_np[slots], minlength=ncell)
        cell_start = np.zeros(ncell + 1, dtype=np.int32)
        np.cumsum(counts, out=cell_start[1:])
        starters = (st_np == P_FREE) & ((mot_np & self.repl_bit) != 0)
        b = np.bincount(cell_np[np.nonzero(starters)[0]], minlength=ncell)
        cell_base = np.zeros(ncell + 1, dtype=np.int32)
        np.cumsum(b, out=cell_base[1:])
        total_res = int(cell_base[-1])
        free = np.nonzero(~alive)[0].astype(np.int32)
        if total_res > len(free):
            log.error("polymer capacity exhausted", extra={"live": n_live, "cap": s.p_cap})
            raise RuntimeError("polymer capacity exhausted — raise polymers.capacity")
        dev = s.device
        wp.copy(self.g_order, wp.array(np.ascontiguousarray(order), dtype=wp.int32, device=dev),
                count=len(order))
        wp.copy(self.g_cell_start, wp.array(cell_start, dtype=wp.int32, device=dev))
        wp.copy(self.g_cell_base, wp.array(cell_base, dtype=wp.int32, device=dev))
        if total_res:
            wp.copy(self.g_free, wp.array(np.ascontiguousarray(free[:total_res]), dtype=wp.int32,
                    device=dev), count=total_res)

        s.catalyst.zero_()
        self.g_ev_n.zero_()
        wp.launch(k_snapshot_plane, dim=s.shape, inputs=[s.species, self.idx["PP"], self.g_pp_pre],
                  device=dev)

        cfgp = self.cfg.polymers
        wp.launch(k_cell_pass, dim=ncell, inputs=[
            s.p_id, s.p_parent, s.p_cell, s.p_seq, s.p_len, s.p_state, s.p_partner, s.p_born,
            s.p_copy_pos, s.p_child, s.p_motifs, s.p_mut, s.p_motor,
            self.g_order, self.g_cell_start, self.g_free, self.g_cell_base,
            wp.int64(s.next_poly_id),
            s.species, s.temp, s.water_depth, s.light_water, s.catalyst, s.e_chem,
            s.membrane_store, self.g_pp_pre,
            self.g_motif, self.g_eff, self.g_tgt, self.g_str, self.g_cost,
            self.nm, self.mk, self.repl_bit,
            self.chem.g_sub_s, self.chem.g_sub_c, self.chem.g_prod_s, self.chem.g_prod_c, 6,
            self.idx["H2O"], self.idx["PP"], self.idx["Pi"], self.idx["M1"], self.idx["L"],
            wp.float32(cfgp.hydrolysis_base_rate), wp.float32(cfgp.mutation_rate_per_monomer),
            cfgp.copy_energy_per_monomer, cfgp.max_length,
            wp.float64(self.cfg.run.dt_seconds), wp.int64(tick),
            wp.int32(seed32(self.key_copy, tick)), w,
            self.g_ev_n, self.g_ev_child, self.g_ev_parent, self.g_ev_cell,
            self.g_ev_len, self.g_ev_mut], device=dev)
        s.next_poly_id += total_res

        # ---- membranes / compartments (two-pass, pull-based split)
        wp.launch(k_snapshot_plane, dim=s.shape, inputs=[s.species, self.idx["L"], self.g_l_pre], device=dev)
        wp.launch(k_membrane, dim=s.shape, inputs=[s.membrane_store, self.g_store_new,
                  s.compartment_id, s.species, self.g_l_pre, self.idx["L"],
                  wp.int32(seed32(self.key_copy, tick, 1)), h, w], device=dev)
        s.membrane_store, self.g_store_new = self.g_store_new, s.membrane_store

        # ---- transport (water drift + diffusion + staged motor moves)
        if self.fluid is not None and self.fluid.b is not None:
            fb = self.fluid.b
            wp.launch(k_transport, dim=s.p_cap, inputs=[s.p_state, s.p_cell, s.p_motor,
                      s.compartment_id, fb["fx"], fb["fy"], fb["h0"],
                      wp.int32(seed32(self.key_move, tick)), h, w], device=dev)

        # ---- lineage events out (sorted by child id -> deterministic DB)
        nev = int(self.g_ev_n.numpy()[0])
        if nev > 0:
            nev = min(nev, EV_CAP)
            ch = self.g_ev_child.numpy()[:nev]
            srt = np.argsort(ch, kind="stable")
            evs = list(zip(ch[srt].tolist(), self.g_ev_parent.numpy()[:nev][srt].tolist(),
                           self.g_ev_cell.numpy()[:nev][srt].tolist(),
                           self.g_ev_len.numpy()[:nev][srt].tolist(),
                           self.g_ev_mut.numpy()[:nev][srt].tolist()))
            self.recent_events = evs
            if self.lineage is not None:
                self.lineage.add_copies(tick, evs)
        else:
            self.recent_events = []
