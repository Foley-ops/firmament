"""God console: natural events only — nothing an inhabitant could only explain by
invoking a god. Every event is logged BEFORE applying and applies on a tick
boundary, so replay is exact. Matter/energy/water injections register with the
audit so the books stay closed.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from firmament.core.rng import SALT_EVENT, np_rng
from firmament.io.logging import get

log = get("console")

NATURAL_EVENTS = ("rain", "drought", "flood", "earthquake", "volcano", "meteor",
                  "climate", "solar", "add_land", "add_species")


def request(sched, event: str, params: dict) -> dict:
    """Log then queue a natural event for the next tick boundary."""
    if event not in NATURAL_EVENTS:
        raise ValueError(f"not a natural event: {event}")
    tick = sched.state.tick + 1
    rec = sched.events.append(event, tick, component="operator", params=params)
    sched.pending_events.append(lambda s, t: _apply(sched, event, params, rec["n"], tick))
    return rec


def apply_event(sched, rec: dict, replaying: bool = False) -> None:
    sched.pending_events.append(
        lambda s, t: _apply(sched, rec["event"], rec["params"], rec["n"], rec["tick"]))


def _region_mask(shape, params) -> np.ndarray:
    h, w = shape
    cx, cy = params.get("cx", w // 2), params.get("cy", h // 2)
    r = params.get("radius", max(h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def _apply(sched, event: str, params: dict, event_n: int, at_tick: int) -> None:
    s = sched.state
    aud = getattr(sched, "audit", None)
    h, w = s.shape
    # rng keyed by the LOGGED tick, not the application instant -> replay-exact
    rng = np_rng(sched.cfg.run.seed, SALT_EVENT, at_tick * 1000 + event_n)
    mask = _region_mask(s.shape, params)
    log.info(f"applying {event}", extra={"params": params, "tick": s.tick})

    if event == "rain":                    # storm system moving in from off-patch
        amount = float(params.get("intensity_mm", 10.0))
        vap = s.vapor.numpy()
        vap[mask] += amount
        s.vapor = wp.array(vap, dtype=wp.float64, device=s.device)
        if aud:
            aud.register_injection(water=amount * mask.sum(),
                                   energy=amount * mask.sum() * (2.5e6 + 4186.0 * 288.0))
    elif event == "drought":               # dry air mass replaces the column
        f = float(params.get("strength", 0.5))
        vap = s.vapor.numpy()
        removed = vap[mask].sum() * f
        vap[mask] *= 1.0 - f
        s.vapor = wp.array(vap, dtype=wp.float64, device=s.device)
        if aud:
            aud.register_injection(water=-removed,
                                   energy=-removed * (2.5e6 + 4186.0 * 288.0))
    elif event == "flood":
        rise = float(params.get("rise_m", 0.5))
        wd = s.water_depth.numpy()
        wd[mask] += rise
        s.water_depth = wp.array(wd, dtype=wp.float64, device=s.device)
        if aud:
            # flood water arrives at the local surface temperature: exact stored delta
            t1 = s.temp.numpy()[1]
            aud.register_injection(water=rise * mask.sum() * 1000.0,
                                   energy=float((rise * 4.186e6 * t1[mask]).sum()))
    elif event == "earthquake":
        mag = float(params.get("magnitude_m", 1.0))
        el = s.elevation.numpy()
        el[mask] += rng.standard_normal(int(mask.sum())) * mag
        s.elevation = wp.array(el, dtype=wp.float64, device=s.device)
    elif event == "volcano":
        heat = float(params.get("heat_J_m2", 5e7))
        el = s.elevation.numpy()
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = params.get("cx", w // 2), params.get("cy", h // 2)
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        el += float(params.get("cone_m", 3.0)) * np.exp(-r2 / 50.0)
        s.elevation = wp.array(el, dtype=wp.float64, device=s.device)
        t = s.temp.numpy()
        dtk = heat * np.exp(-r2 / 200.0) / 4.3e6
        t[2] += dtk
        s.temp = wp.array(t, dtype=wp.float64, device=s.device)
        inj = np.zeros((h, w), dtype=np.int32)
        inj[mask] = int(params.get("mineral_counts", 10000))
        spec = s.species.numpy()
        chem = sched.chem
        spec[chem.index["FeO"]] += inj
        spec[chem.index["H2S"]] += inj // 2
        s.species = wp.array(spec, dtype=wp.int32, device=s.device)
        if aud:
            el_counts = (chem.comp[chem.index["FeO"]] * int(inj.sum())
                         + chem.comp[chem.index["H2S"]] * int(inj.sum() // 2))
            aud.register_injection(element_counts=el_counts.astype(np.int64),
                                   energy=float((dtk * 4.3e6).sum()))
    elif event == "meteor":
        e_j = float(params.get("energy_J", 1e12))
        cx, cy = params.get("cx", w // 2), params.get("cy", h // 2)
        yy, xx = np.mgrid[0:h, 0:w]
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        el = s.elevation.numpy()
        el -= float(params.get("crater_m", 2.0)) * np.exp(-r2 / 100.0)
        s.elevation = wp.array(el, dtype=wp.float64, device=s.device)
        t = s.temp.numpy()
        dtk = (e_j / (h * w)) * np.exp(-r2 / 400.0) / 6.5e5
        t[1] += dtk
        s.temp = wp.array(t, dtype=wp.float64, device=s.device)
        dust = s.albedo_dust.numpy()
        dust += float(params.get("dust", 0.3)) * np.exp(-r2 / (2 * (h / 4) ** 2)).astype(np.float32)
        s.albedo_dust = wp.array(np.clip(dust, 0, 1), dtype=wp.float32, device=s.device)
        if aud:
            aud.register_injection(energy=float((dtk * 6.5e5).sum()))
    elif event in ("climate", "solar"):    # insolation multiplier (ice age / warm / solar)
        s.solar_mult = float(params.get("multiplier", 1.0))
    elif event == "add_land":
        _add_land(sched, int(params.get("cols", 64)))
    elif event == "add_species":
        _add_species(sched, params)


def _add_land(sched, cols: int) -> None:
    """Extend the grid eastward; new terrain dead and inert; existing cells untouched."""
    raise NotImplementedError(
        "add_land requires a full state rebuild; run `firmament fork` into a wider "
        "world config instead (v0 limitation, logged as an open question)")


def _add_species(sched, params: dict) -> None:
    """Append species/reactions; refuse if any existing reaction rate would change."""
    import yaml as _yaml
    new_file = params["file"]
    with open(new_file) as f:
        new_doc = _yaml.safe_load(f)
    old = sched.chem
    for k, r_old in enumerate(old.reactions):
        if k >= len(new_doc["reactions"]) or new_doc["reactions"][k] != r_old:
            raise ValueError("add_species refused: existing reaction table would change")
    if list(new_doc["species"].keys())[: old.n_species] != old.names:
        raise ValueError("add_species refused: existing species order would change")
    raise NotImplementedError(
        "add_species validated but hot-reload requires a state rebuild; fork into a "
        "run with the new chemistry file (v0 limitation)")


def place_seed(sched, state, sequence: str, x: int, y: int) -> None:
    """Event #0: the one act of creation. Logs the full sequence."""
    import numpy as np

    sym = {"M1": 1, "M2": 2, "M3": 3, "M4": 4}
    toks = [sequence[i:i + 2] for i in range(0, len(sequence), 2)]
    seq = np.array([sym[t] for t in toks], dtype=np.uint8)
    poly = sched.poly
    h, w = state.shape
    cell = y * w + x
    slot = 0
    assert int(state.p_state.numpy()[slot]) == 0, "slot 0 not free — seed must be first"
    dev = state.device

    def setv(arr, val, dtype):
        a = arr.numpy()
        a[slot] = val
        return wp.array(a, dtype=dtype, device=dev)

    state.p_id = setv(state.p_id, 1, wp.int64)
    state.p_parent = setv(state.p_parent, 0, wp.int64)
    state.p_cell = setv(state.p_cell, cell, wp.int32)
    sq = state.p_seq.numpy()
    sq[slot, :] = 0
    sq[slot, :len(seq)] = seq
    state.p_seq = wp.array(sq, dtype=wp.uint8, device=dev)
    state.p_len = setv(state.p_len, len(seq), wp.int32)
    state.p_state = setv(state.p_state, 1, wp.uint8)
    state.p_partner = setv(state.p_partner, -1, wp.int32)
    state.p_child = setv(state.p_child, -1, wp.int32)
    state.p_born = setv(state.p_born, state.tick, wp.int64)
    # motif scan (host mirror of the kernel scan)
    mask = 0
    for mi, row in enumerate(poly.motif_np):
        k = len(row)
        for pos in range(len(seq) - k + 1):
            if np.array_equal(seq[pos:pos + k], row):
                mask |= 1 << mi
                break
    state.p_motifs = setv(state.p_motifs, mask, wp.int32)
    state.next_poly_id = max(state.next_poly_id, 2)
    # the seed's monomers come from the cell's inventory (conservation: the world
    # must contain the atoms; audit counts chain units). Deduct from species.
    chem = sched.chem
    spec = state.species.numpy()
    counts = np.bincount(seq, minlength=5)
    yx = (y, x)
    for m in range(1, 5):
        idx = chem.index[f"M{m}"]
        if spec[idx][yx] < counts[m]:
            raise RuntimeError(f"seed cell lacks M{m}: have {spec[idx][yx]}, need {counts[m]}")
        spec[idx][yx] -= counts[m]
    spec[chem.index["H2O"]][yx] += len(seq)     # condensation releases water
    state.species = wp.array(spec, dtype=wp.int32, device=dev)
    sched.events.append("seed_placed", state.tick, component="operator",
                        sequence=sequence, cell=[x, y], length=int(len(seq)),
                        motif_mask=mask)
    log.info("SEED PLACED — event #0", extra={"cell": [x, y], "length": int(len(seq))})
