"""God console: natural events only — nothing an inhabitant could only explain by
invoking a god. Every event is logged BEFORE applying and applies on a tick
boundary, so replay is exact. Matter/energy/water injections register with the
audit so the books stay closed.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from firmament.core.rng import SALT_EVENT
from firmament.io.logging import get

log = get("console")

NATURAL_EVENTS = ("rain", "drought", "flood", "earthquake", "volcano", "meteor",
                  "climate", "solar", "add_land", "add_species")


def request(sched, event: str, params: dict) -> None:
    """Queue a natural event; it is logged and applied at the next tick boundary
    (firmament/operator/causal.py), never from the requesting thread."""
    validate_natural(sched, event, params)
    from firmament.operator import causal
    causal.submit(sched, event, params)


def _region_mask(shape, params) -> np.ndarray:
    h, w = shape
    cx, cy = params.get("cx", w // 2), params.get("cy", h // 2)
    r = params.get("radius", max(h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


# natural-event parameters: name -> (min, max); plus cx, cy, radius for every event
EVENT_PARAMS = {
    "rain": {"intensity_mm": (0.0, 1000.0)},
    "drought": {"strength": (0.0, 1.0)},
    "flood": {"rise_m": (0.0, 100.0)},
    "earthquake": {"magnitude_m": (0.0, 100.0)},
    "volcano": {"heat_J_m2": (0.0, 1e10), "cone_m": (0.0, 100.0), "mineral_counts": (0, 10_000_000)},
    "meteor": {"energy_J": (0.0, 1e16), "crater_m": (0.0, 100.0), "dust": (0.0, 1.0)},
    "climate": {"multiplier": (0.01, 10.0)},
    "solar": {"multiplier": (0.01, 10.0)},
}


def validate_natural(sched, event: str, params: dict) -> None:
    """Reject a command BEFORE it can reach the causal log: unknown events, unknown or
    non-numeric/non-finite parameters, and out-of-range values (e.g. a drought strength
    above 1 drives vapor negative; a GUI typo arrives as null)."""
    if event in ("add_land", "add_species"):
        raise ValueError(f"{event} is not supported in this ruleset — fork into a new config")
    if event not in EVENT_PARAMS:
        raise ValueError(f"not a natural event: {event}")
    h, w = sched.state.shape
    allowed = EVENT_PARAMS[event] | {"cx": (0, w - 1), "cy": (0, h - 1),
                                     "radius": (0.5, 10.0 * max(h, w))}
    for k, v in params.items():
        if k not in allowed:
            raise ValueError(f"{event}: unknown parameter {k!r}")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
            raise ValueError(f"{event}: parameter {k} must be a finite number, got {v!r}")
        lo, hi = allowed[k]
        if not lo <= v <= hi:
            raise ValueError(f"{event}: {k}={v} outside [{lo}, {hi}]")
    if "mineral_counts" in params and int(params["mineral_counts"]) != params["mineral_counts"]:
        raise ValueError("volcano: mineral_counts must be an integer")


def _event_rng(seed: int, at_tick: int, event_n: int) -> np.random.Generator:
    """Keyed by (tick, causal n) as separate Philox counter words: no collisions (v0.1
    used tick*1000 + n, so tick 1/n 0 and tick 0/n 1000 drew identical noise)."""
    from firmament.core.rng import mix
    return np.random.Generator(np.random.Philox(
        key=np.uint64(mix(seed, SALT_EVENT)),
        counter=[0, 0, np.uint64(event_n), np.uint64(at_tick)]))


def _assign(state, name: str, arr: np.ndarray) -> None:
    """Write into the EXISTING device array. Rebinding a state array to a new wp.array
    would leave the fluid CUDA graph (fixed pointers) integrating stale memory on GPU."""
    dst = getattr(state, name)
    wp.copy(dst, wp.array(np.ascontiguousarray(arr), dtype=dst.dtype, device=state.device))


def apply_natural(sched, event: str, params: dict, event_n: int, at_tick: int) -> None:
    from firmament.core import physconst as pc
    s = sched.state
    aud = getattr(sched, "audit", None)
    h, w = s.shape
    # rng keyed by CAUSAL identity only (tick + causal sequence number); analysis output
    # has its own log and counter, so observation can never change this draw
    rng = _event_rng(sched.cfg.run.seed, at_tick, event_n)
    mask = _region_mask(s.shape, params)
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = (yy - params.get("cy", h // 2)) ** 2 + (xx - params.get("cx", w // 2)) ** 2
    h_vap = pc.LV + pc.CW_SP * pc.T_REF
    log.info(f"applying {event}", extra={"params": params, "tick": s.tick})

    if event == "rain":                    # storm system moving in from off-patch
        amount = float(params.get("intensity_mm", 10.0))
        vap = s.vapor.numpy().copy()
        vap[mask] += amount
        _assign(s, "vapor", vap)
        if aud:
            aud.register_injection(water=amount * mask.sum(), energy=amount * mask.sum() * h_vap)
    elif event == "drought":               # dry air mass replaces the column
        f = float(params.get("strength", 0.5))
        vap = s.vapor.numpy().copy()
        removed = float(vap[mask].sum() * f)
        vap[mask] *= 1.0 - f
        _assign(s, "vapor", vap)
        if aud:
            aud.register_injection(water=-removed, energy=-removed * h_vap)
    elif event == "flood":
        rise = float(params.get("rise_m", 0.5))
        wd = s.water_depth.numpy().copy()
        wd[mask] += rise
        _assign(s, "water_depth", wd)
        if aud:
            # flood water arrives at the local surface temperature: exact stored delta
            t1 = s.temp.numpy()[1]
            aud.register_injection(water=rise * mask.sum() * pc.RHO_W,
                                   energy=float((rise * pc.CW_VOL * t1[mask]).sum()))
    elif event == "earthquake":
        mag = float(params.get("magnitude_m", 1.0))
        el = s.elevation.numpy().copy()
        el[mask] += rng.standard_normal(int(mask.sum())) * mag
        _assign(s, "elevation", el)
    elif event == "volcano":
        heat = float(params.get("heat_J_m2", 5e7))
        el = s.elevation.numpy().copy()
        el += float(params.get("cone_m", 3.0)) * np.exp(-r2 / 50.0)
        _assign(s, "elevation", el)
        e_cell = heat * np.exp(-r2 / 200.0)                 # J/m^2 into the sediment layer
        t = s.temp.numpy().copy()
        t[2] += e_cell / pc.C_SED
        _assign(s, "temp", t)
        inj = np.zeros((h, w), dtype=np.int64)
        inj[mask] = int(params.get("mineral_counts", 10000))
        chem = sched.chem
        spec = s.species.numpy().copy()
        spec[chem.index["FeO"]] += inj.astype(np.int32)
        spec[chem.index["H2S"]] += (inj // 2).astype(np.int32)
        _assign(s, "species", spec)
        if aud:
            # book exactly what was added per cell (sum of floors, not floor of sum)
            el_counts = (chem.comp[chem.index["FeO"]].astype(np.int64) * int(inj.sum())
                         + chem.comp[chem.index["H2S"]].astype(np.int64) * int((inj // 2).sum()))
            aud.register_injection(element_counts=el_counts, energy=float(e_cell.sum()))
    elif event == "meteor":
        e_j = float(params.get("energy_J", 1e12))
        el = s.elevation.numpy().copy()
        el -= float(params.get("crater_m", 2.0)) * np.exp(-r2 / 100.0)
        _assign(s, "elevation", el)
        # deposit a fixed ENERGY per cell and raise temperature by energy / the layer's
        # real capacity (rock + water); v0.1 used the dry-rock capacity for the
        # temperature rise, so over water it created energy the audit never saw
        e_cell = (e_j / (h * w)) * np.exp(-r2 / 400.0)
        c1 = pc.C_SURF_DRY + pc.CW_VOL * s.water_depth.numpy()
        t = s.temp.numpy().copy()
        t[1] += e_cell / c1
        _assign(s, "temp", t)
        dust = s.albedo_dust.numpy().copy()
        dust += float(params.get("dust", 0.3)) * np.exp(-r2 / (2 * (h / 4) ** 2)).astype(np.float32)
        _assign(s, "albedo_dust", np.clip(dust, 0, 1))
        if aud:
            aud.register_injection(energy=float(e_cell.sum()))
    elif event in ("climate", "solar"):    # insolation multiplier (ice age / warm / solar)
        s.solar_mult = float(params.get("multiplier", 1.0))
    else:
        raise ValueError(f"cannot apply {event}")


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


def _parse_seq(sequence: str) -> np.ndarray:
    sym = {"M1": 1, "M2": 2, "M3": 3, "M4": 4}
    if not isinstance(sequence, str) or len(sequence) % 2:
        raise ValueError("sequence must be a string of M1..M4 tokens")
    toks = [sequence[i:i + 2] for i in range(0, len(sequence), 2)]
    bad = [t for t in toks if t not in sym]
    if bad:
        raise ValueError(f"sequence has invalid monomer tokens {bad[:3]}")
    return np.array([sym[t] for t in toks], dtype=np.uint8)


def validate_seed(sched, params: dict) -> None:
    from firmament.core.polymers import MIN_LEN
    seq = _parse_seq(params.get("sequence"))
    if not MIN_LEN <= len(seq) <= sched.cfg.polymers.max_length:
        raise ValueError(f"seed length {len(seq)} outside [{MIN_LEN}, "
                         f"{sched.cfg.polymers.max_length}]")
    x, y = params.get("cell", (None, None))
    h, w = sched.state.shape
    if not (isinstance(x, int) and isinstance(y, int) and 0 <= x < w and 0 <= y < h):
        raise ValueError(f"seed cell {params.get('cell')} outside the {w}x{h} world")


def place_seed(sched, state, sequence: str, x: int, y: int) -> dict:
    """Event #0: the one act of creation. Validated first, then committed to the causal
    log with its full sequence and cell, so replay recreates it exactly."""
    x, y = int(x), int(y)
    validate_seed(sched, {"sequence": sequence, "cell": [x, y]})
    seq = _parse_seq(sequence)
    spec = state.species.numpy()
    counts = np.bincount(seq, minlength=5)
    for m in range(1, 5):
        have = int(spec[sched.chem.index[f"M{m}"], y, x])
        if have < counts[m]:
            raise RuntimeError(f"seed cell lacks M{m}: have {have}, need {counts[m]}")
    if int(state.p_state.numpy()[0]) != 0:
        raise RuntimeError("slot 0 not free — the seed must be the first polymer")
    from firmament.operator import causal
    rec = causal.commit_now(sched, "seed_placed", {"sequence": sequence, "cell": [x, y]})
    log.info("SEED PLACED — event #0", extra={"cell": [x, y], "length": int(len(seq))})
    return rec


def apply_seed(sched, params: dict) -> None:
    state = sched.state
    seq = _parse_seq(params["sequence"])
    x, y = params["cell"]
    poly = sched.poly
    h, w = state.shape
    slot = 0

    def setv(name, val):
        a = getattr(state, name).numpy().copy()
        a[slot] = val
        _assign(state, name, a)

    setv("p_id", 1)
    setv("p_parent", 0)
    setv("p_cell", y * w + x)
    sq = state.p_seq.numpy().copy()
    sq[slot, :] = 0
    sq[slot, :len(seq)] = seq
    _assign(state, "p_seq", sq)
    setv("p_len", len(seq))
    setv("p_state", 1)
    setv("p_partner", -1)
    setv("p_child", -1)
    setv("p_born", state.tick)
    setv("p_motifs", poly.motif_mask_host(seq))
    state.next_poly_id = max(state.next_poly_id, 2)
    # the seed is built from the cell's own monomers (conservation); condensation
    # releases one H2O per chain unit
    chem = sched.chem
    spec = state.species.numpy().copy()
    counts = np.bincount(seq, minlength=5)
    for m in range(1, 5):
        spec[chem.index[f"M{m}"], y, x] -= counts[m]
    spec[chem.index["H2O"], y, x] += len(seq)
    _assign(state, "species", spec)
