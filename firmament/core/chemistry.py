"""Chemistry: data-driven reaction network, tau-leaping on integer counts.

Loader validates: every reaction balances per element; every species has at least
one source and one sink. Per-cell serial reaction loop -> no atomics, deterministic.
Reaction heat goes to the surface layer and the e_chem accumulator (audit).
"""
from __future__ import annotations

import numpy as np
import warp as wp
import yaml

from firmament.core.rng import SALT_CHEM, mix, seed32
from firmament.io.logging import get

log = get("chemistry")

R_GAS = 8.314
MAX_SUB = 6          # max distinct reactant slots per reaction


@wp.kernel
def k_chemistry(spec: wp.array3d(dtype=wp.int32), temp: wp.array3d(dtype=wp.float64),
                light_water: wp.array2d(dtype=wp.float32), cat: wp.array3d(dtype=wp.float32),
                e_chem: wp.array2d(dtype=wp.float64), water: wp.array2d(dtype=wp.float64),
                sub_s: wp.array2d(dtype=wp.int32), sub_c: wp.array2d(dtype=wp.int32),
                prod_s: wp.array2d(dtype=wp.int32), prod_c: wp.array2d(dtype=wp.int32),
                k300: wp.array(dtype=wp.float32), ea: wp.array(dtype=wp.float32),
                dh: wp.array(dtype=wp.float32), photo: wp.array(dtype=wp.int32),
                catz: wp.array(dtype=wp.int32), n_react: int, norm: wp.float64,
                seed: wp.int32, W: int, dt_scale: wp.float64):
    i, j = wp.tid()
    t1 = temp[1, i, j]
    heat = wp.float64(0.0)
    for r in range(n_react):
        kk = wp.float64(k300[r]) * wp.exp(-wp.float64(ea[r]) / wp.float64(R_GAS)
                                          * (wp.float64(1.0) / t1 - wp.float64(1.0) / wp.float64(300.0)))
        if photo[r] != 0:
            lf = wp.float64(light_water[i, j]) / wp.float64(300.0)
            if lf > wp.float64(3.0):
                lf = wp.float64(3.0)
            kk = kk * lf
        if catz[r] != 0:
            kk = kk * (wp.float64(1.0) + wp.float64(cat[r, i, j]))
        if kk <= wp.float64(0.0):
            continue
        prop = kk * dt_scale
        order = int(0)
        for q in range(MAX_SUB):
            s = sub_s[r, q]
            if s < 0:
                break
            c = sub_c[r, q]
            for _ in range(c):
                prop = prop * wp.float64(spec[s, i, j])
                order += 1
        if prop <= wp.float64(0.0):
            continue
        for _ in range(order - 1):
            prop = prop / norm
        st = wp.rand_init(seed, wp.int32((i * W + j) * n_react + r))
        lam = wp.float32(prop)
        if lam > wp.float32(1.0e6):
            lam = wp.float32(1.0e6)
        ev = int(wp.poisson(st, lam))
        if ev <= 0:
            continue
        # cap by availability (serial per cell -> deterministic, never negative)
        for q in range(MAX_SUB):
            s = sub_s[r, q]
            if s < 0:
                break
            avail = spec[s, i, j] // sub_c[r, q]
            if ev > avail:
                ev = avail
        if ev <= 0:
            continue
        for q in range(MAX_SUB):
            s = sub_s[r, q]
            if s < 0:
                break
            spec[s, i, j] = spec[s, i, j] - sub_c[r, q] * ev
        for q in range(MAX_SUB):
            s = prod_s[r, q]
            if s < 0:
                break
            spec[s, i, j] = spec[s, i, j] + prod_c[r, q] * ev
        heat = heat - wp.float64(dh[r]) * wp.float64(1.0e-3) * wp.float64(ev)
    if heat != wp.float64(0.0):
        c1 = wp.float64(6.5e5) + wp.float64(4.186e6) * water[i, j]  # real layer capacity
        temp[1, i, j] = temp[1, i, j] + heat / c1
        e_chem[i, j] = e_chem[i, j] + heat


class Chemistry:
    def __init__(self, doc: dict):
        self.doc = doc
        self.elements = doc["elements"]
        self.names = list(doc["species"].keys())
        self.index = {n: k for k, n in enumerate(self.names)}
        self.n_species = len(self.names)
        self.reactions = doc["reactions"]
        self.n_react = len(self.reactions)
        self.norm = float(doc.get("norm", 1e5))
        # element composition matrix (n_species x n_elements)
        self.comp = np.zeros((self.n_species, len(self.elements)), dtype=np.int64)
        for sname, sd in doc["species"].items():
            for el, cnt in sd["formula"].items():
                self.comp[self.index[sname], self.elements.index(el)] = cnt
        self.validate()
        self.state = None

    @classmethod
    def load(cls, path: str) -> "Chemistry":
        with open(path) as f:
            return cls(yaml.safe_load(f))

    def validate(self) -> None:
        sources = {n: 0 for n in self.names}
        sinks = {n: 0 for n in self.names}
        for r in self.reactions:
            lhs = np.zeros(len(self.elements), dtype=np.int64)
            rhs = np.zeros(len(self.elements), dtype=np.int64)
            for sp, c in r["sub"].items():
                lhs += self.comp[self.index[sp]] * c
                sinks[sp] += 1
            for sp, c in r["prod"].items():
                rhs += self.comp[self.index[sp]] * c
                sources[sp] += 1
            if not np.array_equal(lhs, rhs):
                raise ValueError(f"reaction {r['name']} does not balance: {lhs} != {rhs}")
        # polymerization/copying consumes M* and PP and produces Pi (polymers.py);
        # hydrolysis of chains returns M*. Count those as chemistry-external source/sink.
        for m in ("M1", "M2", "M3", "M4", "PP"):
            sinks[m] += 1
        for m in ("M1", "M2", "M3", "M4", "Pi"):
            sources[m] += 1
        sinks["L"] += 1   # membranes accrete L
        sources["L"] += 1  # membrane decay returns L
        bad = [n for n in self.names if sources[n] == 0 or sinks[n] == 0]
        if bad:
            raise ValueError(f"species without both source and sink: {bad}")

    def bind(self, state, cfg) -> None:
        self.state = state
        self.cfg = cfg
        self.key = mix(cfg.run.seed, SALT_CHEM)
        dev = state.device
        nr, ns = self.n_react, self.n_species
        h, w = state.shape
        sub_s = np.full((nr, MAX_SUB), -1, dtype=np.int32)
        sub_c = np.zeros((nr, MAX_SUB), dtype=np.int32)
        prod_s = np.full((nr, MAX_SUB), -1, dtype=np.int32)
        prod_c = np.zeros((nr, MAX_SUB), dtype=np.int32)
        for k, r in enumerate(self.reactions):
            for q, (sp, c) in enumerate(r["sub"].items()):
                sub_s[k, q], sub_c[k, q] = self.index[sp], c
            for q, (sp, c) in enumerate(r["prod"].items()):
                prod_s[k, q], prod_c[k, q] = self.index[sp], c
        arr = lambda a, dt: wp.array(a, dtype=dt, device=dev)  # noqa: E731
        self.g_sub_s, self.g_sub_c = arr(sub_s, wp.int32), arr(sub_c, wp.int32)
        self.g_prod_s, self.g_prod_c = arr(prod_s, wp.int32), arr(prod_c, wp.int32)
        self.g_k300 = arr(np.array([r["k300"] for r in self.reactions], np.float32), wp.float32)
        self.g_ea = arr(np.array([r["ea"] for r in self.reactions], np.float32), wp.float32)
        self.g_dh = arr(np.array([r["dh"] for r in self.reactions], np.float32), wp.float32)
        self.g_photo = arr(np.array([1 if r.get("photo") else 0 for r in self.reactions], np.int32), wp.int32)
        self.g_catz = arr(np.array([1 if r.get("catalyzable") else 0 for r in self.reactions], np.int32), wp.int32)
        state.catalyst = wp.zeros((nr, h, w), dtype=wp.float32, device=dev)
        state.e_chem = wp.zeros((h, w), dtype=wp.float64, device=dev)
        self.dt_scale = 1.0  # k300 is already per-tick

    def init_species(self, state) -> None:
        """Initial inventory: wet cells get solution chemistry, dry cells the atmosphere."""
        h, w = state.shape
        wet = state.water_depth.numpy() > 1e-3
        spec = np.zeros((self.n_species, h, w), dtype=np.int32)
        overrides = getattr(getattr(self, "cfg", None), "chemistry", None)
        overrides = overrides.overrides if overrides else {}
        for name, sd in self.doc["species"].items():
            k = self.index[name]
            sd = sd | overrides.get(name, {})     # M1-protocol environment tuning
            spec[k] = np.where(wet, sd.get("init_wet", 0), sd.get("init_dry", 0))
        state.species = wp.array(spec, dtype=wp.int32, device=state.device)
        log.info("species initialized", extra={"n_species": self.n_species, "wet_cells": int(wet.sum())})

    def step(self, state, tick: int) -> None:
        h, w = state.shape
        wp.launch(k_chemistry, dim=(h, w), inputs=[
            state.species, state.temp, state.light_water, state.catalyst, state.e_chem,
            state.water_depth,
            self.g_sub_s, self.g_sub_c, self.g_prod_s, self.g_prod_c,
            self.g_k300, self.g_ea, self.g_dh, self.g_photo, self.g_catz,
            self.n_react, wp.float64(self.norm),
            wp.int32(seed32(self.key, tick)), w, wp.float64(self.dt_scale)],
            device=state.device)
