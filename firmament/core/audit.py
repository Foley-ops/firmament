"""Conservation audit: exact element totals (integer), energy and water balance.

A violation pauses the sim and raises — hard error, never a warning (non-negotiable #3).
Operator events that inject matter/energy/water register here so the books stay closed.
"""
from __future__ import annotations

import numpy as np

from firmament.core import physconst as pc
from firmament.io.logging import get

log = get("audit")

CHAIN_H2O = None  # chain unit = M - H2O, computed from chemistry at bind


class AuditError(RuntimeError):
    pass


class Audit:
    def __init__(self, cfg, chem, run_dir) -> None:
        self.cfg = cfg
        self.chem = chem
        self.every = cfg.run.audit_every_ticks
        self.baseline_elements = None
        self.baseline_water = None
        self.baseline_energy = None
        self.injected_elements = np.zeros(len(chem.elements), dtype=np.int64)
        self.injected_water = 0.0
        self.injected_energy = 0.0
        # chain unit composition: M minus H2O (condensation releases water on synthesis)
        m = chem.comp[chem.index["M1"]].astype(np.int64)
        w = chem.comp[chem.index["H2O"]].astype(np.int64)
        self.unit_comp = m - w
        self.l_comp = chem.comp[chem.index["L"]].astype(np.int64)

    # ---- ledgers used by operator events ----
    def register_injection(self, element_counts: np.ndarray = None, water: float = 0.0,
                           energy: float = 0.0) -> None:
        if element_counts is not None:
            self.injected_elements += element_counts
        self.injected_water += water
        self.injected_energy += energy

    # ---- totals ----
    def element_totals(self, s) -> np.ndarray:
        spec = s.species.numpy().astype(np.int64)
        if hasattr(s, "precipitate"):
            spec = spec + s.precipitate.numpy().astype(np.int64)
        # tripwire: dissolved concentration approaching int32 wrap = unmodeled
        # solubility. Pause loudly rather than ever wrapping silently.
        peak = int(spec.max())
        if peak > 1_600_000_000:
            k = int(np.unravel_index(spec.argmax(), spec.shape)[0])
            raise AuditError(
                f"species '{self.chem.names[k]}' count {peak} nears the int32 limit — "
                "dissolved concentration exceeded modeled solubility (model gap)")
        tot = (spec.sum(axis=(1, 2))[:, None] * self.chem.comp.astype(np.int64)).sum(axis=0)
        alive = s.p_state.numpy() > 0
        n_units = int(s.p_len.numpy()[alive].sum())
        tot = tot + n_units * self.unit_comp
        if hasattr(s, "membrane_store"):
            tot = tot + int(s.membrane_store.numpy().sum()) * self.l_comp
        return tot

    def water_total(self, s) -> float:
        return float(s.water_depth.numpy().sum()) * pc.RHO_W + float(s.vapor.numpy().sum())

    def energy_stored(self, s) -> float:
        t = s.temp.numpy()
        d = s.water_depth.numpy()
        c1 = pc.C_SURF_DRY + pc.CW_VOL * d
        stored = (pc.C_AIR * t[0] + c1 * t[1] + pc.C_SED * t[2]).sum()
        # vapor carries fixed enthalpy LV + CW_SP*T_REF per kg (see fluid.k_evap_rain)
        vap_enthalpy = float(s.vapor.numpy().sum()) * (pc.LV + pc.CW_SP * pc.T_REF)
        return float(stored) + vap_enthalpy

    def step(self, s, tick: int) -> None:
        if tick % self.every != 0:
            return
        el = self.element_totals(s) - self.injected_elements
        water = self.water_total(s) - self.injected_water
        e_in = float(s.e_in.numpy().sum()) + self.injected_energy
        e_out = float(s.e_out.numpy().sum())
        e_chem = float(s.e_chem.numpy().sum()) if hasattr(s, "e_chem") else 0.0
        stored = self.energy_stored(s)
        if self.baseline_elements is None:
            self.baseline_elements = el.copy()
            self.baseline_water = water
            self.baseline_energy = stored - (e_in - e_out + e_chem)
            log.info("audit baseline", extra={"elements": el.tolist(),
                     "water_kg": water, "stored_J": stored})
            return
        # elements: exact to the integer
        if not np.array_equal(el, self.baseline_elements):
            diff = (el - self.baseline_elements).tolist()
            log.error("ELEMENT AUDIT VIOLATION", extra={"diff": diff})
            raise AuditError(f"element conservation violated at tick {tick}: {diff}")
        # water: machine precision (f64 pairwise transfers)
        wdrift = abs(water - self.baseline_water) / max(abs(self.baseline_water), 1.0)
        if wdrift > 1e-9:
            log.error("WATER AUDIT VIOLATION", extra={"rel_drift": wdrift})
            raise AuditError(f"water conservation violated at tick {tick}: rel {wdrift:.2e}")
        # energy: in - out + chem == delta stored, within 0.1% of throughput
        expect = self.baseline_energy + e_in - e_out + e_chem
        scale = max(abs(e_in), abs(e_out), abs(stored - self.baseline_energy), 1.0)
        edrift = abs(stored - expect) / scale
        if edrift > 1e-3:
            log.error("ENERGY AUDIT VIOLATION", extra={"rel_drift": edrift,
                      "stored": stored, "expected": expect})
            raise AuditError(f"energy conservation violated at tick {tick}: rel {edrift:.2e}")
        log.info("audit ok", extra={"tick": tick, "water_rel": f"{wdrift:.2e}",
                 "energy_rel": f"{edrift:.2e}"})
