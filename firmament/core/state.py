"""World state: dense per-cell fields + sparse polymer arrays, on one Warp device.

Precision (docs/DECISIONS.md): conserved continuous fields f64, diagnostics f32,
species counts i32 (exact conservation), sequences u8.
"""
from __future__ import annotations

import numpy as np
import warp as wp

# Polymer `state` codes
P_DEAD, P_FREE, P_BOUND, P_COPYING, P_MEMBRANE = 0, 1, 2, 3, 4


class State:
    def __init__(self, cfg, n_species: int, device: str = "cuda:0") -> None:
        self.device = device
        self.cfg = cfg
        h, w = cfg.world.size
        self.shape = (h, w)
        self.n_species = n_species
        f64, f32, i32 = wp.float64, wp.float32, wp.int32
        z = lambda dtype, shape: wp.zeros(shape, dtype=dtype, device=device)  # noqa: E731

        # --- fields (gather-only kernels write these) ---
        self.elevation = z(f64, (h, w))
        self.sediment = z(f64, (h, w))
        self.water_depth = z(f64, (h, w))
        self.water_u = z(f64, (h, w))       # x-velocity (m/s), cell-centered
        self.water_v = z(f64, (h, w))
        self.vapor = z(f64, (h, w))          # kg/m^2 column water vapor
        self.wind_u = z(f32, (h, w))         # prescribed + perturbed, not prognostic
        self.wind_v = z(f32, (h, w))
        self.temp = z(f64, (3, h, w))        # K: layer 0=air, 1=surface/water, 2=sediment
        self.light = z(f32, (h, w))          # W/m^2 at surface
        self.light_water = z(f32, (h, w))    # W/m^2 at bottom of water column
        self.species = z(i32, (n_species, h, w))
        self.compartment_id = z(i32, (h, w))
        self.catalyst = None                 # (n_reactions, h, w) f32 — allocated by chemistry
        self.vent_flux = z(f32, (h, w))      # W/m^2 geothermal, set by terrain
        self.albedo_dust = z(f32, (h, w))    # meteor/volcano dust, raises albedo, decays
        self.solar_mult = 1.0                # ice age / warm period / solar events

        # --- polymers (sparse; free-list allocation) ---
        p = cfg.polymers.capacity
        self.p_cap = p
        self.l_max = cfg.polymers.max_length
        self.p_id = z(wp.int64, p)
        self.p_parent = z(wp.int64, p)
        self.p_cell = z(i32, p)              # flattened y*w+x
        self.p_seq = z(wp.uint8, (p, self.l_max))
        self.p_len = z(i32, p)
        self.p_state = z(wp.uint8, p)        # P_DEAD=slot free
        self.p_partner = z(i32, p)
        self.p_energy = z(f32, p)
        self.p_born = z(wp.int64, p)
        self.p_copy_pos = z(i32, p)          # progress of an in-flight copy
        self.p_child = z(i32, p)             # slot of the copy being built, or -1
        self.p_motifs = z(i32, p)            # cached motif bitmask (set at birth)
        self.p_count = 0                     # live polymers (host-side mirror)
        self.next_poly_id = 1                # 0 reserved; seed gets id 1

        self.tick = 0

    # ---- host access helpers ----
    def field_names(self) -> list[str]:
        return [k for k, v in vars(self).items() if isinstance(v, wp.array)]

    def to_numpy(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k).numpy() for k in self.field_names()}

    def load_numpy(self, arrays: dict[str, np.ndarray]) -> None:
        for k, arr in arrays.items():
            dst = getattr(self, k)
            wp.copy(dst, wp.array(np.ascontiguousarray(arr), dtype=dst.dtype, device=self.device))

    def nbytes(self) -> int:
        return sum(getattr(self, k).capacity for k in self.field_names())
