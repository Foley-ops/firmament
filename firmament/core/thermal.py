"""Thermal: vertical exchange between layers, lateral conduction. Pure gather, f64.

Internal transfers only — no e_in/e_out. Two-pass (flux from old temps into new array)
keeps the pairwise symmetry exact: cell A's flux to B equals minus B's flux to A.
"""
from __future__ import annotations

import warp as wp

from firmament.core import physconst as pc


@wp.kernel
def k_thermal(t_old: wp.array3d(dtype=wp.float64), t_new: wp.array3d(dtype=wp.float64),
              water: wp.array2d(dtype=wp.float64), dt: wp.float64, h: int, w: int):
    i, j = wp.tid()
    t0 = t_old[0, i, j]
    t1 = t_old[1, i, j]
    t2 = t_old[2, i, j]
    c1 = wp.float64(pc.C_SURF_DRY) + wp.float64(pc.CW_VOL) * water[i, j]

    f_as = wp.float64(pc.H_AIR_SURF) * (t0 - t1)   # air -> surface
    f_ss = wp.float64(pc.H_SURF_SED) * (t1 - t2)   # surface -> sediment

    lat = wp.float64(0.0)                           # lateral surface conduction (gather)
    for k in range(4):
        di = 0
        dj = 0
        if k == 0:
            di = -1
        elif k == 1:
            di = 1
        elif k == 2:
            dj = -1
        else:
            dj = 1
        ni = i + di
        nj = j + dj
        if 0 <= ni and ni < h and 0 <= nj and nj < w:
            lat = lat + wp.float64(pc.K_HORIZ) * (t_old[1, ni, nj] - t1)

    t_new[0, i, j] = t0 + dt * (-f_as) / wp.float64(pc.C_AIR)
    t_new[1, i, j] = t1 + dt * (f_as - f_ss + lat) / c1
    t_new[2, i, j] = t2 + dt * f_ss / wp.float64(pc.C_SED)


class Thermal:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.tmp = None

    def step(self, state, tick: int) -> None:
        if self.tmp is None:
            self.tmp = wp.zeros_like(state.temp)
        wp.launch(k_thermal, dim=state.shape, inputs=[
            state.temp, self.tmp, state.water_depth,
            wp.float64(self.cfg.run.dt_seconds), state.shape[0], state.shape[1]],
            device=state.device)
        state.temp, self.tmp = self.tmp, state.temp
