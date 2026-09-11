"""Radiation: sun geometry, insolation, Beer–Lambert through water, longwave loss.

Two-stream grey atmosphere: surface emits sigma*T1^4; air absorbs fraction g and
re-radiates g*sigma*T0^4 both up and down. E_in/E_out accumulators (J/m^2, f64,
per cell, no atomics) make the energy audit exact by construction.
"""
from __future__ import annotations

import math

import warp as wp

from firmament.core import physconst as pc


@wp.kernel
def k_radiation(temp: wp.array3d(dtype=wp.float64), water: wp.array2d(dtype=wp.float64),
                vent: wp.array2d(dtype=wp.float32), light: wp.array2d(dtype=wp.float32),
                light_water: wp.array2d(dtype=wp.float32), e_in: wp.array2d(dtype=wp.float64),
                e_out: wp.array2d(dtype=wp.float64), dust: wp.array2d(dtype=wp.float32),
                cosz: wp.float64, S: wp.float64,
                dt: wp.float64):
    i, j = wp.tid()
    d = water[i, j]
    wet = d > wp.float64(0.001)
    alb = wp.float64(pc.ALBEDO_ROCK)
    if wet:
        alb = wp.float64(pc.ALBEDO_WATER)
    du = wp.float64(dust[i, j])
    alb = alb + du * (wp.float64(0.6) - alb)      # ash brightens the surface
    dust[i, j] = wp.float32(du * wp.float64(0.99999))  # dust settles (~ sim-week)
    sw = S * cosz * (wp.float64(1.0) - alb)
    light[i, j] = wp.float32(sw)
    light_water[i, j] = wp.float32(sw * wp.exp(-wp.float64(pc.K_WATER_LIGHT) * d))

    g = wp.float64(pc.GREENHOUSE)
    sig = wp.float64(pc.SIGMA)
    t0 = temp[0, i, j]
    t1 = temp[1, i, j]
    up_surf = sig * t1 * t1 * t1 * t1
    up_air = sig * t0 * t0 * t0 * t0
    geo = wp.float64(vent[i, j])

    c1 = wp.float64(pc.C_SURF_DRY) + wp.float64(pc.CW_VOL) * d
    # surface: +SW +geothermal(arrives via sediment? no: vent heat enters sediment) ... vent → layer2
    temp[1, i, j] = t1 + dt * (sw + g * up_air - up_surf) / c1
    temp[0, i, j] = t0 + dt * (g * up_surf - wp.float64(2.0) * g * up_air) / wp.float64(pc.C_AIR)
    temp[2, i, j] = temp[2, i, j] + dt * geo / wp.float64(pc.C_SED)
    e_in[i, j] = e_in[i, j] + dt * (sw + geo)
    e_out[i, j] = e_out[i, j] + dt * ((wp.float64(1.0) - g) * up_surf + g * up_air)


def sun_cos_zenith(tick: int, cfg) -> float:
    """cos of solar zenith angle; uniform over the ~1 km patch. Clamped at 0 (night)."""
    w = cfg.world
    t = tick * cfg.run.dt_seconds
    day_s = w.rotation_period_hours * 3600.0
    year_s = w.year_length_days * day_s
    hour_angle = 2.0 * math.pi * ((t / day_s) % 1.0) - math.pi     # 0 at local noon
    decl = math.radians(w.axial_tilt_deg) * math.sin(2.0 * math.pi * ((t / year_s) % 1.0))
    lat = math.radians(getattr(w, "latitude_deg", 15.0))
    cosz = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(hour_angle)
    return max(0.0, cosz)


class Radiation:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.bound = False

    def _bind(self, state) -> None:
        h, w = state.shape
        state.e_in = wp.zeros((h, w), dtype=wp.float64, device=state.device)
        state.e_out = wp.zeros((h, w), dtype=wp.float64, device=state.device)
        self.bound = True

    def step(self, state, tick: int) -> None:
        if not self.bound:
            self._bind(state)
        cosz = sun_cos_zenith(tick, self.cfg)
        wp.launch(k_radiation, dim=state.shape, inputs=[
            state.temp, state.water_depth, state.vent_flux, state.light, state.light_water,
            state.e_in, state.e_out, state.albedo_dust, wp.float64(cosz),
            wp.float64(self.cfg.world.solar_constant_wm2 * state.solar_mult),
            wp.float64(self.cfg.run.dt_seconds)],
            device=state.device)
