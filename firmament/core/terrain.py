"""Terrain generation: fractal value noise -> elevation, basins, vents, sediment, initial water.

Host-side numpy, fully determined by cfg.world.terrain_seed. Runs once at world creation.
Real analog: tectonic + erosional topography, approximated by self-similar noise.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from firmament.core.rng import SALT_TERRAIN, np_rng
from firmament.io.logging import get

log = get("terrain")

T_REF = 288.0  # K, initial isothermal state


def _fractal_noise(rng: np.random.Generator, h: int, w: int, octaves: int = 6) -> np.ndarray:
    out = np.zeros((h, w))
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = 2 ** (o + 2)
        grid = rng.standard_normal((n + 1, n + 1))
        yi = np.linspace(0, n, h, endpoint=False)
        xi = np.linspace(0, n, w, endpoint=False)
        y0, x0 = yi.astype(int), xi.astype(int)
        fy, fx = (yi - y0)[:, None], (xi - x0)[None, :]
        fy2, fx2 = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)  # smoothstep
        g = (grid[y0][:, x0] * (1 - fy2) * (1 - fx2) + grid[y0 + 1][:, x0] * fy2 * (1 - fx2)
             + grid[y0][:, x0 + 1] * (1 - fy2) * fx2 + grid[y0 + 1][:, x0 + 1] * fy2 * fx2)
        out += amp * g
        total += amp
        amp *= 0.5
    return out / total


def generate(state, cfg) -> None:
    h, w = state.shape
    rng = np_rng(cfg.world.terrain_seed, SALT_TERRAIN)
    elev = _fractal_noise(rng, h, w) * 8.0          # ~±10 m relief over the patch
    elev -= elev.min()

    # Carve a shallow basin around each vent (vents sit in warm pools, per the writeup).
    yy, xx = np.mgrid[0:h, 0:w]
    for vx, vy in cfg.world.vents:
        r2 = (yy - vy) ** 2 + (xx - vx) ** 2
        elev -= 3.0 * np.exp(-r2 / (2 * (h / 16) ** 2))
    elev -= elev.min()

    sediment = np.full((h, w), 0.5)
    # Fill the lowest initial_water_fraction of the surface with water.
    level = np.quantile(elev, cfg.world.initial_water_fraction)
    water = np.maximum(0.0, level - elev)

    vent_flux = np.full((h, w), cfg.world.geothermal_flux_wm2, dtype=np.float32)
    for vx, vy in cfg.world.vents:
        r2 = (yy - vy) ** 2 + (xx - vx) ** 2
        vent_flux += 2000.0 * np.exp(-r2 / (2 * 3.0**2))  # ~2 kW/m^2 at vent core

    dev = state.device
    state.elevation = wp.array(elev, dtype=wp.float64, device=dev)
    state.sediment = wp.array(sediment, dtype=wp.float64, device=dev)
    state.water_depth = wp.array(water, dtype=wp.float64, device=dev)
    state.vent_flux = wp.array(vent_flux, dtype=wp.float32, device=dev)
    state.temp = wp.array(np.full((3, h, w), T_REF), dtype=wp.float64, device=dev)
    log.info("terrain generated", extra={
        "elev_range_m": [round(float(elev.min()), 2), round(float(elev.max()), 2)],
        "wet_fraction": round(float((water > 0).mean()), 3),
        "water_volume_m3": round(float(water.sum()), 1)})
