"""Counter-based RNG. No global RNG anywhere.

GPU kernels call wp.rand_init(mix(seed, SALT_X), tick * n + index) inline.
Python-side draws (terrain, operator events) use numpy Philox keyed the same way,
so every random number is a pure function of (run_seed, salt, tick, index).
"""
from __future__ import annotations

import numpy as np

# Module salts — never reuse, never renumber (renumbering breaks replay).
SALT_TERRAIN = 1
SALT_WIND = 2
SALT_CHEM = 3
SALT_POLY_HYDRO = 4
SALT_POLY_COPY = 5
SALT_POLY_MOVE = 6
SALT_EVENT = 7
SALT_POLY_EFFECT = 8


def mix(seed: int, salt: int) -> int:
    """Stable 63-bit mix of run seed and module salt (splitmix64 finalizer)."""
    z = (seed + 0x9E3779B97F4A7C15 * (salt + 1)) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0x7FFFFFFFFFFFFFFF


def seed32(key: int, tick: int, phase: int = 0) -> int:
    """31-bit kernel seed, pure function of (key, tick, phase). Kernel offsets then
    only need to index within the launch (fits int32 at any world size)."""
    return mix(key ^ (tick * 0x9E3779B1), phase) & 0x7FFFFFFF


def np_rng(seed: int, salt: int, tick: int = 0) -> np.random.Generator:
    """Deterministic numpy generator keyed by (seed, salt, tick)."""
    return np.random.Generator(
        np.random.Philox(key=np.uint64(mix(seed, salt)), counter=[0, 0, 0, np.uint64(tick)]))
