"""Phase 2 acceptance: water conservation to machine precision; slope release ends
in the basin; rain occurs; no cell ever holds negative depth."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from tests.conftest import build_test_sim, tiny_cfg


def _water_sim(tmp_path, device, n=32, dt=60.0, water=0.4):
    """Radiation + thermal + fluid (no chemistry/polymers: water physics only)."""
    cfg = tiny_cfg(n=n, water=water, dt=dt, cap=1000)
    state, sched = build_test_sim(cfg, device, tmp_path / "run")
    sched.modules = sched.modules[:3]                 # radiation, thermal, fluid
    return cfg, state, sched


def _total_water_kg(state) -> float:
    return float(state.water_depth.numpy().sum()) * 1000.0 + float(state.vapor.numpy().sum())


@pytest.mark.slow
def test_water_conserved_machine_precision(tmp_path, device):
    """Liquid + vapor total is invariant under evaporation, rain, flow, mixing.
    5 sim-days of full water cycle; drift must stay at f64 rounding level."""
    cfg, state, sched = _water_sim(tmp_path, device)
    total0 = _total_water_kg(state)
    ticks = int(5 * 86400 / cfg.run.dt_seconds)
    for t in range(ticks):
        sched.tick_once()
        if t % 1440 == 0:
            d = state.water_depth.numpy()
            assert d.min() >= 0.0, f"negative depth {d.min()} at tick {t}"
    total1 = _total_water_kg(state)
    rel = abs(total1 - total0) / total0
    assert rel < 1e-9, f"water drift {rel:.2e} over {ticks} ticks"


def test_slope_release_ends_in_basin(tmp_path, device):
    """A water blob released on a uniform slope must end up in the basin at the foot."""
    cfg, state, sched = _water_sim(tmp_path, device, n=48, water=0.0)
    n = 48
    # uniform slope descending toward the last rows, ending in a flat basin
    elev = np.zeros((n, n))
    for i in range(n):
        elev[i, :] = max(0.0, (n - 12 - i)) * 0.5     # 0.5 m per cell drop; basin: last 12 rows
    state.elevation = wp.array(elev, dtype=wp.float64, device=state.device)
    state.sediment = wp.array(np.zeros((n, n)), dtype=wp.float64, device=state.device)
    wd = np.zeros((n, n))
    wd[2:6, 20:28] = 1.0                              # blob near the top
    state.water_depth = wp.array(wd, dtype=wp.float64, device=state.device)
    state.vapor = wp.array(np.zeros((n, n)), dtype=wp.float64, device=state.device)
    # keep it cold so evaporation doesn't confuse the mass account
    state.temp = wp.array(np.full((3, n, n), 275.0), dtype=wp.float64, device=state.device)

    for _ in range(4000):
        sched.tick_once()
    d = state.water_depth.numpy()
    assert d.min() >= 0.0
    liquid = d.sum()
    in_basin = d[n - 12:, :].sum()
    assert liquid > 0
    assert in_basin / liquid > 0.9, f"only {in_basin / liquid:.1%} of water reached the basin"


def test_rain_occurs(tmp_path, device):
    """Evaporation loads the air; supersaturation must eventually rain back out."""
    cfg, state, sched = _water_sim(tmp_path, device, n=32)
    # warm surface -> vigorous evaporation; cool air -> low saturation ceiling
    t = state.temp.numpy()
    t[0] = 285.0      # mild air: saturation ceiling ~18 kg/m^2 (Tetens)
    t[1] = 305.0      # warm pools: vigorous evaporation
    state.temp = wp.array(t, dtype=wp.float64, device=state.device)
    # humid (but sub-saturated) evening airmass: evaporation must push it over the edge
    state.vapor = wp.array(np.full(state.shape, 15.0), dtype=wp.float64, device=state.device)
    rained = False
    prev_vapor = float(state.vapor.numpy().sum())
    peak_vapor = prev_vapor
    for _ in range(600):
        for _ in range(10):
            sched.tick_once()
        v = float(state.vapor.numpy().sum())
        peak_vapor = max(peak_vapor, v)
        if v < prev_vapor - 1e-9:                     # vapor fell: condensation happened
            rained = True
            break
        prev_vapor = v
    assert peak_vapor > 0, "no evaporation happened"
    assert rained, "no rain within 6000 warm ticks (100 sim-hours)"
    assert state.water_depth.numpy().min() >= 0.0


def test_substep_count_logged_on_change(tmp_path, device, caplog):
    """The guide requires logging every sub-step count change."""
    import logging

    cfg, state, sched = _water_sim(tmp_path, device)
    with caplog.at_level(logging.INFO, logger="firmament.fluid"):
        for _ in range(5):
            sched.tick_once()
    msgs = [r.message for r in caplog.records]
    assert any("substep count changed" in m for m in msgs)


def test_solubility_precipitation_and_redissolution(tmp_path, device):
    """Dissolved counts above the solubility cap precipitate to the immobile store
    (evaporite analog) and redissolve when under-saturated; totals exact."""
    from firmament.core.fluid import SAT_CAP

    cfg, state, sched = _water_sim(tmp_path, device, n=16)
    idx = sched.chem.index["Pi"]
    spec = state.species.numpy()
    wet = np.argwhere(state.water_depth.numpy() > 0.1)
    y, x = (int(v) for v in wet[0])
    spec[idx, y, x] = SAT_CAP * 3   # far past saturation: one tick of advection cannot rescue it
    state.species = wp.array(spec, dtype=wp.int32, device=state.device)
    total0 = int(spec[idx].astype(np.int64).sum()) + int(state.precipitate.numpy()[idx].astype(np.int64).sum())
    sched.tick_once()
    sp = state.species.numpy()[idx]
    pr = state.precipitate.numpy()[idx]
    assert sp[y, x] <= SAT_CAP
    assert pr[y, x] > 0                                  # excess precipitated
    assert int(sp.astype(np.int64).sum()) + int(pr.astype(np.int64).sum()) == total0
    # drain the dissolved phase; the mineral must start redissolving
    sp2 = state.species.numpy()
    sp2[idx, y, x] = 0
    state.species = wp.array(sp2, dtype=wp.int32, device=state.device)
    before = int(state.precipitate.numpy()[idx, y, x])
    for _ in range(5):
        sched.tick_once()
    assert int(state.precipitate.numpy()[idx, y, x]) < before
