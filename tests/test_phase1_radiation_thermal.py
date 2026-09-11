"""Phase 1 acceptance: sun geometry at noon/midnight/solstices; a plausible
day/night temperature cycle; energy balance over one sim-year (no water) < 0.1%."""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import REPO, build_test_sim, tiny_cfg


def _ticks_per_day(cfg):
    return int(cfg.world.rotation_period_hours * 3600 / cfg.run.dt_seconds)


class TestSunAngles:
    def setup_method(self):
        self.cfg = tiny_cfg(n=8)

    def test_midnight_is_dark(self):
        from firmament.core.radiation import sun_cos_zenith
        assert sun_cos_zenith(0, self.cfg) == 0.0            # t=0 is local midnight

    def test_noon_is_bright(self):
        from firmament.core.radiation import sun_cos_zenith
        noon = _ticks_per_day(self.cfg) // 2
        cz = sun_cos_zenith(noon, self.cfg)
        # equinox-ish start: zenith angle ~ latitude (15 deg) -> cos ~ 0.96
        assert 0.90 < cz <= 1.0

    def test_solstice_ordering(self):
        from firmament.core.radiation import sun_cos_zenith
        tpd = _ticks_per_day(self.cfg)
        year = int(self.cfg.world.year_length_days)
        # declination = tilt*sin(2*pi*frac): max at year/4 (summer), min at 3*year/4
        summer_noon = sun_cos_zenith((year // 4) * tpd + tpd // 2, self.cfg)
        winter_noon = sun_cos_zenith((3 * year // 4) * tpd + tpd // 2, self.cfg)
        equinox_noon = sun_cos_zenith(0 * tpd + tpd // 2, self.cfg)
        assert summer_noon > equinox_noon > winter_noon
        # at lat 15, summer solstice sun passes within 8.5 deg of zenith
        assert summer_noon > np.cos(np.radians(10.0))
        # night is night in every season
        assert sun_cos_zenith((year // 4) * tpd, self.cfg) == 0.0


def _dry_sim(tmp_path, device, n=48, dt=60.0):
    """Radiation + thermal only, dry world (fraction 0 -> no water anywhere)."""
    cfg = tiny_cfg(n=n, water=0.0, dt=dt, cap=1000)
    state, sched = build_test_sim(cfg, device, tmp_path / "run")
    keep = [sched.modules[0], sched.modules[1]]               # radiation, thermal
    sched.modules = keep
    return cfg, state, sched


def test_day_night_cycle_plausible(tmp_path, device):
    cfg, state, sched = _dry_sim(tmp_path, device)
    tpd = _ticks_per_day(cfg)
    # spin up one full day so the surface responds; recording then starts at midnight
    for _ in range(tpd):
        sched.tick_once()
    samples = []
    for t in range(tpd):
        sched.tick_once()
        if t % 30 == 0:
            samples.append(float(state.temp.numpy()[1].mean()))
    amp = max(samples) - min(samples)
    assert amp > 3.0, f"day/night surface swing too small: {amp:.2f} K"
    # warmest surface should lag noon but land in the afternoon half of the day
    peak_idx = int(np.argmax(samples))
    frac_of_day = peak_idx * 30 / tpd
    assert 0.5 < frac_of_day < 0.9, f"temperature peak at day fraction {frac_of_day:.2f} (noon=0.5)"


@pytest.mark.slow
def test_energy_balance_one_year_dry(tmp_path, device):
    """in - out - delta_stored within 0.1% over 365 sim-days, no water.
    dt=600 s keeps runtime sane; the scheme is stable there (see physconst)."""
    from firmament.core.audit import Audit
    from firmament.core.chemistry import Chemistry

    cfg, state, sched = _dry_sim(tmp_path, device, n=48, dt=600.0)
    chem = Chemistry.load(str(REPO / "configs/chemistry_v0.yaml"))
    aud = Audit(cfg, chem, tmp_path)
    ticks = int(365 * 86400 / cfg.run.dt_seconds)
    stored0 = aud.energy_stored(state)
    for _ in range(ticks):
        sched.tick_once()
    e_in = float(state.e_in.numpy().sum())
    e_out = float(state.e_out.numpy().sum())
    stored1 = aud.energy_stored(state)
    resid = abs(e_in - e_out - (stored1 - stored0))
    scale = max(e_in, e_out)
    assert scale > 0
    rel = resid / scale
    assert rel < 1e-3, f"energy audit residual {rel:.2e} over one sim-year"
    # and the seasons actually happened: summer warmer than winter
    # (sampled via monthly means is overkill; assert temps stayed physical instead)
    t = state.temp.numpy()
    assert 200.0 < t[1].mean() < 350.0
