"""Phase 3 acceptance: loader rejects bad networks; closed box conserves atoms to
the integer and settles toward equilibrium; P~P is made by light, stops at night,
and shows depth structure (Beer-Lambert)."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from tests.conftest import REPO, build_test_sim, tiny_cfg


def _load_doc():
    with open(REPO / "configs/chemistry_v0.yaml") as f:
        return yaml.safe_load(f)


class TestLoader:
    def test_valid_network_loads(self):
        from firmament.core.chemistry import Chemistry
        chem = Chemistry(_load_doc())
        assert chem.n_species >= 20 and chem.n_react >= 25

    def test_rejects_unbalanced_reaction(self):
        from firmament.core.chemistry import Chemistry
        doc = copy.deepcopy(_load_doc())
        doc["reactions"][2]["prod"] = {"Pi": 3}          # PP + H2O -> 3 Pi: P and O wrong
        with pytest.raises(ValueError, match="does not balance"):
            Chemistry(doc)

    def test_rejects_species_without_sink(self):
        from firmament.core.chemistry import Chemistry
        doc = copy.deepcopy(_load_doc())
        # drop every reaction consuming N2 (its only sink is n2_fixation)
        doc["reactions"] = [r for r in doc["reactions"] if "N2" not in r["sub"]]
        with pytest.raises(ValueError, match="without both source and sink"):
            Chemistry(doc)


def _sim(tmp_path, device, keep, **cfg_kw):
    cfg = tiny_cfg(cap=1000, **cfg_kw)
    state, sched = build_test_sim(cfg, device, tmp_path / "run")
    name_of = {0: "rad", 1: "thermal", 2: "fluid", 3: "chem", 4: "poly", 5: "audit"}
    sched.modules = [m for k, m in enumerate(sched.modules) if name_of[k] in keep]
    return cfg, state, sched


def test_closed_box_atoms_exact_and_equilibrating(tmp_path, device):
    """No light, no vents: every atom accounted to the integer; activity decays."""
    from firmament.core.audit import Audit
    from firmament.core.chemistry import Chemistry

    cfg, state, sched = _sim(tmp_path, device, keep=("chem",), solar=0.0, geo=0.0, vents=[])
    chem = Chemistry.load(str(REPO / "configs/chemistry_v0.yaml"))
    aud = Audit(cfg, chem, tmp_path)
    el0 = aud.element_totals(state)
    v_prev = state.species.numpy().astype(np.int64).sum(axis=(1, 2))
    activity = []
    for w in range(6):
        for _ in range(500):
            sched.tick_once()
        v = state.species.numpy().astype(np.int64).sum(axis=(1, 2))
        activity.append(int(np.abs(v - v_prev).sum()))
        v_prev = v
    el1 = aud.element_totals(state)
    assert np.array_equal(el0, el1), f"atom drift: {(el1 - el0).tolist()}"
    assert activity[0] > 0, "network completely inert — not a real chemistry"
    # closed systems relax on the slowest reaction timescale (M hydrolysis ~23 sim-days
    # here); at test horizon the honest check is steady monotonic decline of activity.
    for a, b in zip(activity, activity[1:]):
        assert b < a * 1.02, f"activity not declining: {activity}"
    assert activity[-1] < 0.55 * activity[0], f"too slow toward equilibrium: {activity}"
    # no photochemistry happened in the dark
    assert float(state.light_water.numpy().max()) == 0.0


def test_lit_box_pp_day_night_and_depth(tmp_path, device):
    """Photochemical P~P, isolated by a DARK CONTROL: an identical sim with the sun
    off. Light must drive daytime production far beyond the thermal path, the gain
    must vanish at night, and shallow water must out-produce deep (Beer-Lambert)."""
    def run_protocol(solar):
        cfg, state, sched = _sim(tmp_path / f"s{int(solar)}", device,
                                 keep=("rad", "thermal", "chem"), solar=solar, geo=0.1)
        idx = sched.chem.index["PP"]

        def pp():
            return int(state.species.numpy()[idx].astype(np.int64).sum())
        tpd = int(24 * 3600 / cfg.run.dt_seconds)      # t=0 is midnight
        while state.tick < tpd // 4:                   # 06:00 dawn
            sched.tick_once()
        dawn0 = pp()
        while state.tick < 3 * tpd // 4:               # 18:00 dusk
            sched.tick_once()
        dusk = pp()
        d = state.water_depth.numpy()
        ppmap = state.species.numpy()[idx].astype(np.float64)
        while state.tick < 5 * tpd // 4:               # 06:00 next dawn
            sched.tick_once()
        dawn1 = pp()
        return dusk - dawn0, dawn1 - dusk, d, ppmap

    day_lit, night_lit, d, ppmap = run_protocol(1361.0)
    day_dark, night_dark, _, _ = run_protocol(0.0)

    assert day_lit > 0
    # light drives production well beyond the geothermal path
    assert day_lit > 3 * max(day_dark, 1), f"lit day gain {day_lit} vs dark {day_dark}"
    # the photochemical gain disappears at night (net goes to ~zero or decay)
    assert night_lit < 0.15 * day_lit, f"night gain {night_lit} vs day gain {day_lit}"
    # Beer-Lambert depth structure at dusk: shallow cells out-produce deep ones
    wet = d > 0.05
    assert wet.sum() > 50
    r = float(np.corrcoef(d[wet], ppmap[wet])[0, 1])
    assert r < -0.2, f"no Beer-Lambert depth structure in P~P (corr={r:.2f})"
