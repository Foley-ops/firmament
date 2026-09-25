"""Phase 6 acceptance: every milestone detector fires on a synthetic fixture and
stays silent on a null fixture; instruments running vs disabled change the sim
state by ZERO bytes; a seeded run produces metrics parquet, a lineage DB, and the
first_copy milestone in events.jsonl."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


class FakeEvents:
    def __init__(self):
        self.records = []

    def append(self, event, tick, component="x", **fields):
        self.records.append({"event": event, "tick": tick, "component": component, **fields})
        return self.records[-1]


class FakeLineage:
    def __init__(self, depth=150):
        self._depth = depth

    def depth_of(self, pid):
        return self._depth


def make_view(n=8, cap=32, **over):
    """Synthetic read-only view shaped like sampler.ReadOnlyView."""
    v = {
        "tick": over.pop("tick", 1000),
        "shape": (n, n),
        "water_depth": np.full((n, n), 0.5),
        "vapor": np.zeros((n, n)),
        "temp": np.full((3, n, n), 288.0),
        "light": np.zeros((n, n), dtype=np.float32),
        "light_water": np.zeros((n, n), dtype=np.float32),
        "species": np.zeros((22, n, n), dtype=np.int32),
        "compartment_id": np.zeros((n, n), dtype=np.int32),
        "membrane_store": np.zeros((n, n), dtype=np.int32),
        "p_id": np.zeros(cap, dtype=np.int64),
        "p_parent": np.zeros(cap, dtype=np.int64),
        "p_cell": np.zeros(cap, dtype=np.int32),
        "p_len": np.zeros(cap, dtype=np.int32),
        "p_state": np.zeros(cap, dtype=np.uint8),
        "p_partner": np.full(cap, -1, dtype=np.int32),
        "p_motifs": np.zeros(cap, dtype=np.int32),
        "p_mut": np.zeros(cap, dtype=np.int32),
        "p_born": np.zeros(cap, dtype=np.int64),
        "p_seq": np.zeros((cap, 64), dtype=np.uint8),
        "recent_copies": [],
    }
    v.update(over)
    return v


BINDER = 1 << 4          # genetic_code_v0: bit 4 = binder
REPL = 1


def fake_sched():
    names = ["replicase", "catalyst_formose", "catalyst_m1", "catalyst_lipid", "binder",
             "membrane", "photoactive", "motor", "emit_A", "sense_A"]
    return SimpleNamespace(analysis=FakeEvents(), lineage=FakeLineage(),
                           chem=SimpleNamespace(index={"A": 18}),
                           poly=SimpleNamespace(motif_names=names))


def fired(sched):
    return {r["event"] for r in sched.analysis.records}


def _pop_row(v):
    from firmament.instruments import population
    return population.sample(v)


def _living(cap=200, n_alive=120, seq_val=1, state=1):
    v = make_view(cap=cap)
    v["p_state"][:n_alive] = state
    v["p_id"][:n_alive] = np.arange(2, n_alive + 2)
    v["p_parent"][:n_alive] = 1
    v["p_len"][:n_alive] = 30
    v["p_seq"][:n_alive, :30] = seq_val
    return v


def _check(m, v):
    m.check(v, _pop_row(v))


class TestMilestoneDetectors:
    def test_null_fixture_fires_nothing(self):
        from firmament.instruments.milestones import Milestones
        s = fake_sched()
        m = Milestones(s)
        for _ in range(6):
            _check(m, make_view())
        assert fired(s) == set()

    def test_first_copy(self):
        from firmament.instruments.milestones import Milestones
        s = fake_sched()
        m = Milestones(s)
        _check(m, make_view(recent_copies=[(2, 1, 10, 30, 0)]))
        assert "first_copy" in fired(s)

    def test_fixation_requires_a_sweep_from_minority(self):
        """A dominant ancestor is NOT fixation. A variant first seen at <=10% that then
        exceeds 90% is (the genesis v0.1 detector fired on a 95% unmutated population)."""
        from firmament.instruments.milestones import Milestones
        s = fake_sched()
        m = Milestones(s)
        v = _living()
        v["p_seq"][:6, :30] = 3               # variant at 5%
        _check(m, v)
        _check(m, _living())                  # ancestor 100%: not a sweep
        assert "candidate_fixation" not in fired(s)
        swept = _living(seq_val=3)            # the minority variant now at 100%
        _check(m, swept)
        assert "candidate_fixation" in fired(s)

    def test_fixation_ignores_children_under_construction(self):
        from firmament.instruments.milestones import Milestones
        s = fake_sched()
        m = Milestones(s)
        v = _living(n_alive=10)
        v["p_state"][10:190] = 5              # 180 empty in-progress children
        _check(m, v)
        assert fired(s) == set()

    def test_mutant_depth100(self):
        from firmament.instruments.milestones import Milestones
        s = fake_sched()
        m = Milestones(s)
        v = _living()
        v["p_mut"][:5] = 2
        _check(m, v)
        assert "candidate_mutant_depth100" in fired(s)

    def test_compartment_advantage_needs_persistence(self):
        from firmament.instruments.milestones import Milestones
        s = fake_sched()
        m = Milestones(s)
        v = _living()
        v["compartment_id"][2, 3] = 1
        v["p_cell"][:80] = 2 * 8 + 3
        for _ in range(4):
            _check(m, v)
        assert "first_compartment" in fired(s)
        assert "candidate_compartment_advantage" not in fired(s)
        _check(m, v)                          # 5th consecutive sample
        assert "candidate_compartment_advantage" in fired(s)

    def _binder_pair(self, v, a=0, b=1):
        v["p_partner"][a], v["p_partner"][b] = b, a
        v["p_motifs"][a] |= BINDER
        v["p_motifs"][b] |= BINDER
        return v

    def test_copier_template_pairs_are_not_interactions(self):
        """The v0.1 bug: p_partner is also the copier<->template link. A COPYING/BOUND
        pair with different parents, whose template then dies, must fire nothing."""
        from firmament.instruments.milestones import Milestones
        s = fake_sched()
        m = Milestones(s)
        v = _living()
        v["p_partner"][0], v["p_partner"][1] = 1, 0
        v["p_state"][0], v["p_state"][1] = 3, 2            # copying / bound
        v["p_parent"][1] = 99
        v["p_seq"][1, :30] = 2
        for _ in range(3):
            _check(m, v)
        v2 = _living()
        v2["p_partner"][0] = 1
        v2["p_state"][0], v2["p_state"][1] = 3, 0          # template died
        _check(m, v2)
        assert not ({"candidate_predation", "candidate_aggregate"} & fired(s))

    def test_binder_predation(self):
        from firmament.instruments.milestones import Milestones
        s = fake_sched()
        m = Milestones(s)
        _check(m, self._binder_pair(_living()))
        v2 = _living()
        v2["p_state"][1] = 0                               # partner died
        _check(m, v2)
        assert "candidate_predation" in fired(s)

    def test_binder_aggregate_of_distinct_lineages_needs_persistence(self):
        from firmament.instruments.milestones import Milestones
        s = fake_sched()
        m = Milestones(s)
        v = self._binder_pair(_living())
        v["p_parent"][1] = 99
        v["p_seq"][1, :30] = 2
        _check(m, v)
        _check(m, v)
        assert "candidate_aggregate" not in fired(s)
        _check(m, v)
        assert "candidate_aggregate" in fired(s)

    def test_signal_correlation_needs_persistence(self):
        from firmament.instruments.milestones import Milestones
        s = fake_sched()
        m = Milestones(s)
        v = _living()
        cells = np.arange(120) % 64
        v["p_cell"][:120] = cells
        v["species"][18] = np.bincount(cells, minlength=64).reshape(8, 8) * 500
        _check(m, v)
        _check(m, v)
        assert "candidate_signal_correlation" not in fired(s)
        _check(m, v)
        assert "candidate_signal_correlation" in fired(s)

    def test_nonmetabolic_flow_needs_persistence(self):
        from firmament.instruments.milestones import Milestones
        s = fake_sched()
        m = Milestones(s)
        v = _living()
        v["membrane_store"][:] = 100
        for _ in range(4):
            _check(m, v)
        assert "candidate_nonmetabolic_flow" not in fired(s)
        _check(m, v)
        assert "candidate_nonmetabolic_flow" in fired(s)

    def test_state_roundtrip_prevents_refire(self):
        """Detector state saved in a snapshot and restored after resume: no re-fire."""
        from firmament.instruments.milestones import Milestones
        s = fake_sched()
        m = Milestones(s)
        v = make_view(recent_copies=[(2, 1, 10, 30, 0)])
        _check(m, v)
        m2 = Milestones(s)
        m2.load_state(m.state_dict())
        _check(m2, v)
        assert sum(r["event"] == "first_copy" for r in s.analysis.records) == 1


def test_novelty_detector_fixture_and_null():
    from firmament.instruments.novelty import Novelty
    s = fake_sched()
    nov = Novelty(s)
    v = make_view(cap=64)
    nov.sample(v)
    assert fired(s) == set()
    v["p_state"][:10] = 1
    v["p_len"][:10] = 8
    v["p_motifs"][:10] = 0b11                  # first combo: baseline, no event
    nov.sample(v)
    v["p_motifs"][5:10] = 0b111                # a never-seen combination
    nov.sample(v)
    assert "novelty_motif_combo" in fired(s)


@pytest.mark.slow
def test_instruments_change_sim_state_by_zero_bytes(tmp_path, device):
    """Observation never touches the world — on a LIVING seeded world, so every
    polymer-reading instrument path actually runs (v0.1 tested a dead world)."""
    from firmament.instruments.sampler import attach_instruments
    from tests.test_phase4_replay import assert_states_identical, state_bytes
    from tests.test_phase5_polymers import _life_sim

    cfg, s1, sched1, _ = _life_sim(tmp_path / "a", device)
    cfg.run.metrics_every_ticks = 20
    attach_instruments(sched1, cfg, tmp_path / "a" / "run")
    for _ in range(300):
        sched1.tick_once()
    assert len(sched1.metrics.buf) + sched1.metrics.part > 0
    _, s2, sched2, _ = _life_sim(tmp_path / "b", device)
    for _ in range(300):
        sched2.tick_once()
    assert_states_identical(state_bytes(s1), state_bytes(s2))


def test_ordinary_copying_never_fires_interaction_candidates(tmp_path, device):
    """Real-sim negative control: a binder-free replicase copying for hundreds of ticks
    must not produce candidate_predation or candidate_aggregate (v0.1 fired both
    within 0.6 sim-days of seeding)."""
    from firmament.instruments.sampler import attach_instruments
    from tests.test_phase5_polymers import _life_sim

    cfg, state, sched, _ = _life_sim(tmp_path, device, pp=200000, monomers=4000)
    cfg.run.metrics_every_ticks = 10
    attach_instruments(sched, cfg, tmp_path / "run")
    for _ in range(600):
        sched.tick_once()
    names = {r["event"] for r in sched.analysis.read_all()}
    assert "first_copy" in names
    assert not ({"candidate_predation", "candidate_aggregate"} & names), names


def test_seeded_run_produces_metrics_lineage_and_first_copy(tmp_path, device):
    from firmament.instruments.sampler import attach_instruments
    from tests.test_phase5_polymers import _life_sim

    cfg, state, sched, _ = _life_sim(tmp_path, device)
    cfg.run.metrics_every_ticks = 20
    attach_instruments(sched, cfg, tmp_path / "run")
    for _ in range(200):
        sched.tick_once()
    sched.metrics.flush()
    sched.lineage.flush()
    t = sched.metrics.read_all().to_pydict()
    assert len(t["tick"]) >= 5
    st = state.p_state.numpy()
    completed_now = int(((st >= 1) & (st <= 3)).sum())
    assert t["n_polymers"][-1] == completed_now          # counts exclude unfinished children
    assert t["n_building"][-1] == int((st == 5).sum())
    assert sched.lineage.db.execute("SELECT COUNT(*) FROM copies").fetchone()[0] > 0
    assert any(r["event"] == "first_copy" for r in sched.analysis.read_all())
    causal = sched.events.read_all()
    assert [r["event"] for r in causal] == ["seed_placed"]        # analysis never in causal log
