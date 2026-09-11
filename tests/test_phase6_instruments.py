"""Phase 6 acceptance: every milestone detector fires on a synthetic fixture and
stays silent on a null fixture; instruments running vs disabled change the sim
state by ZERO bytes; a seeded run produces metrics parquet, a lineage DB, and the
first_copy milestone in events.jsonl."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tests.conftest import REPO, build_test_sim, tiny_cfg


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


def fake_sched():
    return SimpleNamespace(events=FakeEvents(), lineage=FakeLineage(),
                           chem=SimpleNamespace(index={"A": 18}))


def fired(sched):
    return {r["event"] for r in sched.events.records}


def _pop_row(v):
    from firmament.instruments import population
    return population.sample(v)


class TestMilestoneDetectors:
    def test_null_fixture_fires_nothing(self):
        from firmament.instruments.milestones import Milestones
        s = fake_sched()
        m = Milestones(s)
        v = make_view()                                   # dead world
        for _ in range(6):
            m.check(v, _pop_row(v))
        assert fired(s) == set()

    def test_first_copy(self):
        from firmament.instruments.milestones import Milestones
        s = fake_sched()
        m = Milestones(s)
        v = make_view(recent_copies=[(2, 1, 10, 30, 0)])
        m.check(v, _pop_row(v))
        assert "first_copy" in fired(s)

    def _living(self, cap=200, n_alive=120, seq_val=1):
        v = make_view(cap=cap)
        v["p_state"][:n_alive] = 1
        v["p_id"][:n_alive] = np.arange(2, n_alive + 2)
        v["p_parent"][:n_alive] = 1
        v["p_len"][:n_alive] = 30
        v["p_seq"][:n_alive, :30] = seq_val
        return v

    def test_fixation(self):
        from firmament.instruments.milestones import Milestones
        s = fake_sched()
        m = Milestones(s)
        v = self._living()
        v["p_seq"][0, :30] = 3                            # one variant, 119 identical others
        m.check(v, _pop_row(v))
        assert "fixation" in fired(s)

    def test_mutant_100_generations(self):
        from firmament.instruments.milestones import Milestones
        s = fake_sched()                                  # FakeLineage depth=150
        m = Milestones(s)
        v = self._living()
        v["p_mut"][:5] = 2
        m.check(v, _pop_row(v))
        assert "mutant_100_generations" in fired(s)

    def test_first_compartment_and_outcompete(self):
        from firmament.instruments.milestones import Milestones
        s = fake_sched()
        m = Milestones(s)
        v = self._living()
        v["compartment_id"][2, 3] = 1
        cell = 2 * 8 + 3
        v["p_cell"][:80] = cell                           # 80/120 live inside
        m.check(v, _pop_row(v))
        assert "first_compartment" in fired(s)
        assert "compartment_outcompetes" in fired(s)

    def test_predation_like(self):
        from firmament.instruments.milestones import Milestones
        s = fake_sched()
        m = Milestones(s)
        v = self._living()
        v["p_partner"][0], v["p_partner"][1] = 1, 0       # bound pair, slots 0 and 1
        m.check(v, _pop_row(v))
        v2 = self._living()
        v2["p_partner"][0] = 1                            # survivor still points at victim
        v2["p_state"][1] = 0                              # partner died
        m.check(v2, _pop_row(v2))
        assert "predation_like" in fired(s)

    def test_aggregate_two_lineages(self):
        from firmament.instruments.milestones import Milestones
        s = fake_sched()
        m = Milestones(s)
        v = self._living()
        v["p_parent"][1] = 99                             # different ancestry
        v["p_partner"][0], v["p_partner"][1] = 1, 0
        m.check(v, _pop_row(v))
        assert "aggregate_two_lineages" in fired(s)

    def test_signal_correlation_needs_persistence(self):
        from firmament.instruments.milestones import Milestones
        s = fake_sched()
        m = Milestones(s)
        v = self._living()
        cells = np.arange(120) % 64
        v["p_cell"][:120] = cells
        a = np.zeros((8, 8), dtype=np.int32)
        dens = np.bincount(cells, minlength=64).reshape(8, 8)
        a[:] = dens * 500                                 # A tracks polymer density
        v["species"][18] = a
        m.check(v, _pop_row(v))
        m.check(v, _pop_row(v))
        assert "signal_correlation" not in fired(s)       # not yet persistent
        m.check(v, _pop_row(v))
        assert "signal_correlation" in fired(s)           # 3 consecutive samples

    def test_non_metabolic_energy_flow(self):
        from firmament.instruments.milestones import Milestones
        s = fake_sched()
        m = Milestones(s)
        v = self._living()
        v["membrane_store"][:] = 100                      # 6400 L held in membranes
        for _ in range(4):
            m.check(v, _pop_row(v))
        assert "non_metabolic_energy_flow" not in fired(s)
        m.check(v, _pop_row(v))                           # 5th consecutive sample
        assert "non_metabolic_energy_flow" in fired(s)

    def test_detectors_fire_once(self):
        from firmament.instruments.milestones import Milestones
        s = fake_sched()
        m = Milestones(s)
        v = make_view(recent_copies=[(2, 1, 10, 30, 0)])
        m.check(v, _pop_row(v))
        m.check(v, _pop_row(v))
        assert sum(1 for r in s.events.records if r["event"] == "first_copy") == 1


def test_novelty_detector_fixture_and_null(self=None):
    from firmament.instruments.novelty import Novelty
    s = fake_sched()
    nov = Novelty(s)
    v = make_view(cap=64)
    nov.sample(v)                                         # null: nothing
    assert fired(s) == set()
    v["p_state"][:10] = 1
    v["p_motifs"][:10] = 0b11                             # first combo: baseline, no event
    nov.sample(v)
    v["p_motifs"][5:10] = 0b111                           # a never-seen combination
    nov.sample(v)
    assert "novelty_motif_combo" in fired(s)


@pytest.mark.slow
def test_instruments_change_sim_state_by_zero_bytes(tmp_path, device):
    """Replay-grade proof that observation never touches the world."""
    from firmament.instruments.sampler import attach_instruments
    from tests.test_phase4_replay import state_bytes, assert_states_identical

    cfg = tiny_cfg(n=32, cap=2000)
    s1, sched1 = build_test_sim(cfg, device, tmp_path / "a")
    attach_instruments(sched1, cfg, tmp_path / "a")
    for _ in range(300):
        sched1.tick_once()

    s2, sched2 = build_test_sim(cfg, device, tmp_path / "b")
    for _ in range(300):                                  # no instruments at all
        sched2.tick_once()
    assert_states_identical(state_bytes(s1), state_bytes(s2))


def test_seeded_run_produces_metrics_lineage_and_first_copy(tmp_path, device):
    """Phase 6 done-when: a Phase-5-style world yields metrics parquet, lineage DB
    rows, and the first_copy milestone in events.jsonl."""
    import json

    from firmament.instruments.sampler import attach_instruments
    from tests.test_phase5_polymers import _life_sim

    cfg, state, sched, _ = _life_sim(tmp_path, device)
    cfg.run.metrics_every_ticks = 20
    attach_instruments(sched, cfg, tmp_path / "run")
    for _ in range(200):
        sched.tick_once()
    sched.metrics.flush()
    sched.lineage.flush()
    t = sched.metrics.read_all()
    assert t is not None and t.num_rows >= 5
    assert "n_polymers" in t.column_names and "distinct_seqs" in t.column_names
    assert max(t["n_polymers"].to_pylist()) > 1
    rows = sched.lineage.db.execute("SELECT COUNT(*) FROM copies").fetchone()[0]
    assert rows > 0
    events = [json.loads(x) for x in open(tmp_path / "run" / "events.jsonl") if x.strip()]
    assert any(e["event"] == "first_copy" for e in events)
    assert any(e["event"] == "seed_placed" for e in events)
