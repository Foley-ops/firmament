"""Phase 8 acceptance: every natural event is replayable (event log -> identical
state); add-species refuses anything that changes existing rates; developer edits
mark the run TOUCHED and the mark propagates to forks."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import yaml

from tests.conftest import REPO, build_test_sim, tiny_cfg
from tests.test_phase4_replay import assert_states_identical, state_bytes

EVENTS = [
    (60, "rain", {"cx": 10, "cy": 10, "radius": 8, "intensity_mm": 20.0}),
    (100, "drought", {"cx": 20, "cy": 20, "radius": 10, "strength": 0.4}),
    (140, "flood", {"cx": 16, "cy": 16, "radius": 6, "rise_m": 0.3}),
    (180, "earthquake", {"cx": 8, "cy": 24, "radius": 6, "magnitude_m": 0.5}),
    (220, "volcano", {"cx": 24, "cy": 8, "radius": 5, "heat_J_m2": 1e6, "cone_m": 1.0,
                      "mineral_counts": 1000}),
    (260, "meteor", {"cx": 16, "cy": 24, "energy_J": 1e9, "crater_m": 0.5, "dust": 0.2}),
    (300, "solar", {"multiplier": 1.05}),
    (340, "climate", {"multiplier": 0.95}),
]


@pytest.mark.slow
def test_every_event_replays_to_identical_state(tmp_path, device):
    from firmament.io.events import replay_events
    from firmament.operator import console

    cfg = tiny_cfg(n=32, cap=1000)
    s1, sched1 = build_test_sim(cfg, device, tmp_path / "run")
    todo = list(EVENTS)
    while s1.tick < 400:
        if todo and s1.tick + 1 == todo[0][0]:
            _, ev, params = todo.pop(0)
            console.request(sched1, ev, params)   # logs then queues for next tick
        sched1.tick_once()
    b1 = state_bytes(s1)
    n_applied = len(EVENTS) - len(todo)
    assert n_applied == len(EVENTS), f"only {n_applied} events applied"

    # fresh world, same config+seed, re-apply from the LOG only
    s2, sched2 = build_test_sim(cfg, device, tmp_path / "run2")
    replay_events(sched2, tmp_path / "run" / "events.jsonl")
    assert len(sched2.replay_queue) == len(EVENTS)
    while s2.tick < 400:
        sched2.tick_once()
    assert_states_identical(b1, state_bytes(s2))


def test_solar_multiplier_survives_snapshot(tmp_path, device):
    from firmament.operator import console

    cfg = tiny_cfg(n=16, cap=500)
    s1, sched1 = build_test_sim(cfg, device, tmp_path / "r")
    console.request(sched1, "solar", {"multiplier": 1.25})
    sched1.tick_once()
    sched1.tick_once()
    assert s1.solar_mult == 1.25
    sched1.snapshots.save(s1, sched1)
    s2, sched2 = build_test_sim(cfg, device, tmp_path / "r")
    sched2.snapshots.load(s2, sched2)
    assert s2.solar_mult == 1.25


class TestAddSpecies:
    def _write(self, tmp_path, doc):
        p = tmp_path / "chem_new.yaml"
        p.write_text(yaml.safe_dump(doc, sort_keys=False))
        return str(p)

    def test_refuses_changed_rate(self, tmp_path, device):
        from firmament.operator.console import _add_species
        cfg = tiny_cfg(n=8, cap=100)
        state, sched = build_test_sim(cfg, device, tmp_path / "r")
        doc = yaml.safe_load(open(REPO / "configs/chemistry_v0.yaml"))
        doc["reactions"][0]["k300"] = 99.0            # existing rate changed
        with pytest.raises(ValueError, match="refused"):
            _add_species(sched, {"file": self._write(tmp_path, doc)})

    def test_refuses_reordered_species(self, tmp_path, device):
        from firmament.operator.console import _add_species
        cfg = tiny_cfg(n=8, cap=100)
        state, sched = build_test_sim(cfg, device, tmp_path / "r")
        doc = yaml.safe_load(open(REPO / "configs/chemistry_v0.yaml"))
        sp = doc["species"]
        keys = list(sp)
        keys[0], keys[1] = keys[1], keys[0]
        doc["species"] = {k: sp[k] for k in keys}
        with pytest.raises(ValueError, match="refused"):
            _add_species(sched, {"file": self._write(tmp_path, doc)})

    def test_valid_superset_accepted_as_v0_fork_path(self, tmp_path, device):
        """A pure addition passes validation; v0 applies it via fork (documented)."""
        from firmament.operator.console import _add_species
        cfg = tiny_cfg(n=8, cap=100)
        state, sched = build_test_sim(cfg, device, tmp_path / "r")
        doc = copy.deepcopy(yaml.safe_load(open(REPO / "configs/chemistry_v0.yaml")))
        doc["species"]["XE"] = {"formula": {"S": 2}, "init_wet": 0, "init_dry": 0}
        doc["reactions"].append({"name": "xe_form", "sub": {"S": 2}, "prod": {"XE": 1},
                                 "dh": -1, "ea": 50000, "k300": 1e-9})
        doc["reactions"].append({"name": "xe_decay", "sub": {"XE": 1}, "prod": {"S": 2},
                                 "dh": 1, "ea": 50000, "k300": 1e-9})
        with pytest.raises(NotImplementedError, match="fork"):
            _add_species(sched, {"file": self._write(tmp_path, doc)})


class TestDeveloperMode:
    def test_edit_marks_touched_and_propagates_to_fork(self, isolated_cwd, device):
        from firmament.io import rundir
        from firmament.io.snapshot import fork_run
        from firmament.operator import developer

        cfg = tiny_cfg(n=16, cap=500)
        run_dir = rundir.create(cfg)
        state, sched = build_test_sim(cfg, device, run_dir)
        assert rundir.read_meta(run_dir)["touched"] is False
        developer.set_species(sched, "PP", 3, 3, 12345, "test edit")
        assert rundir.read_meta(run_dir)["touched"] is True     # marked immediately
        events = [json.loads(x) for x in open(run_dir / "events.jsonl") if x.strip()]
        edits = [e for e in events if e["event"] == "EDIT"]
        assert len(edits) == 1 and edits[0]["component"] == "developer"
        assert int(state.species.numpy()[sched.chem.index["PP"], 3, 3]) == 12345
        for _ in range(5):
            sched.tick_once()
        sched.snapshots.save(state, sched)
        fork_dir = fork_run(run_dir)
        assert rundir.read_meta(fork_dir)["touched"] is True    # mark propagates

    def test_edit_polymer_rescans_motifs(self, tmp_path, device):
        from firmament.io import rundir
        from firmament.operator import developer
        from tests.test_phase5_polymers import SEED_SEQ, _life_sim

        cfg, state, sched, _ = _life_sim(tmp_path, device)
        sched.run = rundir.create(cfg)          # developer marks via run dir
        developer.edit_polymer(sched, 0, sequence="M1M3M2M4" * 4)  # replicase removed
        assert int(state.p_motifs.numpy()[0]) == 0
        developer.edit_polymer(sched, 0, sequence=SEED_SEQ)
        assert int(state.p_motifs.numpy()[0]) == sched.poly.repl_bit


def test_cli_event_command(isolated_cwd, device):
    """Guide: events work from CLI too (offline path on a resumable run)."""
    import yaml as _yaml

    from firmament import cli

    cfg = tiny_cfg(n=16, cap=500, log_level="INFO")
    p = isolated_cwd / "c.yaml"
    p.write_text(_yaml.safe_dump(cfg.model_dump(mode="json")))
    cli.main(["--device", device, "run", "--config", str(p), "--ticks", "3"])
    from firmament.io import rundir
    run_id = rundir.list_runs()[0]["run_id"]
    cli.main(["--device", device, "event", "--run", run_id, "--type", "rain",
              "--params", '{"intensity_mm": 5.0, "radius": 6}'])
    events = [json.loads(x) for x in open(f"runs/{run_id}/events.jsonl") if x.strip()]
    assert any(e["event"] == "rain" for e in events)
