"""Causal history & provenance (Codex review P0 items, 2026-09-25).

A run is (frozen config + rule-file contents, seed, causal log). These tests prove the
log alone reproduces the world — including the seed, developer edits, and events
logged after the last snapshot — and that analysis output can never alter physics."""
from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from tests.conftest import build_test_sim, tiny_cfg
from tests.test_phase4_replay import assert_states_identical, state_bytes
from tests.test_phase5_polymers import SEED_SEQ


def _fresh(tmp_path, name, device, n=16):
    cfg = tiny_cfg(n=n, cap=2000)
    return (cfg, *build_test_sim(cfg, device, tmp_path / name))


def _stock_cell(sched, state, x, y):
    import warp as wp
    spec = state.species.numpy()
    for m in ("M1", "M2", "M3", "M4"):
        spec[sched.chem.index[m], y, x] += 500
    spec[sched.chem.index["PP"], y, x] += 50000
    state.species = wp.array(spec, dtype=wp.int32, device=state.device)


def _wet_cell(state):
    y, x = np.argwhere(state.water_depth.numpy() > 0.3)[0]
    return int(x), int(y)


def test_seed_and_edit_replay_from_log_alone(tmp_path, device):
    """Codex repro: a logged seed_placed did not recreate the seed on replay."""
    from firmament.io.events import replay_events
    from firmament.operator import developer
    from firmament.operator.console import place_seed

    cfg, s1, k1 = _fresh(tmp_path, "a", device)
    x, y = _wet_cell(s1)
    _stock_cell(k1, s1, x, y)            # identical pre-seed preparation below
    for _ in range(5):
        k1.tick_once()
    place_seed(k1, s1, SEED_SEQ, x, y)
    for _ in range(20):
        k1.tick_once()
    developer.set_species(k1, "PP", x, y, 777, "test edit")
    for _ in range(100):
        k1.tick_once()

    _, s2, k2 = _fresh(tmp_path, "b", device)
    _stock_cell(k2, s2, x, y)
    assert replay_events(k2, tmp_path / "a" / "events.jsonl") == 2
    while s2.tick < s1.tick:
        k2.tick_once()
    assert not k2.replay_queue
    assert_states_identical(state_bytes(s1), state_bytes(s2))
    assert s2.causal_n == 2


def test_resume_replays_events_logged_after_the_snapshot(tmp_path, device):
    """Codex repro: resume dropped an earthquake logged after the last snapshot."""
    from firmament.io.events import replay_events
    from firmament.operator import console

    cfg, s1, k1 = _fresh(tmp_path, "a", device)
    for _ in range(40):
        k1.tick_once()
    k1.snapshots.save(s1, k1)
    for _ in range(30):
        k1.tick_once()
    console.request(k1, "earthquake", {"magnitude_m": 0.7, "radius": 6})
    for _ in range(30):
        k1.tick_once()

    _, s2, k2 = _fresh(tmp_path, "a", device)             # same run dir: same log
    k2.snapshots.load(s2, k2, tick=40)
    assert replay_events(k2, tmp_path / "a" / "events.jsonl") == 1
    while s2.tick < s1.tick:
        k2.tick_once()
    assert_states_identical(state_bytes(s1), state_bytes(s2))


def test_analysis_output_cannot_change_a_stochastic_event(tmp_path, device):
    """Codex repro: an extra analysis event before an earthquake changed its RNG draw."""
    from firmament.operator import console

    worlds = []
    for name, noise in (("quiet", 0), ("chatty", 5)):
        _, s, k = _fresh(tmp_path, name, device)
        for _ in range(10):
            k.tick_once()
        for i in range(noise):
            k.analysis.append("novelty_motif_combo", s.tick, component="novelty", combo=i)
        console.request(k, "earthquake", {"magnitude_m": 1.0, "radius": 8})
        for _ in range(5):
            k.tick_once()
        worlds.append(s)
    assert np.array_equal(worlds[0].elevation.numpy(), worlds[1].elevation.numpy())


def test_fork_history_is_cut_at_the_fork_point(isolated_cwd, device):
    from firmament.io import rundir
    from firmament.io.snapshot import fork_run
    from firmament.operator import console

    cfg = tiny_cfg(n=16, cap=500)
    run = rundir.create(cfg)
    s, k = build_test_sim(cfg, device, run)
    console.request(k, "rain", {"intensity_mm": 5.0})
    for _ in range(10):
        k.tick_once()
    k.snapshots.save(s, k)                                  # contains the rain
    console.request(k, "drought", {"strength": 0.3})       # AFTER the fork point
    for _ in range(10):
        k.tick_once()
    k.snapshots.save(s, k)
    fork = fork_run(run, "tick_000000000010.zarr")
    events = [json.loads(x)["event"] for x in open(fork / "events.jsonl") if x.strip()]
    assert events == ["rain"]


def test_config_hash_covers_rule_file_contents(tmp_path):
    """Codex: editing a rules file used to leave the config hash unchanged."""
    cfg = tiny_cfg()
    h0 = cfg.hash()
    doc = yaml.safe_load(open(cfg.chemistry.file))
    doc["reactions"][0]["k300"] *= 2
    edited = tmp_path / "chem.yaml"
    edited.write_text(yaml.safe_dump(doc, sort_keys=False))
    cfg2 = cfg.model_copy(deep=True)
    cfg2.chemistry.file = str(edited)
    assert cfg2.hash() != h0
    copy = tmp_path / "same.yaml"
    copy.write_bytes(open(cfg.chemistry.file, "rb").read())
    cfg3 = cfg.model_copy(deep=True)
    cfg3.chemistry.file = str(copy)
    assert cfg3.hash() == h0                                # path moves are not rule changes


def test_run_dir_freezes_rules_and_records_provenance(isolated_cwd):
    from firmament.config import Config
    from firmament.io import rundir
    cfg = tiny_cfg()
    run = rundir.create(cfg)
    frozen = Config.load(run / "config.yaml")
    assert frozen.chemistry.file.startswith(str(run))
    assert frozen.hash() == cfg.hash()
    prov = rundir.read_meta(run)["provenance"]
    assert prov["ruleset"] == "genesis-v0.2" and len(prov["git_commit"]) == 40
    assert set(prov["rule_files_sha256"]) == {"chemistry", "genetic_code"}


def test_developer_mode_cannot_be_enabled_over_the_network(tmp_path, device):
    """Codex repro: an unauthenticated client enabled developer mode and edited state."""
    from fastapi.testclient import TestClient

    from firmament.server.api import make_app
    _, s, k = _fresh(tmp_path, "r", device)
    (tmp_path / "r" / "meta.json").write_text(json.dumps(
        {"run_id": "r", "parent": None, "created": "x", "touched": False}))
    client = TestClient(make_app(k, tmp_path / "r"))
    client.post("/api/developer/enable", json={"on": True})
    r = client.post("/api/developer/set_species", json={"species": "PP", "x": 1, "y": 1, "count": 9})
    assert r.status_code == 403 and not k.mailbox
    k.developer_allowed = True                              # only the --developer flag does this
    assert client.post("/api/developer/set_species",
                       json={"species": "PP", "x": 1, "y": 1, "count": 9}).status_code == 200
    k.tick_once()
    assert int(s.species.numpy()[k.chem.index["PP"], 1, 1]) == 9
    assert [r["event"] for r in k.events.read_all()] == ["EDIT"]


def test_only_transient_faults_are_auto_restarted():
    from firmament.cli import _is_transient
    from firmament.core.audit import AuditError
    assert _is_transient(RuntimeError("Warp error: unknown stream"))
    assert not _is_transient(AuditError("element conservation violated"))
    assert not _is_transient(RuntimeError("polymer capacity exhausted — raise polymers.capacity"))


@pytest.mark.slow
def test_resume_is_indistinguishable_from_an_uninterrupted_run(isolated_cwd, device):
    """End to end through the CLI: seed a tiny world, then either run 300 ticks straight
    or stop halfway and resume. Causal log, analysis log, metrics and final state must
    match exactly — no re-fired milestones, no duplicated rows, no lost history."""
    import zarr

    from firmament import cli
    from firmament.io import rundir
    from firmament.io.metrics import MetricsWriter

    cfg = tiny_cfg(n=16, cap=2000)
    cfg.run.metrics_every_ticks = 20
    cfg.chemistry.overrides = {m: {"init_wet": 400} for m in ("M1", "M2", "M3", "M4")} | \
        {"PP": {"init_wet": 20000}}
    p = isolated_cwd / "c.yaml"
    p.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))

    def make(label):
        cli.main(["--device", device, "run", "--config", str(p), "--ticks", "1"])
        rid = sorted(r["run_id"] for r in rundir.list_runs())[-1]
        s, _ = build_test_sim(cfg, "cpu", isolated_cwd / "probe" / label)
        x, y = _wet_cell(s)
        cli.main(["--device", device, "seed", "--run", rid, "--sequence", SEED_SEQ,
                  "--cell", f"{x},{y}"])
        return rid

    straight = make("a")
    cli.main(["--device", device, "resume", "--run", straight, "--ticks", "300"])
    split = make("b")
    cli.main(["--device", device, "resume", "--run", split, "--ticks", "150"])
    cli.main(["--device", device, "resume", "--run", split, "--ticks", "150"])

    def strip(rs):
        return [{k: v for k, v in r.items() if k not in ("run_id",)} for r in rs]

    for name in ("events.jsonl", "analysis.jsonl"):
        a = strip(json.loads(x) for x in open(f"runs/{straight}/{name}") if x.strip())
        b = strip(json.loads(x) for x in open(f"runs/{split}/{name}") if x.strip())
        assert a == b, f"{name} differs after resume"
    assert any(r["event"] == "first_copy" for r in
               (json.loads(x) for x in open(f"runs/{split}/analysis.jsonl") if x.strip()))
    ma = MetricsWriter(rundir.RUNS / straight / "metrics").read_all().to_pylist()
    mb = MetricsWriter(rundir.RUNS / split / "metrics").read_all().to_pylist()
    assert sorted(r["tick"] for r in mb) == sorted({r["tick"] for r in mb})   # no duplicates
    assert sorted(ma, key=lambda r: r["tick"]) == sorted(mb, key=lambda r: r["tick"])
    ga = zarr.open_group(f"runs/{straight}/snapshots/tick_000000000301.zarr", mode="r")
    gb = zarr.open_group(f"runs/{split}/snapshots/tick_000000000301.zarr", mode="r")
    for key in ga.array_keys():
        assert np.array_equal(ga[key][:], gb[key][:]), key
