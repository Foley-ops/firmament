"""Phase 10 acceptance: crash leaves a usable crash.json and the run resumes from
the last snapshot; dead-era acceleration engages only without life and is logged;
the daily report is generated and readable."""
from __future__ import annotations

import json

import pytest
import yaml

from tests.conftest import build_test_sim, tiny_cfg


def test_crash_recovery(isolated_cwd, device):
    """Force a crash mid-run; verify crash.json; resume completes the mission."""
    from firmament import cli
    from firmament.io import rundir

    cfg = tiny_cfg(n=16, cap=500, log_level="INFO")
    cfg.run.snapshot_every_sim_days = 30.0 / 1440       # snapshot every 30 ticks
    p = isolated_cwd / "c.yaml"
    p.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))

    # sabotage: a module that raises at tick 50 (monkeypatched into the build)
    import firmament.cli as cli_mod
    orig_build = cli_mod.build_sim

    def sabotaged(cfg2, run_dir, device2):
        state, sched = orig_build(cfg2, run_dir, device2)

        def bomb(s, tick):
            if tick == 50:
                raise RuntimeError("injected fault")
        sched.add(bomb)
        return state, sched
    cli_mod.build_sim = sabotaged
    try:
        with pytest.raises(RuntimeError, match="injected fault"):
            cli.main(["--device", device, "run", "--config", str(p), "--ticks", "100"])
    finally:
        cli_mod.build_sim = orig_build

    run_id = rundir.list_runs()[0]["run_id"]
    crash = json.loads(open(f"runs/{run_id}/crash.json").read())
    assert "injected fault" in crash["exception"]
    assert crash["tick"] == 50
    assert crash["last_good_snapshot"] and "tick_000000000030" in crash["last_good_snapshot"]

    # resume from the snapshot (fault module gone) and finish
    cli.main(["--device", device, "resume", "--run", run_id, "--ticks", "40"])
    snaps = sorted((rundir.RUNS / run_id / "snapshots").glob("tick_*.zarr"))
    assert any("tick_000000000070" in s.name for s in snaps)   # 30 + 40


def test_dead_era_acceleration_engages_and_reverts(tmp_path, device, caplog):
    import logging

    import warp as wp

    from tests.test_phase5_polymers import SEED_SEQ

    cfg = tiny_cfg(n=16, cap=500)
    state, sched = build_test_sim(cfg, device, tmp_path / "r")
    with caplog.at_level(logging.INFO, logger="firmament.chemistry"):
        sched.tick_once()
        assert sched.chem.dt_scale == 10.0              # dead world -> 10x chemistry
        # give the seed cell materials, then place life
        spec = state.species.numpy()
        for m in ("M1", "M2", "M3", "M4"):
            spec[sched.chem.index[m], 8, 8] += 100
        state.species = wp.array(spec, dtype=wp.int32, device=device)
        from firmament.operator.console import place_seed
        place_seed(sched, state, SEED_SEQ, 8, 8)
        sched.tick_once()                               # poly.step sets p_count=1
        sched.tick_once()                               # chemistry sees life
        assert sched.chem.dt_scale == 1.0
    assert any("chemistry dt mode change" in r.message for r in caplog.records)


def test_daily_report_readable(tmp_path, device):
    from firmament.instruments.sampler import attach_instruments
    from firmament.io.report import write_report
    from tests.test_phase5_polymers import _life_sim

    cfg, state, sched, _ = _life_sim(tmp_path, device)
    run_dir = tmp_path / "run"
    (run_dir / "meta.json").write_text(json.dumps(
        {"run_id": "report-test", "parent": None, "created": "x", "touched": False}))
    cfg.run.metrics_every_ticks = 20
    attach_instruments(sched, cfg, run_dir)
    for _ in range(120):
        sched.tick_once()
    sched.metrics.flush()
    out = write_report(run_dir)
    text = out.read_text()
    for section in ("Population", "Diversity", "Milestones", "Disk"):
        assert f"## {section}" in text
    assert "first_copy" in text                         # the milestone made the report
    assert "polymers **" in text
