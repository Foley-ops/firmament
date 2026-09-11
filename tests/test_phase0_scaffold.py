"""Phase 0 acceptance: structured JSONL logging; CLI creates the run directory
structure and completes a short headless run logging throughput."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from tests.conftest import REPO, tiny_cfg

REQUIRED_FIELDS = {"run_id", "tick", "sim_time", "component", "level", "msg"}


def test_logging_valid_jsonl_with_required_fields(tmp_path):
    from firmament.io.logging import CLOCK, get, setup

    CLOCK.run_id = "test-run"
    CLOCK.tick = 42
    CLOCK.sim_seconds = 42 * 60.0
    setup(tmp_path, "INFO")
    get("phase0").info("hello", extra={"custom_field": 7})
    get("phase0").warning("warn msg")
    lines = (tmp_path / "sim.log").read_text().strip().splitlines()
    assert len(lines) >= 2
    for line in lines:
        rec = json.loads(line)                      # valid JSON on every line
        assert REQUIRED_FIELDS <= set(rec)
        assert rec["run_id"] == "test-run"
        assert rec["tick"] == 42
    assert json.loads(lines[0])["custom_field"] == 7


def test_log_level_runtime_adjustable(tmp_path):
    from firmament.io.logging import get, set_level, setup

    setup(tmp_path, "INFO")
    set_level("ERROR")
    get("phase0").info("should not appear")
    set_level("INFO")
    get("phase0").info("should appear")
    text = (tmp_path / "sim.log").read_text()
    assert "should not appear" not in text
    assert "should appear" in text


def test_cli_run_creates_structure_and_completes(isolated_cwd, device):
    from firmament import cli

    cfg = tiny_cfg(n=32, cap=5000, log_level="INFO")
    cfg_path = isolated_cwd / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    cli.main(["--device", device, "run", "--config", str(cfg_path), "--ticks", "5"])

    runs = list(Path("runs").iterdir())
    assert len(runs) == 1
    d = runs[0]
    for sub in ("logs", "snapshots", "metrics"):
        assert (d / sub).is_dir()
    assert (d / "events.jsonl").exists()
    assert (d / "lineage.sqlite").exists()
    assert (d / "meta.json").exists()
    frozen = yaml.safe_load((d / "config.yaml").read_text())
    assert frozen["run"]["seed"] == cfg.run.seed     # frozen copy is canonical
    log_text = (d / "logs" / "sim.log").read_text()
    assert "ticks_per_sec" in log_text               # throughput logged
    assert (d / "snapshots").is_dir() and list((d / "snapshots").iterdir())  # shutdown snapshot
