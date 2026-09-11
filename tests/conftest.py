"""Shared fixtures: tiny in-memory configs, isolated cwd, device selection."""
from __future__ import annotations

from pathlib import Path

import pytest
import warp as wp

REPO = Path(__file__).resolve().parents[1]

wp.config.log_level = getattr(wp, "LOG_WARNING", None) or 30
wp.init()


def tiny_cfg(n: int = 32, water: float = 0.4, seed: int = 20260910, cap: int = 20000,
             dt: float = 60.0, vents=None, solar: float = 1361.0, log_level: str = "WARNING"):
    from firmament.config import Config
    return Config.model_validate({
        "run": {"name": "test", "seed": seed, "dt_seconds": dt,
                "snapshot_every_sim_days": 100000, "metrics_every_ticks": 100,
                "audit_every_ticks": 500},
        "world": {"size": [n, n], "cell_meters": 1.0, "terrain_seed": 7,
                  "vents": vents if vents is not None else [[n // 4, n // 2]],
                  "rotation_period_hours": 24, "year_length_days": 365,
                  "axial_tilt_deg": 23.5, "solar_constant_wm2": solar,
                  "geothermal_flux_wm2": 0.1, "initial_water_fraction": water},
        "chemistry": {"file": str(REPO / "configs/chemistry_v0.yaml"), "integer_counts": True},
        "polymers": {"capacity": cap, "max_length": 256,
                     "alphabet": ["M1", "M2", "M3", "M4"],
                     "mutation_rate_per_monomer": 0.005, "hydrolysis_base_rate": 1e-7,
                     "copy_energy_per_monomer": 1,
                     "genetic_code": str(REPO / "configs/genetic_code_v0.yaml")},
        "logging": {"level": log_level, "dir": None},
    })


@pytest.fixture
def isolated_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(scope="session")
def device():
    return "cuda:0" if wp.is_cuda_available() else "cpu"


def build_test_sim(cfg, device: str, run_dir: Path):
    """Full module stack, no server/instruments, ready to tick."""
    from firmament.cli import build_sim
    state, sched = build_sim(cfg, run_dir, device)
    from firmament.io.events import EventLog
    from firmament.io.lineage import LineageDB
    from firmament.io.snapshot import SnapshotManager
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "snapshots").mkdir(exist_ok=True)
    sched.events = EventLog(run_dir / "events.jsonl")
    sched.lineage = LineageDB(run_dir / "lineage.sqlite")
    sched.poly.lineage = sched.lineage
    sched.snapshots = SnapshotManager(run_dir / "snapshots", cfg)
    return state, sched
