"""Config loading and validation. The frozen copy in runs/<id>/config.yaml is canonical."""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class RunCfg(BaseModel):
    name: str
    seed: int
    dt_seconds: float = 60.0
    snapshot_every_sim_days: float = 30.0
    metrics_every_ticks: int = 100
    audit_every_ticks: int = 1000


class WorldCfg(BaseModel):
    size: tuple[int, int]
    cell_meters: float = 1.0
    terrain_seed: int = 7
    vents: list[tuple[int, int]] = Field(default_factory=list)
    rotation_period_hours: float = 24.0
    year_length_days: float = 365.0
    axial_tilt_deg: float = 23.5
    solar_constant_wm2: float = 1361.0
    geothermal_flux_wm2: float = 0.1
    initial_water_fraction: float = 0.4


class ChemistryCfg(BaseModel):
    file: str
    integer_counts: bool = True
    # per-species init overrides (M1-protocol environment tuning)
    overrides: dict[str, dict] = Field(default_factory=dict)


class PolymersCfg(BaseModel):
    capacity: int
    max_length: int = 256
    alphabet: list[str] = Field(default_factory=lambda: ["M1", "M2", "M3", "M4"])
    mutation_rate_per_monomer: float = 0.005
    hydrolysis_base_rate: float = 1e-7
    copy_energy_per_monomer: int = 1
    genetic_code: str = "configs/genetic_code_v0.yaml"


class LoggingCfg(BaseModel):
    level: str = "INFO"
    dir: str | None = None


class Config(BaseModel):
    run: RunCfg
    world: WorldCfg
    chemistry: ChemistryCfg
    polymers: PolymersCfg
    logging: LoggingCfg = Field(default_factory=LoggingCfg)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))

    def canonical_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=True)

    def hash(self) -> str:
        return hashlib.sha256(self.canonical_yaml().encode()).hexdigest()[:8]
