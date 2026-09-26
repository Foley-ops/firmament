"""Config loading and validation. The frozen copy in runs/<id>/config.yaml is canonical.

Strict (docs/DECISIONS.md, 2026-09-25): unknown fields and out-of-range values are
errors, never silently dropped or accepted.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

POLYMER_STEP_SECONDS = 60     # polymer events are calibrated per 60 s (see polymers.py)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RunCfg(Strict):
    name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    dt_seconds: float = Field(60.0, gt=0)
    snapshot_every_sim_days: float = Field(30.0, gt=0)
    metrics_every_ticks: int = Field(100, ge=1)
    audit_every_ticks: int = Field(1000, ge=1)

    @model_validator(mode="after")
    def _dt_is_whole_polymer_steps(self):
        k = self.dt_seconds / POLYMER_STEP_SECONDS
        if k != int(k):
            raise ValueError(f"dt_seconds must be a whole multiple of {POLYMER_STEP_SECONDS} s "
                             f"(polymer processes are sub-stepped at that resolution); got "
                             f"{self.dt_seconds}")
        return self


class WorldCfg(Strict):
    size: tuple[int, int]
    cell_meters: float = 1.0
    terrain_seed: int = 7
    vents: list[tuple[int, int]] = Field(default_factory=list)
    rotation_period_hours: float = Field(24.0, gt=0)
    year_length_days: float = Field(365.0, gt=0)
    axial_tilt_deg: float = Field(23.5, ge=0, le=90)
    solar_constant_wm2: float = Field(1361.0, ge=0)
    geothermal_flux_wm2: float = Field(0.1, ge=0)
    initial_water_fraction: float = Field(0.4, ge=0, le=1)

    @model_validator(mode="after")
    def _check(self):
        h, w = self.size
        if h < 4 or w < 4:
            raise ValueError(f"world size must be at least 4x4, got {self.size}")
        for vx, vy in self.vents:
            if not (0 <= vx < w and 0 <= vy < h):
                raise ValueError(f"vent {(vx, vy)} lies outside the {w}x{h} world")
        if self.cell_meters != 1.0:
            # fluid fluxes, lateral conduction and transport are calibrated for 1 m cells;
            # accepting another value would silently change the physics
            raise ValueError("cell_meters must be 1.0 in this ruleset (see docs/DECISIONS.md)")
        return self


class ChemistryCfg(Strict):
    file: str
    integer_counts: bool = True
    # per-species init overrides (M1-protocol environment tuning)
    overrides: dict[str, dict[str, int]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self):
        if not self.integer_counts:
            raise ValueError("integer_counts must be true (exact element conservation)")
        for sp, d in self.overrides.items():
            bad = set(d) - {"init_wet", "init_dry"}
            if bad:
                raise ValueError(f"override for {sp} has unknown keys {sorted(bad)}")
            if any(v < 0 for v in d.values()):
                raise ValueError(f"override for {sp} has a negative count")
        return self


class PolymersCfg(Strict):
    capacity: int = Field(ge=1)
    max_length: int = Field(256, ge=2, le=256)
    alphabet: list[str] = Field(default_factory=lambda: ["M1", "M2", "M3", "M4"])
    mutation_rate_per_monomer: float = Field(0.005, ge=0, lt=1)
    hydrolysis_base_rate: float = Field(1e-7, ge=0)
    copy_energy_per_monomer: int = Field(2, ge=1)
    genetic_code: str = "configs/genetic_code_v0.yaml"

    @model_validator(mode="after")
    def _check(self):
        if self.alphabet != ["M1", "M2", "M3", "M4"]:
            raise ValueError("alphabet must be [M1, M2, M3, M4] in this ruleset")
        return self


class LoggingCfg(Strict):
    level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")
    dir: str | None = None


class Config(Strict):
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

    def rule_files(self) -> dict[str, str]:
        return {"chemistry": self.chemistry.file, "genetic_code": self.polymers.genetic_code}

    def hash(self) -> str:
        """Identity of the run's physics: config values plus the CONTENTS of every rule
        file (paths excluded), so editing a rules file can never be mistaken for the
        same world."""
        d = self.model_dump(mode="json")
        d["chemistry"]["file"] = file_sha256(self.chemistry.file)
        d["polymers"]["genetic_code"] = file_sha256(self.polymers.genetic_code)
        blob = yaml.safe_dump(d, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:8]


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
