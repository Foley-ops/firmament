# Progress

## Phase 0 — Scaffold (2026-09-11)

**Acceptance criteria met** (test names that prove it):
- `tests/test_phase0_scaffold.py::test_logging_valid_jsonl_with_required_fields` — JSONL log records carry run_id, tick, sim_time, component, level, msg.
- `tests/test_phase0_scaffold.py::test_log_level_runtime_adjustable`
- `tests/test_phase0_scaffold.py::test_cli_run_creates_structure_and_completes` — CLI builds runs/<run_id>/{config.yaml,logs,snapshots,metrics,events.jsonl,lineage.sqlite,meta.json}.
- Done-when: `firmament run --config configs/world_small.yaml --ticks 1000` completed headless, logging throughput.

**Simplifications vs writeup** (real analog preserved):
- GPU benchmark decision (Warp over Taichi, 8.7×) and determinism strategy in docs/DECISIONS.md.
- run_id = wall-timestamp + config hash (identity only; no wall-clock enters sim state).

**Open questions for the Operator:** none at this phase.

**Benchmarks:** 256² world, full module pipeline, RTX 4090: **51 ticks/s** (water solver
sub-stepping dominates; dead-world chemistry+fields alone would be much faster).

## Phase 1 — Terrain, radiation, thermal (2026-09-11)

**Acceptance criteria met** (test names that prove it):
- `tests/test_phase1_radiation_thermal.py::TestSunAngles::test_midnight_is_dark`
- `tests/test_phase1_radiation_thermal.py::TestSunAngles::test_noon_is_bright`
- `tests/test_phase1_radiation_thermal.py::TestSunAngles::test_solstice_ordering` — summer > equinox > winter noon sun; night dark in all seasons.
- `tests/test_phase1_radiation_thermal.py::test_day_night_cycle_plausible` — surface swing > 3 K, peak in the afternoon (lags noon).
- `tests/test_phase1_radiation_thermal.py::test_energy_balance_one_year_dry` — 365 sim-days, no water: |in − out − Δstored| / max(in,out) < 0.1% (f64 accumulators make it ~exact by construction).

**Simplifications vs writeup** (real analog preserved):
- Sun direction uniform across the ~1 km patch (real: parallax negligible at this scale); terrain self-shadowing omitted (flagged optional in the guide).
- Grey two-stream atmosphere with greenhouse fraction 0.78 (real: longwave absorption by H2O/CO2; Earth-like equilibrium temperatures result).
- Fixed `latitude_deg` (default 15°) added to config — guide allows extending.
- Geothermal enters the sediment layer; vents are hot spots via `vent_flux` (real: hydrothermal conduction).

**Open questions:** none blocking; latitude is Operator-tunable.

**Benchmarks:** radiation+thermal alone, 48²: ~30k ticks/s (the year-long audit test runs in ~1 s).
