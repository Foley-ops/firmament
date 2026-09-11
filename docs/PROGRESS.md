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

## Phase 2 — Water (2026-09-11)

**Acceptance criteria met** (test names that prove it):
- `tests/test_phase2_water.py::test_water_conserved_machine_precision` — liquid+vapor total invariant to < 1e-9 relative over 5 sim-days of full water cycle (f64 pairwise transfers; the 1000-sim-day M0 run in Phase 9 extends this).
- `tests/test_phase2_water.py::test_slope_release_ends_in_basin` — >90% of a released blob reaches the basin at the slope's foot; no negative depths.
- `tests/test_phase2_water.py::test_rain_occurs` — evaporation drives a humid airmass over saturation; condensation (vapor drop, water gain, latent heat to air) observed.
- `tests/test_phase2_water.py::test_substep_count_logged_on_change` — CFL sub-step count changes are logged, per the guide.

**Fixes made by tests:** linear bottom drag was unphysically weak (runoff hit >10 m/s and
pinned the CFL cap); replaced with quadratic Manning friction (n=0.04, real shallow-flow
drag). CFL cap raised to 800 and clamps loudly.

**Simplifications vs writeup** (real analog preserved):
- Vapor lateral transport = conservative relaxation toward patch mean (turbulent
  boundary-layer mixing at ~1 km scale; advective CFL at 1 m cells would be ~120).
- Dissolved species ride per-tick accumulated face water fluxes with stochastic integer
  rounding (keyed per face — bitwise deterministic, exactly conservative).
- Erosion: sediment flux ∝ |u|³ through faces (stream-power law analog).

**Benchmarks:** 32² water world ~ 150–600 substeps/tick after Manning friction settles
flows; full pipeline at 256² ≈ 51 tps.

## Phase 3 — Chemistry (2026-09-11)

**Acceptance criteria met** (test names that prove it):
- `tests/test_phase3_chemistry.py::TestLoader::test_valid_network_loads`
- `tests/test_phase3_chemistry.py::TestLoader::test_rejects_unbalanced_reaction`
- `tests/test_phase3_chemistry.py::TestLoader::test_rejects_species_without_sink`
- `tests/test_phase3_chemistry.py::test_closed_box_atoms_exact_and_equilibrating` — dark sealed box: element totals EXACT to the integer over 3000 ticks; reaction activity declines monotonically toward equilibrium (full detailed balance takes ~23 sim-days — the slowest reaction's timescale; direction + exactness tested at unit horizon).
- `tests/test_phase3_chemistry.py::test_lit_box_pp_day_night_and_depth` — dark-control experiment: lit world's daytime P~P gain > 3x the geothermal-only control; the gain vanishes at night; P~P anti-correlates with water depth (Beer-Lambert).

**Simplifications vs writeup** (real analog preserved):
- Rates parameterized as k300 (rate at 300 K) + Ea, i.e. k(T) = k300·exp(−Ea/R·(1/T−1/300)) — standard Arrhenius re-parameterization, easier to tune honestly.
- Vents have no special reaction flag: they are simply HOT, and Arrhenius does the rest.
- Dissolved reactive H2O is a conserved integer species; bulk water_depth is the inert solvent (water activity vs bulk phase). No vent outgassing in v0 — element totals strictly closed.

**Benchmarks:** chemistry kernel (30 reactions, serial per cell) adds ~1 ms/tick at 32².
