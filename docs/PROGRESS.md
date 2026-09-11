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

## Phase 4 — Snapshot, replay, determinism (2026-09-11)

**Acceptance criteria met** (test names that prove it):
- `tests/test_phase4_replay.py::test_replay_bit_identical[cpu]` and `[cuda:0]` — run 500 → snapshot A → run 500 → B; fresh sim resumed from A runs 500 → B'. Every state array matches B byte-for-byte on BOTH backends. This test blocks everything after it.
- `tests/test_phase4_replay.py::test_fork_first_snapshot_equals_parent` — fork gets a new run_id, parent pointer, and a first snapshot identical to the parent's, loadable into a fresh sim.

**Bug the replay test caught (fixed in code):** evaporation/rain changed the surface
layer's heat capacity without moving the water's sensible heat — an 8.8% energy audit
leak on the full pipeline. Fix: vapor carries fixed enthalpy LV + c_w·T_ref per kg;
phase changes move exact energy; audit's stored term includes vapor enthalpy.
Console rain/drought/flood events register the same ledger.

## Phase 5 — Polymers (2026-09-11)

**Acceptance criteria met** (test names that prove it):
- `tests/test_phase5_polymers.py::TestCopyGating::test_copies_with_resources` — replicase + monomers + P~P → copies (lineage rows written).
- `TestCopyGating::test_no_copy_without_energy` / `test_no_copy_without_monomers` — binding may occur but no child monomer is ever added.
- `TestCopyGating::test_copy_stalls_and_resumes` — starvation stalls a copy; refeeding finishes it.
- `test_motif_scan_matches_hand_computed` — kernel motif mask equals a hand-built truth table; child verified to be the template's REVERSE complement.
- `test_measured_mutation_rate_matches_epsilon` — 60k+ copied monomers at ε=0.02; observed mutations within 4σ binomial.
- `test_audits_hold_with_life` — element totals exact to the integer with polymers copying/dying (chain unit = M − H2O); energy books close (Δstored == e_chem exactly).
- `test_replay_bit_identical_with_polymers` — the Phase 4 protocol, seeded.

**Bug the motif test caught (fixed in code):** copying was building the PARALLEL
complement, which mirrors reverse-complement-palindromic motifs and would have lost
function every generation. Fix: antiparallel synthesis (child = reverse complement),
exactly like real polymerases — the reason palindromic sites survive replication.

**Simplifications vs writeup** (real analog preserved):
- Deterministic per-cell serial competition in polymer-id order (physics is
  observer-order-independent; GPU scheduling must be too).
- Compartments are cell-granular bags: membrane L accretes per cell, splits overflow
  to the L-richest neighbor (budding), compartment id gates diffusion/transport.
- Motor climbs the P~P gradient (chemotaxis toward energy).

Full suite: 29 tests green (GPU; replay also on CPU backend).

## Phase 6 — Instruments, lineage, milestones (2026-09-11)

**Acceptance criteria met** (test names that prove it):
- `tests/test_phase6_instruments.py::TestMilestoneDetectors` — 9 detectors each fire on a synthetic fixture, stay silent on the null fixture, and fire exactly once: first_copy, fixation, mutant_100_generations, first_compartment, compartment_outcompetes, predation_like, aggregate_two_lineages, signal_correlation (needs 3 consecutive samples), non_metabolic_energy_flow (needs 5).
- `test_novelty_detector_fixture_and_null` — never-seen motif combination flagged; silent on dead world.
- `test_instruments_change_sim_state_by_zero_bytes` — 300 ticks with instruments attached vs without: every state array byte-identical (instruments receive numpy copies only; enforced by construction).
- `test_seeded_run_produces_metrics_lineage_and_first_copy` — metrics parquet rows, lineage DB rows, first_copy + seed_placed in events.jsonl.

**Bug the integration test caught (fixed in code):** milestones only saw copies that
completed exactly on sampling ticks; the sampler now accumulates copy events every
tick (host-side only) and delivers them at sample time.

**Simplifications:** mutant-100-generations uses lineage depth of a mutated polymer;
"living lineages" proxied by distinct sequence hashes; energy split (copy vs decay)
estimated from lineage events × copy cost.

## Phase 7 — GUI (2026-09-11)

**Acceptance criteria met** (test names that prove it):
- `tests/test_phase7_gui.py::TestPanelsOnDeadWorld` — 12 tests: run picker (status/touched), meta, all map layers finite on a dead world, inspector cell + 404 polymer, charts metrics, empty lineage tree, event timeline, console confirm-token flow, developer-mode 403 gate, time control, websocket attach→stream→detach with `viewers==0` and `view_cache is None` after (zero residue), gui/ static files served.
- `test_throughput_with_viewer_within_5pct` — measured tps with a live websocket viewer vs detached: < 5% slowdown (view cache built at most 2 Hz, only while viewers > 0).

**Design:** single-page vanilla-JS app (no dependencies, LAN-friendly): run picker landing,
canvas map with layer select/zoom/pan/hover/click-inspect, time panel, inspector with
motif highlighting + lineage path, multi-series charts (log/linear), collapsible lineage
tree, event timeline with map jump, god console with confirm dialog, red developer
toggle + TOUCHED banner, run manager with fork and metrics compare.

**v0 limitation (open question for Operator):** the API serves the sim in its own
process; the run picker lists all runs but attaches only to the running one — attach
to a resumable run by `firmament resume --serve`.
