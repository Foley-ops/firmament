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

## Phase 8 — God console & developer mode (2026-09-11)

**Acceptance criteria met** (test names that prove it):
- `tests/test_phase8_console.py::test_every_event_replays_to_identical_state` — rain, drought, flood, earthquake, volcano, meteor, solar, climate applied live at 8 different ticks; a fresh world replaying ONLY the event log reaches byte-identical state at tick 400.
- `test_solar_multiplier_survives_snapshot` — persistent event state round-trips.
- `TestAddSpecies` — changed rate refused; reordered species refused; pure superset validates (v0 applies additions via fork — documented limitation).
- `TestDeveloperMode::test_edit_marks_touched_and_propagates_to_fork` — first EDIT writes the event, marks meta immediately, and the fork inherits TOUCHED.
- `TestDeveloperMode::test_edit_polymer_rescans_motifs` — state stays coherent after raw edits.
- `test_cli_event_command` — events work from the CLI too (offline path), not just the GUI.

**Bugs the tests caught (fixed in code):**
1. Replay applied events one tick later than live and keyed event RNG off the
   application instant — now RNG keys off the LOGGED tick and replay matches live
   semantics exactly.
2. run_id collision: a fork created in the same second as its parent (same config
   hash) landed in the parent's directory — ids now disambiguate.

## Phase 10 — Long-run operations (2026-09-11)

**Acceptance criteria met** (test names that prove it):
- `tests/test_phase10_ops.py::test_crash_recovery` — injected fault at tick 50 → crash.json (exception, tick, last good snapshot) → resume continues from the snapshot and completes. `resume --auto-restart` loops this automatically (3-crash human stop).
- `test_dead_era_acceleration_engages_and_reverts` — zero polymers → chemistry at 10× dt (logged mode change, pure function of state → replay-exact); reverts the moment life exists.
- `test_daily_report_readable` — `firmament report --run <id>` writes a markdown daily report (population, diversity, milestones, novelty, operator actions, lineage count, disk use).
- `docs/RUNBOOK.md` — start/stop/resume/fork/replay/rotation/disk budget/crash checklist/scaling to world_default.

## Phase 5 world acceptance — 64² replicase world (2026-09-11)

**Life works.** Hand-placed 60-mer replicase seed in the vent pool of `configs/world_test64.yaml`:
- population peaked at **13,165 polymers**, ended 8,041 at tick 60k of metrics — never zero;
- **35,218 polymer ids issued**, 20,000+ copy events on record (lineage DB);
- 6,590 distinct sequences, 6,401 mutant carriers — evolution has raw material;
- first_copy milestone at tick 200; 66 consecutive audits exact.

**Two real bugs this run caught (both fixed and regression-tested):**
1. int32 wrap when brine-pool concentration drove one cell's H2O species over 2³¹ —
   inventory rescaled with kinetics-preserving k300 rescale + audit tripwire (DECISIONS.md).
2. Under investigation at the time of writing: at tick 67,000 the element audit caught
   creation of exactly one L molecule (C6H12O2) — a first-membrane-split-era event.
   The audit did its job (hard stop, run marked); deterministic replay bisection is
   running to pin the responsible module. M1 will not launch until this is fixed.

## L-molecule audit violation — RESOLVED (2026-09-11)

The tick-67,000 creation of one L molecule in the 64² acceptance run is fixed.
Deterministic replay bisection (snapshot 36,000 → per-module element ledger) pinned
it to the polymers module at tick 66,934; an in-tick probe found membrane_store at
cell (17,29) entering the membrane pass at −1 and being silently clamped to 0.

Causal chain: species DIFFUSION was not availability-capped — a cell holding one L
with emptier neighbors could win the stochastic rounding on several faces in one tick
(~1e-8 per cell-tick; billions of opportunities) and go negative; the membrane
effect's `take = min(4, spec_L)` then moved the −1 into the store; the membrane
kernel's `max(0, ·)` clamp created the molecule. Three fixes, all tested:
1. Diffusion outflow is now sequentially availability-capped per source cell,
   symmetric-recomputable (`tests/…::test_diffusion_never_underflows_scarce_species`
   fires the old bug thousands of times and stays exact).
2. The silent membrane clamp is removed — a negative store now reaches the audit.
3. New audit tripwire: any negative species count is a hard error.

Also this session: solubility/precipitation added after the M0 tripwire showed
unbounded brine concentration (evaporite bed, int64, audited, replay-exact).
M0 restarted on the fixed code.


## Phase 7 correction — always-on GUI bridge (2026-09-11)

The Operator flagged that the `--serve` opt-in violated writeup 4.10 (sims run
headless 24/7 and the GUI attaches to ANY of them, never requiring a restart).
Fixed: every sim now always exposes its read-only bridge on a free port recorded in
meta.json (pid liveness-probed); the run picker lists all runs machine-wide and
cross-attaches by redirecting to the target sim's own bridge. Viewer-independent
cost is an idle socket; view work still happens only while viewers > 0.

## M0 CERTIFIED (2026-09-11, 20:03 CDT)

`20260911-070117-shakeout-72d741`: 1000 sim-days (1,440,000 ticks) complete on
world_small. **1,508 consecutive clean conservation audits**; final drift: water
9.0e-16 relative, energy 5.5e-12; elements exact to the integer throughout.
Day/night + seasonal cycles, full water cycle, chemistry equilibria, brine pools
with evaporite deposition — all validated with zero polymers. Baseline metrics on
disk. Mean throughput after CUDA-graph + GPU-isolation fixes: ~47.5 ticks/s.

**Finding:** the certified world is monomer-poor (max 2–3 M per cell planet-wide
after 1000 days) — abiotic nucleotide synthesis is the bottleneck, as in real
origin-of-life chemistry. Consequence: the approved "fork + chemistry.overrides"
seeding plan is mechanically impossible (overrides apply only at world creation).
Per the guide's M1 protocol (monomer supply = initial concentrations ⇒ new run),
**M1 attempt #1 is a fresh world** — identical physics/terrain, enriched initial
monomer/energy pool — with M0 preserved untouched as the certified dead baseline.
