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
