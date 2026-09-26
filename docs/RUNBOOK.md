# RUNBOOK — operating FIRMAMENT

All commands from the repo root. `uv run python -m firmament.cli …` is aliased below as `firmament …`.

## Start a new world
```bash
firmament run --config configs/world_small.yaml            # 256², shakeout
firmament run --config configs/world_default.yaml          # 1024²
```
Creates `runs/<run_id>/` (frozen config, logs, snapshots, metrics, events, lineage).
`--ticks N` bounds the run; omit for indefinite. **Every sim always exposes its GUI
bridge** (preferred `--port`, else a free port, recorded in `meta.json`) — on
**localhost only** unless started with `--lan`. Developer edits exist only when the
sim is started with `--developer`; the network can never switch them on.

Only one process may write a run at a time (`runs/<id>/.lease`); a second writer is
refused while the first is alive. `dt_seconds` must be a multiple of 60.

## Stop
Ctrl-C or `kill` (SIGTERM) = graceful: the current tick finishes, then a shutdown snapshot
is written and all buffered metrics/lineage/analysis are flushed. A second Ctrl-C forces
an immediate exit. `kill -9` loses only what resume regenerates deterministically.

## Resume / long-run mode
```bash
firmament resume --run <run_id> --auto-restart
```
Resume = load the latest snapshot, restore instrument state, drop derived records newer
than it (they are regenerated identically), and replay every causal command logged
after it. The result is indistinguishable from never having stopped.
`--auto-restart` re-execs a fresh process only for device faults (Warp/CUDA errors);
deterministic crashes (AuditError, capacity) would reproduce exactly, so they stop for a
human. `crash.json` records exception, tick and last good snapshot.

## Fork / replay
```bash
firmament fork --run <run_id> [--snapshot tick_000000001000.zarr]
firmament replay --run <run_id> --to-tick 200000 [--snapshot TICK] [--verify]
```
Fork = new run_id + parent pointer; first snapshot equals the parent's; the TOUCHED
mark propagates. Replay re-applies the event log from a snapshot and must reproduce
later state byte-for-byte (CI: `tests/test_phase4_replay.py`).

## Seed placement (event #0 — requires Operator approval of docs/SEED_PROPOSAL.md)
```bash
firmament seed --run <run_id> --sequence M1M3M1M2M4M2… --cell x,y
```

## Natural events from the CLI (GUI console is the live path)
```bash
firmament event --run <run_id> --type rain --params '{"intensity_mm":10,"radius":60}'
# parameters are validated first; if logged history after the snapshot is pending it
# replays first, then the event applies — the printed tick is the real one
```

## Daily report
```bash
firmament report --run <run_id>       # writes runs/<id>/reports/YYYY-MM-DD.md
```
Nightly cron: `0 6 * * * cd <repo> && uv run python -m firmament.cli report --run <id>`

## Logs and rotation
- `logs/sim.log` — JSONL, rotates at 64 MB × 10 files. Grep by `run_id` and `tick`.
- `events.jsonl`, `metrics/*.parquet`, `lineage.sqlite` — **never rotated away**.
- Log level at runtime: `POST /api/loglevel {"level":"DEBUG"}`.

## Disk budget
Snapshot size ≈ state size (≈ fields + polymer arrays). At 1024²/10 M polymers a
snapshot is ~3.5 GB; at `snapshot_every_sim_days: 30` budget ~100 GB/sim-year and
prune old snapshots by hand (keep the fork points). 256² worlds are ~40 MB/snapshot.

## Crash checklist
1. Read `runs/<id>/crash.json` (exception, tick, last good snapshot).
2. `firmament resume --run <id> --auto-restart` continues from the snapshot.
3. If the crash is an **AuditError**: do not resume blindly — conservation broke or a
   tripwire fired (e.g. solubility overflow). That is a world-model problem for the
   Operator, not a restart problem.

## Moving to world_default.yaml (1024²)
Dead worlds transfer by starting fresh (terrain is config-determined). A living run
cannot be resized in v0 (`add_land` is a documented refusal) — start the big world
before seeding. Expect ~50 ticks/s with an active water solver on the RTX 4090.

## Logs
`events.jsonl` = causal history (seed, natural events, EDITs) — the run's definition.
`analysis.jsonl` = milestones/novelty — derived, never read by physics. Replay writes only
to `runs/_replays/`; `--verify` compares against the run's own snapshot byte-for-byte.
(The v0.1 "dead-era acceleration" was removed: it changed chemistry rates while lifeless.)
