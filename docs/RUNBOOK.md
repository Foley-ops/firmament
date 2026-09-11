# RUNBOOK — operating FIRMAMENT

All commands from the repo root. `uv run python -m firmament.cli …` is aliased below as `firmament …`.

## Start a new world
```bash
firmament run --config configs/world_small.yaml            # 256², shakeout
firmament run --config configs/world_default.yaml --serve  # 1024², with GUI API on :8000
```
Creates `runs/<run_id>/` (frozen config, logs, snapshots, metrics, events, lineage).
`--ticks N` bounds the run; omit for indefinite. `--serve --port P` exposes the GUI/API on the LAN.

## Stop
Ctrl-C (or kill). A shutdown snapshot is written on clean exit; an unclean kill loses
at most `snapshot_every_sim_days` of progress (resume from the last snapshot).

## Resume / long-run mode
```bash
firmament resume --run <run_id> --serve --auto-restart
```
`--auto-restart` = crash recovery: on any crash a `crash.json` is written (exception,
tick, last good snapshot) and the run resumes automatically from the last snapshot,
logging the replayed gap. Three consecutive crashes stop it for a human.

## Fork / replay
```bash
firmament fork --run <run_id> [--snapshot tick_000000001000.zarr]
firmament replay --run <run_id> --to-tick 200000
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

## Dead-era acceleration
With zero living polymers the chemistry integrates at 10× dt (logged as a mode
change, replay-exact, pure function of state). It reverts the moment a polymer exists.
