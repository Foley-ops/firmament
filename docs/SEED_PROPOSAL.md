# SEED PROPOSAL — event #0 (awaiting Operator approval)

*This is the one act of creation. Nothing else living will ever be designed.*

## Proposed sequence (60 monomers)

```
M1M3M2M4 M1M3M2M4 M1M3M2M4 M1M3M2M4 M1M3M2M4 M1M3M2M4 M1M3M2  ← 27 neutral
M1M3M1M2M4M2                                                    ← replicase motif
M1M3M2M4 M1M3M2M4 M1M3M2M4 M1M3M2M4 M1M3M2M4 M1M3M2M4 M1M3M2  ← 27 neutral
```

One line, as the `seed` command takes it:
```
M1M3M2M4M1M3M2M4M1M3M2M4M1M3M2M4M1M3M2M4M1M3M2M4M1M3M2M1M3M1M2M4M2M1M3M2M4M1M3M2M4M1M3M2M4M1M3M2M4M1M3M2M4M1M3M2M4M1M3M2
```

## Design rationale

- **Length 60** → `ε·L = 0.005 × 60 = 0.30`, inside the writeup's target (ε·L < 1, aim ≈ 0.3): each copy carries on average 0.3 mutations — enough exploration, below the error catastrophe.
- **Exactly one motif.** Programmatic scan (rerunnable, see below) confirms the only catalog motif anywhere in the sequence — including across pad/motif junctions — is `replicase` at position 27. No catalyst, no membrane, no motor, no senses: the dumbest possible self-copier.
- **Motif centered** so neither end is special; padding is the repeating unit `M1M3M2M4`, which contains no catalog motif in any 6-window.
- **Copying is antiparallel** (child = reverse complement, like real polymerases). The replicase motif is a reverse-complement palindrome, so function survives every generation; the verification below also scans the first-copy strand and finds only `replicase` at 27.
- **Cost**: one copy = 60 monomers × (1 M + 1 P~P) over 60 ticks (1 sim-hour per copy).

Verification (run any time):
```bash
uv run pytest tests/test_phase5_polymers.py::test_motif_scan_matches_hand_computed -q
```
and the scan used for this proposal is reproduced in `docs/PROGRESS.md` Phase 9 notes.

## Proposed placement

- **World**: `configs/world_small.yaml` (256²) for M0/M1 shakeout, per open question #5.
- **Cell**: the vent pool at (300→scaled) — for world_small, vent at (75,128); place the seed 2 cells off the vent core in the warm shallow rim (e.g. `--cell 77,128`), where: temperature is elevated (fast chemistry, P~P from vent phosphorylation and daylight photophosphorylation), monomer supply is granted via `chemistry.overrides` (M1–M4, P~P, Pi initial concentrations — the M1-protocol environment lever), and hydrolysis is not yet punishing (rim, not core).
- **When**: fork from the M0 end state (M0 = 1000 sim-days dead, all audits green), so the seed enters an equilibrated world.

## What approval means

On your approval I will run, exactly:
```bash
firmament fork --run <M0-run-id>                        # M1 world, M0 lineage preserved
firmament seed --run <fork-id> --sequence <above> --cell 77,128
firmament resume --run <fork-id> --serve --auto-restart # M1: population > 0 for 1000 sim-days
```
If the seed dies: **new run, different environment** (pool temperature via vent flux,
monomer supply via initial concentrations, energy via solar constant), never an edited
seed or physics in a living run. Each attempt logged in `docs/M1_ATTEMPTS.md`; after
10 failures, stop and report.

---

## APPROVED

**Operator approval given 2026-09-11 16:12 CDT** (Nick, in session): seed as proposed,
placement as proposed, M1 to start automatically upon M0 certification. The M1
protocol (environment-only tuning, 10-attempt limit, docs/M1_ATTEMPTS.md) applies.
