# M1 attempts log

Protocol (build guide Phase 9): if the seed dies, the response is a NEW RUN with a
different *environment* — pool temperature (vent flux/location), monomer supply
(initial concentrations), energy production (solar constant / vent chemistry), dt.
Never the seed, never the physics of a living run. After 10 failed attempts: stop
and report.

| # | run_id | environment change vs previous | population lasted | outcome |
|---|--------|-------------------------------|-------------------|---------|
| 0a | 20260911-200444-genesis | fresh world + monomer/PP overrides, 3-day dead equilibration | seed never placed | FALSE START: monomer half-life ~5.5 h ate 99.99% of the stock before seeding. Lesson: seed at world-dawn. |
| 1 | 20260911-200738-genesis-57719a | same overrides, seeded at tick 1; cell (83,128) — nearest wet, stocked vent-rim cell ((77,128) is dry at dawn) | halted day 388 (cap) | **INVALID — simulator artifact.** Population hit the 1M polymer cap; ≥96% were zero-length "polymers". See post-mortem below. |

**Fork note (Operator asked to fork):** the M0 fork path proved impossible — the
certified world is monomer-poor (2–3 M/cell max) and `chemistry.overrides` apply
only at world creation, never to forks. Per the guide's M1 protocol ("monomer
supply = initial concentrations"), attempt 1 is a fresh world; M0 remains untouched
as the certified baseline. Both remain viewable side by side in the run picker.


## Attempt 1 post-mortem (2026-09-25)

Not an extinction and not a success: the run is invalid as evidence of evolution.
- Monomers ran out planet-wide by day ~30. Copies then stalled almost permanently.
- **Kernel bug:** mutation draws happen every *tick*, not per monomer incorporated, so a
  stalled copy keeps rolling deletions that advance for free. Lengths collapsed
  48 → 31 (d42) → 6 (d150) → 1 (d300) → 0.
- **Physics hole:** zero-length polymers are allowed. They hold no atoms (invisible to
  the audit), never hydrolyze (rate ∝ length), and are copied at zero cost. They were
  ≥96.5% of the 741k population at the last sample and filled the 1M cap at day 388.
- **Instrument bugs:** predation/aggregation milestones fired on ordinary template
  copying (p_partner is shared); "fixation" was the empty sequence; milestones re-fire
  after every restart; population counts include unfinished children.
- Genuine: first_compartment (d137.5) came from a real membrane-motif family, short-lived.
- Auto-restart replayed the deterministic cap crash 3 times.

Per build guide non-negotiable #9: stopped and reported. No fixes applied; fixes change
the polymer rules and need Operator approval, then fresh M0/M1 runs.
