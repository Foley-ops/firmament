# M1 attempts log

Protocol (build guide Phase 9): if the seed dies, the response is a NEW RUN with a
different *environment* — pool temperature (vent flux/location), monomer supply
(initial concentrations), energy production (solar constant / vent chemistry), dt.
Never the seed, never the physics of a living run. After 10 failed attempts: stop
and report.

| # | run_id | environment change vs previous | population lasted | outcome |
|---|--------|-------------------------------|-------------------|---------|
| 0a | 20260911-200444-genesis | fresh world + monomer/PP overrides, 3-day dead equilibration | seed never placed | FALSE START: monomer half-life ~5.5 h ate 99.99% of the stock before seeding. Lesson: seed at world-dawn. |
| 1 | 20260911-200738-genesis-57719a | same overrides, seeded at tick 1; cell (83,128) — nearest wet, stocked vent-rim cell ((77,128) is dry at dawn) | IN PROGRESS | first_copy at tick 100; running toward the 1000-sim-day M1 bar |

**Fork note (Operator asked to fork):** the M0 fork path proved impossible — the
certified world is monomer-poor (2–3 M/cell max) and `chemistry.overrides` apply
only at world creation, never to forks. Per the guide's M1 protocol ("monomer
supply = initial concentrations"), attempt 1 is a fresh world; M0 remains untouched
as the certified baseline. Both remain viewable side by side in the run picker.
