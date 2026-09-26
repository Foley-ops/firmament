# Decisions

## 2026-09-10 — GPU kernel library: NVIDIA Warp
Benchmarked 1024² 5-point stencil, 500 iters, RTX 4090 (`bench_gpu.py`):
Warp 67,271 iters/s vs Taichi 7,753 iters/s. Both installed cleanly; Warp ~8.7× faster.
**Chosen: warp-lang 1.17.0.** Taichi will be removed. Never mixed.

## 2026-09-10 — Determinism strategy (GPU)
Float atomic adds are scheduling-order-dependent on GPU and would break bit-identical
replay. Rules adopted:
1. All scatter-adds use **integer atomics** (species counts, fixed-point heat deposit).
2. Field kernels are **pure gather** (each cell reads neighbors, writes only itself).
3. Polymer resource competition is resolved **per-cell serially in polymer-id order**
   (cell→polymer index rebuilt each polymer step; in-cell list insertion-sorted by id,
   which is order-independent of the atomic build order).
4. RNG is counter-based (`wp.rand_init(seed, offset)`), keyed by (run_seed ⊕ module
   salt, tick, cell/polymer index). No global RNG anywhere.
Real-world analog: physics is the same regardless of which observer computes it first.

## 2026-09-10 — Precision
Conserved continuous fields (water, vapor, temperature, momenta) are **float64**;
diagnostic fields (light) float32; species counts int32; sequences uint8.
f64 costs ~2× bandwidth on these few fields but keeps the water audit at machine
precision over 10⁶+ ticks. Species integers give exact element conservation.

## 2026-09-10 — Python 3.12
Taichi (needed for the benchmark) has no 3.13 wheels; Warp supports 3.12 fully.

## 2026-09-11 — Solute concentration and the int32 limit
The Phase 5 acceptance run wrapped a cell's H2O species count: water draining into
the vent basin concentrates dissolved species (real: brine pools), and int32 wrapped
at 2^31 — the element audit caught a deficit of exactly 2^32 H2O. Fixes: H2O species
inventory scaled down 10x with k300 rescaled by 10^(H2O stoichiometric order) so
effective kinetics are unchanged; audit tripwire hard-errors if any count exceeds
1.6e9 (precipitation/solubility is a v1 candidate, added — never patched — per
Principle 5).

## 2026-09-25 — Genesis v0.2 (after the M1 attempt-1 post-mortem and the Codex review)
v0.1 is frozen as git tag `genesis-v0.1-historical`; its runs are read-only history
(docs/history/). v0.2 replaces invalid mechanisms in a NEW ruleset — old runs are never
migrated.
- **Errors only during incorporation.** A copy whose next correct monomer or P~P is
  missing stalls with no random draw. Substitutions always pick a wrong base, so the
  realized per-base error rate equals ε. (Real analog: a stalled polymerase makes no
  mistakes; v0.1's free per-tick deletions drove every lineage to length 0.)
- **MIN_LEN = 2.** Fewer than two units has no backbone bond and is not a polymer;
  such fragments dissolve to monomers. (v0.1 allowed zero-atom, immortal, free "polymers".)
- **Polymers move only through water.** (v0.1 diffused them across dry land.)
- **No dead-era law switch.** v0.1 multiplied chemistry rates ×10 while lifeless.
- **Causal vs analysis history.** events.jsonl holds only causal commands (seed, natural
  events, EDITs), committed at tick boundaries with their own sequence numbers; natural-
  event RNG keys use only those. Milestones/novelty go to analysis.jsonl. Snapshots carry
  causal_n + detector state; resume/replay re-apply logged history after the snapshot.
- **Provenance.** Config hash = config values + rule-file CONTENTS; rule files are copied
  into each run; meta.json records git commit, dirty flag, ruleset, library versions.
- **Milestone names.** Causal-sounding detectors are `candidate_*` alarms needing
  confirmation. Individuals = completed polymers only (children under construction excluded).
- **Server.** Binds localhost unless `--lan`; developer edits only with `--developer`.
- **Auto-restart** retries only device faults; deterministic crashes stop for a human.

**Energy economy — Operator chose (a) thermodynamic, 2026-09-25** (chemistry_v0_2.yaml):
- One count = 1 µmol in the cell's 1 m² water column → molar concentrations per cell.
- Reactions may declare `dg0` (kJ/mol at 300 K; van't Hoff via `dh`). Their forward flux
  is multiplied by 1 − exp(ΔG/RT) with ΔG from local concentrations (solvent H2O at
  activity 1), and is zero when uphill. Photochemistry is exempt (photon free energy).
- Loader rule: any non-photochemical reaction that makes the energy carrier (P~P) must
  declare `dg0` — no ungated route to energy can be added by accident.
- P~P: ΔG°' hydrolysis −19 kJ/mol (pyrophosphate). Thermal condensation of P~P in the
  dark is therefore essentially nil; light (photophosphorylation) is the energy source.
- Copy step: M(aq) → chain unit + H2O has ΔG° +21 kJ/mol (phosphodiester bond) − RT ln[M];
  it is paid by hydrolysing `copy_energy_per_monomer` P~P at local [P~P], [Pi], and
  happens only if the total is downhill. Default cost is now **2 P~P per monomer**, as in
  biology (NTP → NMP + PPi, PPi → 2 Pi); with 1, copying at µM monomer is uphill.
- Limits (honest): other reactions remain kinetic-only (phenomenological); ΔG° values
  are textbook approximations, not fitted.

## 2026-09-25 — Time/space units, strict config, run lease
- **dt is a numerical choice.** Chemistry k300 values are per 60 s (`k300_dt_seconds`) and
  scale by dt/60; fluid rain/diffusion/redissolution/vapor-mixing are per-second rates
  (validated against stability limits); polymers run dt/60 sub-steps per tick. So
  `dt_seconds` must be a whole multiple of 60. Tests show chemistry extent and copy
  speed are the same at dt = 60 and 120.
- **cell_meters must be 1.0.** Fluxes, lateral conduction and transport are calibrated for
  1 m cells; any other value is rejected loudly instead of silently changing physics.
- **Strict config:** unknown fields, out-of-range values, vents off-grid, unknown override
  species/keys, unknown rule-file keys → errors.
- **Single-writer lease:** `runs/<id>/.lease` (pid). A live holder blocks other writers;
  a dead holder's lease is reclaimed and logged; the same pid (auto-restart) re-enters.
