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

**Open (Operator decision):** the energy economy. At ambient temperature the P~P
pool sits near a thermal equilibrium, and copying spends any P~P regardless of its
chemical potential, so work can be drawn from a single-temperature heat bath. Options:
(a) thermodynamic — couple copy steps to ΔG of P~P hydrolysis at local concentrations;
(b) phenomenological — keep kinetics, narrow claims to ledger conservation, and lower
phosphate to realistic levels so vents/light dominate. Not changed in v0.2.
