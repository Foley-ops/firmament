# M1 plan — Genesis v0.2 (ready to launch on Operator GPU OK)

The Codex review's control design: environment enrichment and the seed must not be
confounded. Three feasible conditions × 2 RNG seeds, 1000 sim-days each, 256² world:

| condition | config | seed placed? |
|---|---|---|
| baseline, unseeded | `configs/world_m1_baseline.yaml` | no |
| ~~baseline, seeded~~ | — | impossible: the baseline world holds no monomers at dawn, and the seed must be built from its cell's own atoms (conservation). Recorded as a finding, not skipped silently. |
| enriched, unseeded | `configs/world_m1_enriched.yaml` | no |
| enriched, seeded | `configs/world_m1_enriched.yaml` | yes |

RNG seeds: 20260910 and 20260926 (edit `run.seed`). Seeded runs: `firmament seed` at
tick 1 into the nearest wet cell to vent (75,128) that holds enough monomers (the
approved 60-mer). Success = the seeded population never reaches zero for 1000 days
AND differs from its unseeded twin; "never zero" alone is not success.

First, before any of these: GPU half of the replay test and viewer-throughput test,
then a fresh v0.2 M0 (1000 sim-days dead, `world_small.yaml`).

GPU budget: runs are launch-bound and small (~1 GB each), so 4 can share the GPU;
estimate ~8–10 h per batch of 4 at 256².
