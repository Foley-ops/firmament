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
