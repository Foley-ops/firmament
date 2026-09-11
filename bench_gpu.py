"""Phase 0 benchmark: 1024^2 5-point stencil, Warp vs Taichi. Result goes to docs/DECISIONS.md."""
import time

import numpy as np

N, ITERS = 1024, 500


def bench_warp():
    import warp as wp

    wp.init()

    @wp.kernel
    def stencil(src: wp.array2d(dtype=float), dst: wp.array2d(dtype=float)):
        i, j = wp.tid()
        if 0 < i < N - 1 and 0 < j < N - 1:
            dst[i, j] = 0.2 * (src[i, j] + src[i - 1, j] + src[i + 1, j] + src[i, j - 1] + src[i, j + 1])

    a = wp.array(np.random.default_rng(0).random((N, N), dtype=np.float32), dtype=float)
    b = wp.zeros_like(a)
    for _ in range(5):
        wp.launch(stencil, dim=(N, N), inputs=[a, b])
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        wp.launch(stencil, dim=(N, N), inputs=[a, b])
        a, b = b, a
    wp.synchronize()
    return time.perf_counter() - t0


def bench_taichi():
    import taichi as ti

    ti.init(arch=ti.cuda, log_level=ti.ERROR)
    a = ti.field(ti.f32, shape=(N, N))
    b = ti.field(ti.f32, shape=(N, N))
    a.from_numpy(np.random.default_rng(0).random((N, N), dtype=np.float32))

    @ti.kernel
    def stencil(src: ti.template(), dst: ti.template()):
        for i, j in src:
            if 0 < i < N - 1 and 0 < j < N - 1:
                dst[i, j] = 0.2 * (src[i, j] + src[i - 1, j] + src[i + 1, j] + src[i, j - 1] + src[i, j + 1])

    for _ in range(5):
        stencil(a, b)
    ti.sync()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        stencil(a, b)
        a, b = b, a
    ti.sync()
    return time.perf_counter() - t0


if __name__ == "__main__":
    import sys

    which = sys.argv[1]
    t = bench_warp() if which == "warp" else bench_taichi()
    print(f"{which}: {ITERS} iters in {t:.3f}s = {ITERS / t:.0f} iters/s")
