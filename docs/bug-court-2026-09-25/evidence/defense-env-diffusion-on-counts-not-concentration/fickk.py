import warp as wp
from firmament.core.fluid import stoch_round

@wp.func
def fick_out(spec: wp.array3d(dtype=wp.int32), s: int, ci: int, cj: int, face: int,
             hw: wp.array2d(dtype=wp.float64), seed: wp.int32,
             H: int, W: int, ns: int, diff_frac: wp.float64) -> int:
    n = spec[s, ci, cj]
    ha = hw[ci, cj]
    if n <= 0 or ha < wp.float64(1.0e-4):
        return 0
    taken = int(0)
    for k in range(4):
        ni = ci
        nj = cj
        if k == 0:
            nj = cj + 1
        elif k == 1:
            nj = cj - 1
        elif k == 2:
            ni = ci + 1
        else:
            ni = ci - 1
        if ni < 0 or nj < 0 or ni >= H or nj >= W:
            continue
        hb = hw[ni, nj]
        if hb < wp.float64(1.0e-4):
            continue
        dc = wp.float64(n) / ha - wp.float64(spec[s, ni, nj]) / hb
        if dc <= wp.float64(0.0):
            continue
        st = wp.rand_init(seed, wp.int32(((ci * W + cj) * 4 + k) * ns + s))
        amt = stoch_round(dc * wp.min(ha, hb) * diff_frac, st)
        if amt > n - taken:
            amt = n - taken
        if k == face:
            return amt
        taken += amt
    return 0


@wp.kernel
def k_fick(spec: wp.array3d(dtype=wp.int32), spec_new: wp.array3d(dtype=wp.int32),
           hw: wp.array2d(dtype=wp.float64), seed: wp.int32,
           H: int, W: int, ns: int, diff_frac: wp.float64):
    s, i, j = wp.tid()
    n = spec[s, i, j]
    out = int(0)
    for k in range(4):
        out += fick_out(spec, s, i, j, k, hw, seed, H, W, ns, diff_frac)
    inn = int(0)
    if j + 1 < W:
        inn += fick_out(spec, s, i, j + 1, 1, hw, seed, H, W, ns, diff_frac)
    if j - 1 >= 0:
        inn += fick_out(spec, s, i, j - 1, 0, hw, seed, H, W, ns, diff_frac)
    if i + 1 < H:
        inn += fick_out(spec, s, i + 1, j, 3, hw, seed, H, W, ns, diff_frac)
    if i - 1 >= 0:
        inn += fick_out(spec, s, i - 1, j, 2, hw, seed, H, W, ns, diff_frac)
    spec_new[s, i, j] = n - out + inn


