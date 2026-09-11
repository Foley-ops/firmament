"""Water cycle: shallow-water solver (CFL sub-stepped), evaporation/rain with latent
heat, vapor mixing, erosion, dissolved-species transport.

Determinism rules (docs/DECISIONS.md): all kernels pure gather; face quantities are
pure functions of both adjacent cells so each side computes the identical value; the
only randomness is counter-based stochastic rounding of integer species moves, keyed
by (tick, face, species). Water mass moves in f64 add/subtract pairs -> exact.
"""
from __future__ import annotations

import math

import numpy as np
import warp as wp

from firmament.core import physconst as pc
from firmament.core.rng import mix, seed32
from firmament.io.logging import get

log = get("fluid")

SALT_TRANSPORT = 9
H_MIN = 1e-4          # m; below this a cell is "dry"
MANNING_N = 0.04      # Manning roughness (natural channels); drag = g n^2 |u| / h^(4/3)
K_EVAP = 3e-6         # kg/m^2/s per kg/m^2 humidity deficit (ocean ~5 mm/day)
RAIN_FRAC = 0.1       # fraction of supersaturation raining out per tick
DIFF_FRAC = 0.01      # per-face species exchange per tick in connected water
ERO_RATE = 1e-6       # sediment m per (m/s)^3 per s
SAT_CAP = 500_000_000  # counts/cell solubility limit; excess precipitates (evaporites)
REDISSOLVE = 0.01     # fraction of the unsaturated deficit redissolving per tick


@wp.func
def qsat(t: wp.float64) -> wp.float64:
    # Tetens saturation vapor pressure -> column kg/m^2 (~41 at 25 C, Earth-like)
    tc = t - wp.float64(273.15)
    es = wp.float64(0.6108) * wp.exp(wp.float64(17.27) * tc / (tc + wp.float64(237.3)))
    return wp.float64(13.0) * es


@wp.func
def face_flux(hA: wp.float64, hB: wp.float64, etaA: wp.float64, etaB: wp.float64,
              vA: wp.float64, vB: wp.float64, dts: wp.float64) -> wp.float64:
    """Signed water volume A->B through a face this substep. Pure in both cells' state."""
    if hA < wp.float64(H_MIN) and hB < wp.float64(H_MIN):
        return wp.float64(0.0)
    v = wp.float64(0.5) * (vA + vB)
    hup = hA
    if v < wp.float64(0.0):
        hup = hB
    f = v * hup * dts
    # cap: no face may drain more than 1/4 of the source cell
    lim = wp.float64(0.25) * hup
    if f > lim:
        f = lim
    if f < -wp.float64(0.25) * hB and v < wp.float64(0.0):
        f = -wp.float64(0.25) * hB
    return f


@wp.kernel
def k_wind(wind_u: wp.array2d(dtype=wp.float32), wind_v: wp.array2d(dtype=wp.float32),
           t: wp.float32, day_s: wp.float32, h: int):
    i, j = wp.tid()
    ph = wp.float32(6.2831853) * wp.float32(i) / wp.float32(h)
    wind_u[i, j] = 1.5 + 1.0 * wp.sin(wp.float32(6.2831853) * t / day_s + ph)
    wind_v[i, j] = 0.5 * wp.cos(wp.float32(6.2831853) * t / day_s + ph)


@wp.kernel
def k_momentum(hw: wp.array2d(dtype=wp.float64), elev: wp.array2d(dtype=wp.float64),
               sed: wp.array2d(dtype=wp.float64), u: wp.array2d(dtype=wp.float64),
               v: wp.array2d(dtype=wp.float64), u_new: wp.array2d(dtype=wp.float64),
               v_new: wp.array2d(dtype=wp.float64), dts: wp.float64, H: int, W: int):
    i, j = wp.tid()
    if hw[i, j] < wp.float64(H_MIN):
        u_new[i, j] = wp.float64(0.0)
        v_new[i, j] = wp.float64(0.0)
        return
    jm = wp.max(j - 1, 0)
    jp = wp.min(j + 1, W - 1)
    im = wp.max(i - 1, 0)
    ip = wp.min(i + 1, H - 1)
    ex = (elev[i, jp] + sed[i, jp] + hw[i, jp] - (elev[i, jm] + sed[i, jm] + hw[i, jm])) / wp.float64(jp - jm)
    ey = (elev[ip, j] + sed[ip, j] + hw[ip, j] - (elev[im, j] + sed[im, j] + hw[im, j])) / wp.float64(ip - im)
    # quadratic Manning bottom friction (real shallow-flow drag), semi-implicit
    hh = wp.max(hw[i, j], wp.float64(0.05))
    sp = wp.sqrt(u[i, j] * u[i, j] + v[i, j] * v[i, j])
    mann = wp.float64(pc.G) * wp.float64(MANNING_N) * wp.float64(MANNING_N)
    drag = wp.float64(1.0) / (wp.float64(1.0) + dts * mann * sp / wp.pow(hh, wp.float64(4.0 / 3.0)))
    u_new[i, j] = (u[i, j] - wp.float64(pc.G) * ex * dts) * drag
    v_new[i, j] = (v[i, j] - wp.float64(pc.G) * ey * dts) * drag


@wp.kernel
def k_height(hw: wp.array2d(dtype=wp.float64), h_new: wp.array2d(dtype=wp.float64),
             elev: wp.array2d(dtype=wp.float64), sed: wp.array2d(dtype=wp.float64),
             u: wp.array2d(dtype=wp.float64), v: wp.array2d(dtype=wp.float64),
             t1: wp.array2d(dtype=wp.float64), t1_new: wp.array2d(dtype=wp.float64),
             fx: wp.array2d(dtype=wp.float64), fy: wp.array2d(dtype=wp.float64),
             dts: wp.float64, H: int, W: int):
    i, j = wp.tid()
    hc = hw[i, j]
    net = wp.float64(0.0)
    e_net = wp.float64(0.0)   # advected thermal energy, J/m^2
    cw = wp.float64(pc.CW_VOL)
    # right face (i,j)->(i,j+1), owned by this cell for the accumulators
    if j + 1 < W:
        f = face_flux(hc, hw[i, j + 1], elev[i, j] + sed[i, j] + hc,
                      elev[i, j + 1] + sed[i, j + 1] + hw[i, j + 1], u[i, j], u[i, j + 1], dts)
        net -= f
        tup = t1[i, j]
        if f < wp.float64(0.0):
            tup = t1[i, j + 1]
        e_net -= cw * f * tup
        fx[i, j] = fx[i, j] + f
    # left face (i,j-1)->(i,j)
    if j - 1 >= 0:
        f = face_flux(hw[i, j - 1], hc, elev[i, j - 1] + sed[i, j - 1] + hw[i, j - 1],
                      elev[i, j] + sed[i, j] + hc, u[i, j - 1], u[i, j], dts)
        net += f
        tup = t1[i, j - 1]
        if f < wp.float64(0.0):
            tup = t1[i, j]
        e_net += cw * f * tup
    # down face (i,j)->(i+1,j)
    if i + 1 < H:
        f = face_flux(hc, hw[i + 1, j], elev[i, j] + sed[i, j] + hc,
                      elev[i + 1, j] + sed[i + 1, j] + hw[i + 1, j], v[i, j], v[i + 1, j], dts)
        net -= f
        tup = t1[i, j]
        if f < wp.float64(0.0):
            tup = t1[i + 1, j]
        e_net -= cw * f * tup
        fy[i, j] = fy[i, j] + f
    # up face (i-1,j)->(i,j)
    if i - 1 >= 0:
        f = face_flux(hw[i - 1, j], hc, elev[i - 1, j] + sed[i - 1, j] + hw[i - 1, j],
                      elev[i, j] + sed[i, j] + hc, v[i - 1, j], v[i, j], dts)
        net += f
        tup = t1[i - 1, j]
        if f < wp.float64(0.0):
            tup = t1[i, j]
        e_net += cw * f * tup
    hn = hc + net
    h_new[i, j] = hn
    c_old = wp.float64(pc.C_SURF_DRY) + cw * hc
    c_new = wp.float64(pc.C_SURF_DRY) + cw * hn
    t1_new[i, j] = (c_old * t1[i, j] + e_net) / c_new


@wp.kernel
def k_evap_rain(hw: wp.array2d(dtype=wp.float64), vapor: wp.array2d(dtype=wp.float64),
                temp: wp.array3d(dtype=wp.float64), dt: wp.float64):
    """Phase change with EXACT energy books: every kg of vapor carries the fixed
    enthalpy LV + CW_SP*T_REF; the sensible deviation stays in the source layer.
    (The surface layer's capacity changes with depth, so the departing/arriving
    water's energy must move explicitly or the audit leaks — it did, once.)"""
    i, j = wp.tid()
    t0 = temp[0, i, j]
    t1 = temp[1, i, j]
    d = hw[i, j]
    cs = wp.float64(pc.C_SURF_DRY)
    cw = wp.float64(pc.CW_VOL)
    h_vap = wp.float64(pc.LV) + wp.float64(pc.CW_SP) * wp.float64(pc.T_REF)  # J/kg
    # evaporation from wet cells, limited by available water and humidity deficit
    if d > wp.float64(H_MIN):
        deficit = qsat(t1) - vapor[i, j]
        if deficit > wp.float64(0.0):
            ev = wp.float64(K_EVAP) * deficit * dt              # kg/m^2
            evmax = d * wp.float64(pc.RHO_W)
            if ev > evmax:
                ev = evmax
            dn = d - ev / wp.float64(pc.RHO_W)
            t1 = (( cs + cw * d) * t1 - ev * h_vap) / (cs + cw * dn)
            d = dn
            hw[i, j] = dn
            vapor[i, j] = vapor[i, j] + ev
            temp[1, i, j] = t1
    # rain where the air column is supersaturated
    ex = vapor[i, j] - qsat(t0)
    if ex > wp.float64(0.0):
        rain = ex * wp.float64(RAIN_FRAC)
        vapor[i, j] = vapor[i, j] - rain
        dn = d + rain / wp.float64(pc.RHO_W)
        # water lands with sensible CW_SP*T_REF; latent heat releases into the air
        sens = rain * wp.float64(pc.CW_SP) * wp.float64(pc.T_REF)
        temp[1, i, j] = ((cs + cw * d) * t1 + sens) / (cs + cw * dn)
        hw[i, j] = dn
        temp[0, i, j] = t0 + rain * wp.float64(pc.LV) / wp.float64(pc.C_AIR)


@wp.kernel
def k_vapor_mix(vapor: wp.array2d(dtype=wp.float64), mean: wp.array(dtype=wp.float64),
                alpha: wp.float64, n: wp.float64):
    i, j = wp.tid()
    vapor[i, j] = vapor[i, j] + alpha * (mean[0] / n - vapor[i, j])


@wp.kernel
def k_row_sum(f: wp.array2d(dtype=wp.float64), out: wp.array(dtype=wp.float64), W: int):
    i = wp.tid()
    s = wp.float64(0.0)
    for j in range(W):
        s += f[i, j]
    out[i] = s


@wp.kernel
def k_row_max_speed(hw: wp.array2d(dtype=wp.float64), u: wp.array2d(dtype=wp.float64),
                    v: wp.array2d(dtype=wp.float64), out: wp.array(dtype=wp.float64), W: int):
    i = wp.tid()
    m = wp.float64(0.0)
    for j in range(W):
        c = wp.sqrt(wp.float64(pc.G) * wp.max(hw[i, j], wp.float64(0.0)))
        s = wp.abs(u[i, j]) + wp.abs(v[i, j]) + c
        if s > m:
            m = s
    out[i] = m


@wp.kernel
def k_erosion(hw: wp.array2d(dtype=wp.float64), u: wp.array2d(dtype=wp.float64),
              v: wp.array2d(dtype=wp.float64), sed: wp.array2d(dtype=wp.float64),
              sed_new: wp.array2d(dtype=wp.float64), dt: wp.float64, H: int, W: int):
    i, j = wp.tid()
    net = wp.float64(0.0)
    for k in range(4):
        di = 0
        dj = 0
        if k == 0:
            dj = 1
        elif k == 1:
            dj = -1
        elif k == 2:
            di = 1
        else:
            di = -1
        ni = i + di
        nj = j + dj
        if 0 <= ni and ni < H and 0 <= nj and nj < W:
            # symmetric pure function: sediment moves along velocity through the face
            vA = wp.float64(0.0)
            if di != 0:
                vA = wp.float64(0.5) * (v[i, j] + v[ni, nj]) * wp.float64(di)
            else:
                vA = wp.float64(0.5) * (u[i, j] + u[ni, nj]) * wp.float64(dj)
            if vA > wp.float64(0.0):   # outflow from (i,j) toward neighbor
                spd = vA
                f = wp.min(wp.float64(ERO_RATE) * spd * spd * spd * dt, wp.float64(0.25) * sed[i, j])
                net -= f
            else:                       # inflow from neighbor
                spd = -vA
                f = wp.min(wp.float64(ERO_RATE) * spd * spd * spd * dt, wp.float64(0.25) * sed[ni, nj])
                net += f
    sed_new[i, j] = sed[i, j] + net


@wp.func
def stoch_round(x: wp.float64, st: wp.uint32) -> int:
    fl = wp.floor(x)
    r = wp.float64(wp.randf(st))
    n = int(fl)
    if r < x - fl:
        n += 1
    return n


@wp.func
def cell_face_out(spec: wp.array3d(dtype=wp.int32), s: int, ci: int, cj: int, face: int,
                  fx: wp.array2d(dtype=wp.float64), fy: wp.array2d(dtype=wp.float64),
                  h0: wp.array2d(dtype=wp.float64), seed: wp.int32,
                  W: int, ns: int) -> int:
    """Capped sequential integer outflow of species s from cell (ci,cj) through `face`
    (0=right,1=left,2=down,3=up). Recomputable identically from either side of the face."""
    n = spec[s, ci, cj]
    hh = h0[ci, cj]
    if n <= 0 or hh < wp.float64(H_MIN):
        return 0
    taken = int(0)
    for k in range(4):
        # signed outward water volume for face k of (ci,cj)
        f = wp.float64(0.0)
        if k == 0 and cj + 1 < W:
            f = fx[ci, cj]
        elif k == 1 and cj - 1 >= 0:
            f = -fx[ci, cj - 1]
        elif k == 2:
            f = fy[ci, cj]
        elif k == 3 and ci - 1 >= 0:
            f = -fy[ci - 1, cj]
        if f > wp.float64(0.0):
            frac = wp.min(f / hh, wp.float64(0.45))
            st = wp.rand_init(seed, wp.int32(((ci * W + cj) * 4 + k) * ns + s))
            amt = stoch_round(wp.float64(n - taken) * frac, st)
            if amt > n - taken:
                amt = n - taken
            if k == face:
                return amt
            taken += amt
    return 0


@wp.kernel
def k_species_advect(spec: wp.array3d(dtype=wp.int32), spec_new: wp.array3d(dtype=wp.int32),
                     fx: wp.array2d(dtype=wp.float64), fy: wp.array2d(dtype=wp.float64),
                     h0: wp.array2d(dtype=wp.float64), seed: wp.int32,
                     H: int, W: int, ns: int):
    s, i, j = wp.tid()
    n = spec[s, i, j]
    out = int(0)
    for k in range(4):
        out += cell_face_out(spec, s, i, j, k, fx, fy, h0, seed, W, ns)
    inn = int(0)
    if j + 1 < W:
        inn += cell_face_out(spec, s, i, j + 1, 1, fx, fy, h0, seed, W, ns)
    if j - 1 >= 0:
        inn += cell_face_out(spec, s, i, j - 1, 0, fx, fy, h0, seed, W, ns)
    if i + 1 < H:
        inn += cell_face_out(spec, s, i + 1, j, 3, fx, fy, h0, seed, W, ns)
    if i - 1 >= 0:
        inn += cell_face_out(spec, s, i - 1, j, 2, fx, fy, h0, seed, W, ns)
    spec_new[s, i, j] = n - out + inn


@wp.func
def diff_pair(spec: wp.array3d(dtype=wp.int32), s: int, ai: int, aj: int, bi: int, bj: int,
              hw: wp.array2d(dtype=wp.float64), seed: wp.int32,
              face_id: wp.int32) -> int:
    """Net signed diffusion A->B (symmetric pure function; one rng draw per face)."""
    if hw[ai, aj] < wp.float64(H_MIN) or hw[bi, bj] < wp.float64(H_MIN):
        return 0
    d = spec[s, ai, aj] - spec[s, bi, bj]
    st = wp.rand_init(seed, face_id)
    x = wp.float64(d) * wp.float64(DIFF_FRAC)
    if x >= wp.float64(0.0):
        return stoch_round(x, st)
    return -stoch_round(-x, st)


@wp.kernel
def k_species_diffuse(spec: wp.array3d(dtype=wp.int32), spec_new: wp.array3d(dtype=wp.int32),
                      hw: wp.array2d(dtype=wp.float64), seed: wp.int32,
                      H: int, W: int, ns: int):
    s, i, j = wp.tid()
    n = spec[s, i, j]
    net = int(0)
    if j + 1 < W:
        net -= diff_pair(spec, s, i, j, i, j + 1, hw, seed, wp.int32(((i * W + j) * 2 + 0) * ns + s))
    if j - 1 >= 0:
        net += diff_pair(spec, s, i, j - 1, i, j, hw, seed, wp.int32(((i * W + j - 1) * 2 + 0) * ns + s))
    if i + 1 < H:
        net -= diff_pair(spec, s, i, j, i + 1, j, hw, seed, wp.int32(((i * W + j) * 2 + 1) * ns + s))
    if i - 1 >= 0:
        net += diff_pair(spec, s, i - 1, j, i, j, hw, seed, wp.int32((((i - 1) * W + j) * 2 + 1) * ns + s))
    spec_new[s, i, j] = n + net


@wp.kernel
def k_precipitate(spec: wp.array3d(dtype=wp.int32), precip: wp.array3d(dtype=wp.int32),
                  hw: wp.array2d(dtype=wp.float64)):
    """Solubility: dissolved counts above SAT_CAP precipitate to an immobile store;
    redissolution when under-saturated and wet. Deterministic integer moves."""
    s, i, j = wp.tid()
    n = spec[s, i, j]
    if n > SAT_CAP:
        ex = n - SAT_CAP
        spec[s, i, j] = SAT_CAP
        precip[s, i, j] = precip[s, i, j] + ex
    elif precip[s, i, j] > 0 and hw[i, j] > wp.float64(H_MIN):
        room = SAT_CAP - n
        back = int(wp.float64(room) * wp.float64(REDISSOLVE))
        if back > precip[s, i, j]:
            back = precip[s, i, j]
        if back > 0:
            precip[s, i, j] = precip[s, i, j] - back
            spec[s, i, j] = n + back


@wp.kernel
def k_copy_t1(t1: wp.array2d(dtype=wp.float64), temp: wp.array3d(dtype=wp.float64)):
    i, j = wp.tid()
    temp[1, i, j] = t1[i, j]


@wp.kernel
def k_get_t1(temp: wp.array3d(dtype=wp.float64), t1: wp.array2d(dtype=wp.float64)):
    i, j = wp.tid()
    t1[i, j] = temp[1, i, j]


class Fluid:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.key = mix(cfg.run.seed, SALT_TRANSPORT)
        self.b = None
        self.last_nsub = 0

    def _bind(self, s):
        h, w = s.shape
        f64 = wp.float64
        dev = s.device
        self.b = {
            "h_new": wp.zeros((h, w), dtype=f64, device=dev),
            "u_new": wp.zeros((h, w), dtype=f64, device=dev),
            "v_new": wp.zeros((h, w), dtype=f64, device=dev),
            "t1": wp.zeros((h, w), dtype=f64, device=dev),
            "t1_new": wp.zeros((h, w), dtype=f64, device=dev),
            "fx": wp.zeros((h, w), dtype=f64, device=dev),
            "fy": wp.zeros((h, w), dtype=f64, device=dev),
            "h0": wp.zeros((h, w), dtype=f64, device=dev),
            "row": wp.zeros(h, dtype=f64, device=dev),
            "mean": wp.zeros(1, dtype=f64, device=dev),
            "spec_new": None,
        }

    def step(self, s, tick: int) -> None:
        if self.b is None:
            self._bind(s)
        b = self.b
        cfg = self.cfg
        h, w = s.shape
        dt = cfg.run.dt_seconds
        day_s = cfg.world.rotation_period_hours * 3600.0
        wp.launch(k_wind, dim=s.shape, inputs=[s.wind_u, s.wind_v, wp.float32((tick * dt) % day_s),
                  wp.float32(day_s), h], device=s.device)
        wp.launch(k_evap_rain, dim=s.shape, inputs=[s.water_depth, s.vapor, s.temp,
                  wp.float64(dt)], device=s.device)

        # CFL substep count from current max signal speed (deterministic row reduction)
        wp.launch(k_row_max_speed, dim=h, inputs=[s.water_depth, s.water_u, s.water_v, b["row"], w],
                  device=s.device)
        cmax = float(np.max(b["row"].numpy()))
        need = int(math.ceil(dt * 1.3 * max(cmax, 0.05) / cfg.world.cell_meters))
        n_sub = max(1, min(800, need))
        if need > 800:
            log.warning("CFL substep cap hit — clamping loudly",
                        extra={"needed": need, "capped": 800})
        if n_sub != self.last_nsub:
            log.info("substep count changed", extra={"n_sub": n_sub, "cmax": round(cmax, 3)})
            self.last_nsub = n_sub
        dts = wp.float64(dt / n_sub)

        wp.copy(b["h0"], s.water_depth)
        b["fx"].zero_()
        b["fy"].zero_()
        wp.launch(k_get_t1, dim=s.shape, inputs=[s.temp, b["t1"]], device=s.device)
        for _ in range(n_sub):
            wp.launch(k_momentum, dim=s.shape, inputs=[s.water_depth, s.elevation, s.sediment,
                      s.water_u, s.water_v, b["u_new"], b["v_new"], dts, h, w], device=s.device)
            s.water_u, b["u_new"] = b["u_new"], s.water_u
            s.water_v, b["v_new"] = b["v_new"], s.water_v
            wp.launch(k_height, dim=s.shape, inputs=[s.water_depth, b["h_new"], s.elevation,
                      s.sediment, s.water_u, s.water_v, b["t1"], b["t1_new"], b["fx"], b["fy"],
                      dts, h, w], device=s.device)
            s.water_depth, b["h_new"] = b["h_new"], s.water_depth
            b["t1"], b["t1_new"] = b["t1_new"], b["t1"]
        wp.launch(k_copy_t1, dim=s.shape, inputs=[b["t1"], s.temp], device=s.device)

        # vapor lateral mixing (turbulent boundary layer analog), exactly conservative
        wp.launch(k_row_sum, dim=h, inputs=[s.vapor, b["row"], w], device=s.device)
        total = float(np.sum(b["row"].numpy()))
        wp.copy(b["mean"], wp.array(np.array([total]), dtype=wp.float64, device=s.device))
        wp.launch(k_vapor_mix, dim=s.shape, inputs=[s.vapor, b["mean"], wp.float64(0.12),
                  wp.float64(h * w)], device=s.device)

        wp.launch(k_erosion, dim=s.shape, inputs=[s.water_depth, s.water_u, s.water_v,
                  s.sediment, b["h_new"], wp.float64(dt), h, w], device=s.device)
        s.sediment, b["h_new"] = b["h_new"], s.sediment

        # dissolved species: advect with accumulated water fluxes, then diffuse
        ns = s.n_species
        if b["spec_new"] is None:
            b["spec_new"] = wp.zeros((ns, h, w), dtype=wp.int32, device=s.device)
        wp.launch(k_species_advect, dim=(ns, h, w), inputs=[s.species, b["spec_new"], b["fx"],
                  b["fy"], b["h0"], wp.int32(seed32(self.key, tick, 0)),
                  h, w, ns], device=s.device)
        s.species, b["spec_new"] = b["spec_new"], s.species
        wp.launch(k_species_diffuse, dim=(ns, h, w), inputs=[s.species, b["spec_new"], s.water_depth,
                  wp.int32(seed32(self.key, tick, 1)), h, w, ns],
                  device=s.device)
        s.species, b["spec_new"] = b["spec_new"], s.species
        wp.launch(k_precipitate, dim=(ns, h, w), inputs=[s.species, s.precipitate, s.water_depth],
                  device=s.device)
