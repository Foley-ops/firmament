"""Repro: the photoactive (light) and sense[A] gates are applied ONLY to the catalyst
effect. configs/genetic_code_v0.yaml documents them as gating OTHER motifs
("photosystem (gates other motifs by light)", "signal receptor (gates other motifs by
[A])"), and k_cell_pass's own header says "gates: photoactive x light, sense x [S]" for
the motif-effect block. In total darkness with [A] = 0, a photoactive+catalyst polymer
is silenced, but photoactive+membrane / sense+membrane / photoactive+emit polymers act
at full strength (and pay full P~P) exactly like their ungated twins.
Polymers-only sim: radiation is off, so light_water == 0 everywhere. Each probe
polymer sits alone in its own cell with identical resources; polymers are assembled
from the cell's own monomers (M -> unit + H2O) so element totals stay exact."""
import sys
sys.path.insert(0, "/home/nick/Life/firmament")
import tempfile
from pathlib import Path
import numpy as np
import warp as wp
import yaml
from tests.conftest import tiny_cfg, build_test_sim, REPO

SCR = "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/life"
code = yaml.safe_load(open(REPO / "configs/genetic_code_v0.yaml"))
print("genetic code says: photoactive ->", "'photosystem (gates other motifs by light)'",
      "; sense_A -> 'signal receptor (gates other motifs by [A])'")
M = {n: "".join(code["motifs"][n]["seq"]) for n in code["motifs"]}
SYM = {"M1": 1, "M2": 2, "M3": 3, "M4": 4}
def parse(s): return np.array([SYM[s[i:i+2]] for i in range(0, len(s), 2)], np.uint8)
SP = "M1M3M2M4"
probes = [
    ("catalyst_formose",               M["catalyst_formose"] + SP),
    ("photoactive + catalyst_formose", M["photoactive"] + SP + M["catalyst_formose"] + SP),
    ("sense_A + catalyst_formose",     M["sense_A"] + SP + M["catalyst_formose"] + SP),
    ("membrane",                       M["membrane"] + SP),
    ("photoactive + membrane",         M["photoactive"] + SP + M["membrane"] + SP),
    ("sense_A + membrane",             M["sense_A"] + SP + M["membrane"] + SP),
    ("emit_A",                         M["emit_A"] + SP),
    ("photoactive + emit_A",           M["photoactive"] + SP + M["emit_A"] + SP),
]
n = 8
cfg = tiny_cfg(n=n, water=0.9, cap=100, seed=11)
state, sched = build_test_sim(cfg, "cpu", Path(tempfile.mkdtemp(dir=SCR)) / "run")
sched.modules = [sched.modules[4]]            # polymers only: no radiation -> dark
poly, idx = sched.poly, sched.chem.index
state.water_depth = wp.array(np.full((n, n), 1.0), dtype=wp.float64, device="cpu")
state.temp = wp.array(np.full((3, n, n), 288.0), dtype=wp.float64, device="cpu")
spec = np.zeros_like(state.species.numpy())
spec[idx["H2O"]] = 1_000_000
for m in ("M1", "M2", "M3", "M4"):
    spec[idx[m]] = 1000
spec[idx["PP"]] = 10000
spec[idx["L"]] = 1000
spec[idx["CH2O"]] = 1000; spec[idx["HCN"]] = 1000        # emit_A (strecker) substrates
arr = {k: getattr(state, k).numpy().copy() for k in
       ("p_state", "p_len", "p_cell", "p_id", "p_partner", "p_child", "p_seq", "p_motifs", "p_parent")}
cells = []
for slot, (name, s) in enumerate(probes):
    q = parse(s); yy, xx = divmod(slot, n)
    yy, xx = 2 * (slot // 4) + 1, 2 * (slot % 4)
    cells.append((yy, xx))
    arr["p_state"][slot], arr["p_len"][slot], arr["p_cell"][slot] = 1, len(q), yy * n + xx
    arr["p_id"][slot], arr["p_partner"][slot], arr["p_child"][slot], arr["p_parent"][slot] = slot + 1, -1, -1, 0
    arr["p_seq"][slot, :] = 0; arr["p_seq"][slot, :len(q)] = q
    arr["p_motifs"][slot] = poly.motif_mask_host(q)
    for m in range(1, 5):
        spec[idx[f"M{m}"], yy, xx] -= int((q == m).sum())
    spec[idx["H2O"], yy, xx] += len(q)
state.next_poly_id = len(probes) + 1
state.species = wp.array(spec, dtype=wp.int32, device="cpu")
for k, a in arr.items():
    setattr(state, k, wp.array(a, dtype=getattr(state, k).dtype, device="cpu"))
names = poly.motif_names
sp0 = state.species.numpy().copy()
TICKS = 10
for _ in range(TICKS):
    sched.tick_once()
sp1 = state.species.numpy(); mem = state.membrane_store.numpy(); cat = state.catalyst.numpy()
lw = state.light_water.numpy()
rx = [r["name"] for r in sched.chem.reactions].index("formose")
print(f"max light_water anywhere = {lw.max()} W/m^2 ;  [A] in every probe cell at start = 0")
print(f"after {TICKS} ticks (alive: {[int(v) for v in state.p_state.numpy()[:len(probes)]]}):")
for (name, s), (yy, xx) in zip(probes, cells):
    mask = poly.motif_mask_host(parse(s))
    print(f"  {name:32s} motifs={[names[b] for b in range(len(names)) if mask >> b & 1]}"
          f"\n      P~P spent={int(sp0[idx['PP'], yy, xx] - sp1[idx['PP'], yy, xx]):3d}"
          f"  L accreted={int(sp0[idx['L'], yy, xx] - sp1[idx['L'], yy, xx]):3d}"
          f"  (membrane_store={int(mem[yy, xx])})"
          f"  A emitted={int(sp1[idx['A'], yy, xx] - sp0[idx['A'], yy, xx]):3d}"
          f"  formose catalyst (last substep)={float(cat[rx, yy, xx]):.1f}")
