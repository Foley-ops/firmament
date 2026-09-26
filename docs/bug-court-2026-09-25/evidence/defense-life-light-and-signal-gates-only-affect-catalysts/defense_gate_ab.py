"""Defense's independent A/B check. Each probe polymer is placed through the real console
path (place_seed -> causal commit -> apply_seed) into its own fresh polymers-only world.
Every probe runs twice: gate ON (light_water = 300 W/m^2, [A] = 1e6 -> gate ~ 1) and
gate OFF (light_water = 0, [A] = 0 -> gate = 0). If the gate were applied to all other
motifs, every effect would differ between the ON and OFF columns."""
import sys
sys.path.insert(0, "/home/nick/Life/firmament")
import tempfile
from pathlib import Path
import numpy as np
import warp as wp
import yaml
from tests.conftest import tiny_cfg, build_test_sim, REPO

SCR = Path("/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/"
           "court/defense-life-light-and-signal-gates-only-affect-catalysts")
code = yaml.safe_load(open(REPO / "configs/genetic_code_v0.yaml"))
M = {k: "".join(v["seq"]) for k, v in code["motifs"].items()}
PAD = "M1M3M2M4"
n = 8
Y = X = 4


def run(seq, gate_on, ticks=5):
    cfg = tiny_cfg(n=n, water=0.9, cap=200, seed=99)
    state, sched = build_test_sim(cfg, "cpu", Path(tempfile.mkdtemp(dir=SCR)) / "run")
    sched.modules = [sched.modules[4]]                 # polymers only
    idx = sched.chem.index
    state.water_depth = wp.array(np.full((n, n), 1.0), dtype=wp.float64, device="cpu")
    state.temp = wp.array(np.full((3, n, n), 288.0), dtype=wp.float64, device="cpu")
    state.light_water = wp.array(np.full((n, n), 300.0 if gate_on else 0.0, np.float32),
                                 dtype=wp.float32, device="cpu")
    spec = np.zeros_like(state.species.numpy())
    spec[idx["H2O"]] = 1_000_000
    for m in ("M1", "M2", "M3", "M4"):
        spec[idx[m], Y, X] = 500
    spec[idx["PP"], Y, X] = 10_000
    spec[idx["PP"], Y - 1, X] = 50_000                 # P~P gradient for the motor
    spec[idx["L"], Y, X] = 1000
    spec[idx["CH2O"], Y, X] = 1000
    spec[idx["HCN"], Y, X] = 1000
    spec[idx["A"], Y, X] = 1_000_000 if gate_on else 0
    state.species = wp.array(spec, dtype=wp.int32, device="cpu")
    from firmament.operator.console import place_seed
    place_seed(sched, state, seq, X, Y)               # real console path (event #0)
    sp0 = state.species.numpy().copy()
    rx = [r["name"] for r in sched.chem.reactions].index("formose")
    motor_staged = []
    for _ in range(ticks):
        sched.tick_once()
        motor_staged.append(int(state.p_motor.numpy()[0]))
    sp1 = state.species.numpy()
    lw = float(state.light_water.numpy()[Y, X])
    return dict(
        light=lw, A0=int(sp0[idx["A"], Y, X]),
        pp=int(sp0[idx["PP"], Y, X] - sp1[idx["PP"], Y, X]),
        L=int(state.membrane_store.numpy()[Y, X]),
        A=int(sp1[idx["A"], Y, X] - sp0[idx["A"], Y, X]),
        cat=float(state.catalyst.numpy()[rx, Y, X]),
        cell=int(state.p_cell.numpy()[0]), motor=motor_staged[-1],
        alive=int(state.p_state.numpy()[0]),
        motifs=[sched.poly.motif_names[b] for b in range(sched.poly.nm)
                if int(state.p_motifs.numpy()[0]) >> b & 1])


probes = [
    ("photoactive+catalyst_formose", M["photoactive"] + PAD + M["catalyst_formose"] + PAD),
    ("sense_A+catalyst_formose", M["sense_A"] + PAD + M["catalyst_formose"] + PAD),
    ("photoactive+membrane", M["photoactive"] + PAD + M["membrane"] + PAD),
    ("sense_A+membrane", M["sense_A"] + PAD + M["membrane"] + PAD),
    ("photoactive+emit_A", M["photoactive"] + PAD + M["emit_A"] + PAD),
    ("photoactive+motor", M["photoactive"] + PAD + M["motor"] + PAD),
]
print(f"start cell of each probe = {Y * n + X}; motor target up-gradient = {(Y - 1) * n + X}")
for name, seq in probes:
    on, off = run(seq, True), run(seq, False)
    print(f"{name:30s} motifs={on['motifs']}")
    for tag, r in (("gate ON ", on), ("gate OFF", off)):
        print(f"   {tag} light={r['light']:5.0f} A0={r['A0']:7d} | PP spent={r['pp']:3d} "
              f"cat(formose)={r['cat']:.3f} membrane_store={r['L']:3d} A emitted={r['A']:3d} "
              f"p_motor(last)={r['motor']:3d} cell={r['cell']} alive={r['alive']}")
