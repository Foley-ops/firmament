"""CUDA-graph stale-pointer check, emulated on CPU (no GPU may be used).

fluid.Fluid.step on a device whose name contains "cuda" records the shallow-water
substeps ONCE per n_sub into a CUDA graph and replays it every tick. A CUDA graph
bakes in the kernel arguments (array pointers) captured at record time; fluid.py's
own comment says so ("the substep CUDA graph holds fixed array pointers, so state
arrays it references must never trade places"). console.apply_natural('flood')
REPLACES s.water_depth with a new wp.array (console.py:68).

Emulation: wp.capture_begin/capture_end/capture_launch are replaced by a recorder
that stores each wp.launch(kernel, dim, inputs, device) with the SAME array objects
it was given at record time and re-issues them on capture_launch -- i.e. exactly the
fixed-pointer semantics of a CUDA graph. The device is 'cpu' but its str() contains
"cuda" so fluid takes its graph branch. Validation: before any event the emulated
graph run must be bit-identical to the plain CPU run (fluid.py claims identity)."""
import sys, os, tempfile, math, logging
sys.path.insert(0, "/home/nick/Life/firmament")
from pathlib import Path
import numpy as np
import warp as wp
from tests.conftest import tiny_cfg, build_test_sim
SCR = "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/env"
logging.getLogger("firmament.fluid").setLevel(logging.ERROR)

class EmuCudaDev(str):
    def __str__(self):
        return "cuda:0 (CPU-emulated graph)"

_real_launch = wp.launch
_rec = None
def _launch(kernel, dim, inputs=(), device=None, **kw):
    if _rec is not None:
        _rec.append((kernel, dim, list(inputs), device, kw)); return None
    return _real_launch(kernel, dim=dim, inputs=inputs, device=device, **kw)
def _cap_begin(device=None, **kw):
    global _rec; _rec = []
def _cap_end(device=None, **kw):
    global _rec; g, _rec = _rec, None; return g
def _cap_launch(g, **kw):
    for k, dim, inputs, dev, kw2 in g:
        _real_launch(k, dim=dim, inputs=inputs, device=dev, **kw2)
wp.launch, wp.capture_begin, wp.capture_end, wp.capture_launch = _launch, _cap_begin, _cap_end, _cap_launch

n = 8
RISE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
def run(dev, label):
    from firmament.operator import console
    tmp = Path(tempfile.mkdtemp(dir=SCR)); os.chdir(tmp)
    cfg = tiny_cfg(n=n, cap=200)
    cfg.run.audit_every_ticks = 10
    state, sched = build_test_sim(cfg, dev, tmp / "run")
    # radiation, thermal, fluid, audit (no chemistry/polymers): water physics only
    sched.modules = [sched.modules[0], sched.modules[1], sched.modules[2], sched.modules[5]]
    flu = sched.modules[2].__self__
    # flat-bottomed pond, 1 m deep everywhere, all wet (no shoreline)
    state.elevation = wp.array(np.zeros((n, n)), dtype=wp.float64, device=dev)
    state.sediment = wp.array(np.zeros((n, n)), dtype=wp.float64, device=dev)
    state.water_depth = wp.array(np.full((n, n), 1.0), dtype=wp.float64, device=dev)
    for _ in range(5):
        sched.tick_once()
    pre = state.water_depth.numpy().copy()
    # a flood over the NW corner (radius 2 around (1,1)), +0.5 m, through the real console path
    console.request(sched, "flood", {"cx": 1, "cy": 1, "radius": 2, "rise_m": RISE})
    out = []
    caps_before = set(flu.graphs)
    try:
        for t in range(40):
            h_before = state.water_depth.numpy().copy()
            sched.tick_once()
            if t < 4 or t in (9, 39):
                stale = flu.last_nsub in caps_before
                print(f"[{label}] tick {state.tick}: n_sub {flu.last_nsub} graph-recorded-before-flood={stale}  "
                      f"max|change of s.water_depth this tick| {np.abs(state.water_depth.numpy()-h_before).max():.3e} m")
            if t in (0, 9, 39):
                h = state.water_depth.numpy()
                out.append((state.tick, h.max() - h.min(), h.copy()))
    except Exception as e:
        print(f"[{label}] {type(e).__name__} at tick {state.tick}: {e}")
    print(f"[{label}] graph path used: {bool(flu.graphs)} (graphs cached for n_sub={sorted(flu.graphs)})")
    for tk, spread, _ in out:
        print(f"[{label}] tick {tk}: water-depth spread max-min = {spread:.6f} m")
    return pre, out, state

preA, outA, sA = run("cpu", "plain CPU path ")
preB, outB, sB = run(EmuCudaDev("cpu"), "graph path (emu)")
print("pre-flood depth bit-identical between paths:", np.array_equal(preA, preB))
for (tk, _, hA), (_, _, hB) in zip(outA, outB):
    print(f"tick {tk}: max |depth_plain - depth_graph| = {np.abs(hA - hB).max():.6f} m")
print("graph-path depth field at end (NW corner flood mound should have spread):")
print(np.round(outB[-1][2], 3))
