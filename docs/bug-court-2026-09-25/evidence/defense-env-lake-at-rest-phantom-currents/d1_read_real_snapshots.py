"""READ-ONLY inspection of the owner's historical M1 genesis run (v0.1; k_momentum and
terrain.py are byte-identical to HEAD). Measures water speed and surface flatness in the
real full-stack run, to test reachability of the phantom-current accusation."""
import sys
sys.path.insert(0, "/home/nick/Life/firmament")
import glob
import numpy as np
import zarr

RUN = "/home/nick/Life/firmament/runs/20260911-200738-genesis-57719a/snapshots"
snaps = sorted(glob.glob(RUN + "/tick_*.zarr"))
for p in snaps[:1] + snaps[1:4] + snaps[-1:]:
    g = zarr.open_group(p, mode="r")
    h = np.asarray(g["water_depth"][...])
    u = np.asarray(g["water_u"][...])
    v = np.asarray(g["water_v"][...])
    el = np.asarray(g["elevation"][...]) + np.asarray(g["sediment"][...])
    wet = h > 1e-4
    sp = np.sqrt(u * u + v * v)
    eta = (el + h)[wet]
    # checkerboard diagnostic: fraction of wet horizontal neighbour pairs with opposite-sign u
    both = wet[:, 1:] & wet[:, :-1] & (np.abs(u[:, 1:]) > 0.1) & (np.abs(u[:, :-1]) > 0.1)
    opp = (np.sign(u[:, 1:]) != np.sign(u[:, :-1])) & both
    tick = p.split("tick_")[1].split(".")[0]
    print(f"tick {int(tick):>7d}: wet {wet.sum():6d}  vol {h.sum():9.0f} m^3  max speed {sp.max():7.3f} m/s  "
          f"cells>0.5 m/s {(sp > 0.5).sum():6d}  mean speed(wet) {sp[wet].mean():.4f}  "
          f"eta spread(wet) {eta.max() - eta.min():.3f} m  opposite-sign u pairs among |u|>0.1: "
          f"{opp.sum()}/{both.sum()}")
