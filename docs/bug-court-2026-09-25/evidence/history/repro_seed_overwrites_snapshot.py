"""Lead probe: `firmament seed` re-saves the snapshot at the SAME tick it loaded, replacing
the pre-seed state at tick T with a post-seed state (causal_n=1). Every other snapshot
at tick T is taken BEFORE the tick-T boundary commands, so `replay --to-tick T --verify`
from an earlier snapshot now reports MISMATCH on an honest history."""
import sys

sys.path.insert(0, "/home/nick/Life/firmament")
sys.path.insert(0, "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/history")
import numpy as np  # noqa: E402
import zarr  # noqa: E402

from common import cli, only_run, workdir, write_cfg  # noqa: E402

OVR = {m: {"init_wet": 400} for m in ("M1", "M2", "M3", "M4")} | {"PP": {"init_wet": 20000}}
wd = workdir("E_seed")
cfg = write_cfg(wd, snap_ticks=100, overrides=OVR, cap=2000)
cli(wd, "run", "--config", str(cfg), "--ticks", "200", check=True)
rid = only_run(wd)
sd = wd / "runs" / rid / "snapshots"
r = cli(wd, "replay", "--run", rid, "--snapshot", "100", "--to-tick", "200", "--verify")
print("before seed: replay 100->200 --verify:", [x for x in r.stdout.splitlines() if x.startswith("VERIFY")])
g = zarr.open_group(str(sd / "tick_000000000200.zarr"), mode="r")
wet = np.argwhere(g["water_depth"][:] > 0.3)[0]
before = g.attrs["meta"]["causal_n"]
cli(wd, "seed", "--run", rid, "--sequence", "M1M3M1M2M4M2" + "M1M3M2M4" * 6,
    "--cell", f"{int(wet[1])},{int(wet[0])}", check=True)
g = zarr.open_group(str(sd / "tick_000000000200.zarr"), mode="r")
print("tick_200 causal_n before/after seed:", before, g.attrs["meta"]["causal_n"],
      "| snapshots:", sorted(p.name for p in sd.iterdir()))
r = cli(wd, "replay", "--run", rid, "--snapshot", "100", "--to-tick", "200", "--verify")
print("after seed:  replay 100->200 --verify:", r.returncode,
      [x for x in r.stdout.splitlines() if x.startswith("VERIFY")])
