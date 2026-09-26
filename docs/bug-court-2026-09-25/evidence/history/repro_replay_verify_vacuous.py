"""EXHIBIT D: `firmament replay --to-tick T --verify` prints "VERIFY: identical" without
replaying anything whenever the loaded snapshot (the LATEST one by default) is already at
or past T: it re-saves the loaded state and compares it with the very snapshot it was
loaded from. Here the run's tick-100 snapshot is deliberately corrupted, and the
verification of tick 100 still passes.

Sources: cmd_replay docstring "Re-derive a later state from a snapshot + the causal log
... With --verify, compare byte-for-byte against the run's own snapshot at that tick";
RUNBOOK 'Fork / replay' usage `replay --run <id> --to-tick 200000 [--snapshot TICK] [--verify]`."""
import sys

sys.path.insert(0, "/home/nick/Life/firmament")
sys.path.insert(0, "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/history")
import numpy as np  # noqa: E402
import zarr  # noqa: E402

from common import cli, only_run, workdir, write_cfg  # noqa: E402

wd = workdir("D_verify")
cfg = write_cfg(wd, snap_ticks=100)
cli(wd, "run", "--config", str(cfg), "--ticks", "200", check=True)
rid = only_run(wd)
snapdir = wd / "runs" / rid / "snapshots"
print("run --ticks 200 -> snapshots:", sorted(p.name for p in snapdir.iterdir()))

# control: an honest verification from the earlier snapshot works
r = cli(wd, "replay", "--run", rid, "--snapshot", "100", "--to-tick", "200", "--verify")
print("replay --snapshot 100 --to-tick 200 --verify ->", [x for x in r.stdout.splitlines() if x.startswith(("replayed", "VERIFY"))])

# corrupt the run's own tick-100 snapshot (stand-in for any divergence at tick 100)
g = zarr.open_group(str(snapdir / "tick_000000000100.zarr"), mode="r+")
el = g["elevation"][:]
el[3, 3] += 1000.0
g["elevation"][:] = el
print("tick-100 elevation[3,3] corrupted by +1000 m")

for extra in ([], ["--snapshot", "200"]):
    r = cli(wd, "replay", "--run", rid, "--to-tick", "100", "--verify", *extra)
    print("replay --to-tick 100 --verify", " ".join(extra), "-> rc=%d" % r.returncode,
          [x for x in r.stdout.splitlines() if x.startswith(("replayed", "VERIFY"))])

out = wd / "runs" / "_replays" / f"{rid}@100"
print("files written for the '@100' replay:", sorted(p.name for p in out.iterdir()))
a = zarr.open_group(str(snapdir / "tick_000000000100.zarr"), mode="r")["elevation"][:]
b = zarr.open_group(str(out / "tick_000000000200.zarr"), mode="r")["elevation"][:]
print("run's tick-100 elevation == replay output elevation:", np.array_equal(a, b))
