"""Phase 4 acceptance — THE replay test. Run 500 ticks -> snapshot A; run 500 more
-> snapshot B; resume a FRESH sim from A, run 500 -> B'. Every array in B and B'
must match byte-for-byte, on both CPU and GPU backends. This test blocks everything
after it. Plus: fork produces an independent run whose first snapshot equals the
parent's."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from tests.conftest import build_test_sim, tiny_cfg

DEVICES = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])


def state_bytes(state) -> dict[str, bytes]:
    out = {k: v.tobytes() for k, v in state.to_numpy().items()}
    out["__tick__"] = str(state.tick).encode()
    out["__next_poly_id__"] = str(state.next_poly_id).encode()
    out["__solar_mult__"] = repr(state.solar_mult).encode()
    return out


def assert_states_identical(b1: dict, b2: dict):
    assert set(b1) == set(b2), f"field sets differ: {set(b1) ^ set(b2)}"
    bad = [k for k in b1 if b1[k] != b2[k]]
    assert not bad, f"byte mismatch in fields: {bad}"


def run_replay_protocol(tmp_path, device, seeded: bool):
    cfg = tiny_cfg(n=32, cap=5000)
    state, sched = build_test_sim(cfg, device, tmp_path / "runA")
    if seeded:
        from firmament.operator.console import place_seed
        # a replicase in a wet cell (the world_small vent pool region)
        wet = np.argwhere(state.water_depth.numpy() > 0.5)
        y, x = (int(v) for v in wet[len(wet) // 2])
        seq = "M1M3M1M2M4M2" + "M1M3M2M4" * 6          # replicase + neutral padding
        # give the cell monomers to build the seed from
        spec = state.species.numpy()
        for m in ("M1", "M2", "M3", "M4"):
            spec[sched.chem.index[m], y, x] += 200
        state.species = wp.array(spec, dtype=wp.int32, device=state.device)
        place_seed(sched, state, seq, x, y)
    for _ in range(500):
        sched.tick_once()
    snap_a = sched.snapshots.save(state, sched)
    for _ in range(500):
        sched.tick_once()
    b = state_bytes(state)

    # fresh sim, resume from A, run the same 500 ticks
    state2, sched2 = build_test_sim(cfg, device, tmp_path / "runA")  # same dir: same snaps
    sched2.snapshots.load(state2, sched2, tick=500)
    assert state2.tick == 500
    for _ in range(500):
        sched2.tick_once()
    b2 = state_bytes(state2)
    assert_states_identical(b, b2)
    return snap_a


@pytest.mark.parametrize("device", DEVICES)
def test_replay_bit_identical(tmp_path, device):
    run_replay_protocol(tmp_path, device, seeded=False)


def test_fork_first_snapshot_equals_parent(isolated_cwd, device):
    import zarr

    from firmament.io import rundir
    from firmament.io.snapshot import SnapshotManager, fork_run

    cfg = tiny_cfg(n=32, cap=2000)
    run_dir = rundir.create(cfg)
    state, sched = build_test_sim(cfg, device, run_dir)
    for _ in range(50):
        sched.tick_once()
    sched.snapshots.save(state, sched)
    new_dir = fork_run(run_dir)
    meta = rundir.read_meta(new_dir)
    assert meta["parent"] == run_dir.name
    assert meta["run_id"] != run_dir.name              # independent run
    pa = zarr.open_group(str(sorted((run_dir / "snapshots").glob("tick_*.zarr"))[-1]))
    fo = zarr.open_group(str(sorted((new_dir / "snapshots").glob("tick_*.zarr"))[-1]))
    assert set(pa.array_keys()) == set(fo.array_keys())
    for k in pa.array_keys():
        assert np.array_equal(pa[k][:], fo[k][:]), f"fork snapshot differs in {k}"
    # and the fork is loadable into a fresh sim
    state2, sched2 = build_test_sim(cfg, device, new_dir)
    SnapshotManager(new_dir / "snapshots", cfg).load(state2, sched2)
    assert state2.tick == 50
