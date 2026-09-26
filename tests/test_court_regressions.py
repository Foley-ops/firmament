"""Regression tests for the bug court of 2026-09-25 (docs/bug-court-2026-09-25/).

Each test encodes the behavior the court established as correct and FAILED on the code
the bugs were proven against (fc90e3d) before the fix. Numbers refer to VERDICTS.md."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import pytest
import warp as wp
import yaml

from tests.conftest import REPO, build_test_sim, tiny_cfg
from tests.test_phase5_polymers import PAD, REPLICASE, SEED_SEQ, _completed, _life_sim

ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(REPO)}


def _set(state, name, arr):
    dst = getattr(state, name)
    wp.copy(dst, wp.array(np.ascontiguousarray(arr), dtype=dst.dtype, device=state.device))


def _fluid_only(tmp_path, device, n=8):
    cfg = tiny_cfg(n=n, water=0.5, cap=10)
    state, sched = build_test_sim(cfg, device, tmp_path / "pond")
    sched.modules = [sched.modules[2]]
    _set(state, "sediment", np.zeros((n, n)))
    _set(state, "water_u", np.zeros((n, n)))
    _set(state, "water_v", np.zeros((n, n)))
    _set(state, "vapor", np.zeros((n, n)))
    _set(state, "temp", np.full((3, n, n), 250.0))          # cold: no evaporation dynamics
    return cfg, state, sched


# ------------------------------------------------------------- 4 (critical)
def test_lake_at_rest_with_a_dry_bank_stays_at_rest(tmp_path, device):
    n = 8
    cfg, state, sched = _fluid_only(tmp_path, device, n)
    elev = np.zeros((n, n))
    elev[:, n - 1] = 2.0                                    # dry bank above the surface
    depth = np.where(elev > 0, 0.0, 1.0)
    _set(state, "elevation", elev)
    _set(state, "water_depth", depth)
    for _ in range(30):
        sched.tick_once()
    speed = max(np.abs(state.water_u.numpy()).max(), np.abs(state.water_v.numpy()).max())
    assert speed < 1e-3, f"still lake developed {speed:.3f} m/s currents"


# ------------------------------------------------------------- 5
def test_uniform_solution_stays_uniform_across_depths(tmp_path, device):
    n = 8
    cfg, state, sched = _fluid_only(tmp_path, device, n)
    elev = np.zeros((n, n))
    elev[:, n // 2:] = 1.5                                  # shallow half, flat surface at 2 m
    depth = 2.0 - elev
    _set(state, "elevation", elev)
    _set(state, "water_depth", depth)
    ipi = sched.chem.index["Pi"]
    spec = np.zeros_like(state.species.numpy())
    spec[ipi] = (4000 * depth).astype(np.int32)            # equal CONCENTRATION everywhere
    _set(state, "species", spec)
    for _ in range(1500):
        sched.tick_once()
    pi = state.species.numpy()[ipi].astype(float)
    h = state.water_depth.numpy()
    ratio = (pi[:, n // 2:] / h[:, n // 2:]).mean() / (pi[:, : n // 2] / h[:, : n // 2]).mean()
    assert abs(ratio - 1.0) < 0.05, f"shallow/deep concentration ratio drifted to {ratio:.3f}"


# ------------------------------------------------------------- 3 / 11, and volcano lead
def _audited(tmp_path, device):
    cfg = tiny_cfg(n=16, cap=200)
    cfg.run.audit_every_ticks = 5
    return build_test_sim(cfg, device, tmp_path / "w")


@pytest.mark.parametrize("event, params", [
    ("meteor", {"energy_J": 1e9, "crater_m": 0.5, "dust": 0.2, "cx": 8, "cy": 8}),
    ("volcano", {"mineral_counts": 1001, "radius": 3, "heat_J_m2": 1e6, "cx": 8, "cy": 8}),
])
def test_natural_events_keep_the_books(tmp_path, device, event, params):
    from firmament.operator import console
    state, sched = _audited(tmp_path, device)
    for _ in range(6):
        sched.tick_once()
    console.request(sched, event, params)
    for _ in range(20):
        sched.tick_once()                                   # AuditError = failure


# ------------------------------------------------------------- 7 / 10
@pytest.mark.parametrize("event, params", [
    ("rain", {"intensity_mm": None}),                       # GUI "10mm" -> NaN -> null
    ("rain", {"intensity_mm": 5.0, "typo_key": 1}),
    ("drought", {"strength": 1.5}),                         # would drive vapor negative
    ("add_land", {"cols": 8}),                              # documented refusal
])
def test_invalid_commands_are_rejected_before_the_log(tmp_path, device, event, params):
    from firmament.operator import console
    cfg = tiny_cfg(n=16, cap=100)
    state, sched = build_test_sim(cfg, device, tmp_path / "w")
    with pytest.raises(ValueError):
        console.request(sched, event, params)
    sched.tick_once()
    assert sched.events.read_all() == [] and state.causal_n == 0


def test_a_command_that_fails_while_applying_is_never_logged(tmp_path, device, monkeypatch):
    from firmament.operator import causal, console
    cfg = tiny_cfg(n=16, cap=100)
    state, sched = build_test_sim(cfg, device, tmp_path / "w")

    def boom(*a, **k):
        raise RuntimeError("apply failed")
    monkeypatch.setattr(console, "apply_natural", boom)
    with pytest.raises(RuntimeError):
        causal.commit_now(sched, "rain", {"intensity_mm": 5.0})
    assert sched.events.read_all() == [] and state.causal_n == 0


def test_api_rejects_a_malformed_console_command(tmp_path, device):
    from fastapi.testclient import TestClient

    from firmament.server.api import make_app
    cfg = tiny_cfg(n=16, cap=100)
    state, sched = build_test_sim(cfg, device, tmp_path / "w")
    (tmp_path / "w" / "meta.json").write_text(json.dumps(
        {"run_id": "w", "parent": None, "created": "x", "touched": False}))
    c = TestClient(make_app(sched, tmp_path / "w"))
    tok = c.post("/api/console/rain", json={"intensity_mm": None}).json()["confirm_token"]
    r = c.post("/api/console/rain", json={"intensity_mm": None, "confirm_token": tok})
    assert r.status_code == 400 and not sched.mailbox


# ------------------------------------------------------------- 9 (seed)
@pytest.mark.parametrize("seq", ["M1", REPLICASE + "M1M2" * 140])   # 1-mer, 286-mer
def test_bad_seed_is_rejected_before_the_log(tmp_path, device, seq):
    from firmament.operator.console import place_seed
    cfg2 = tiny_cfg(n=16, cap=100)
    state2, sched2 = build_test_sim(cfg2, device, tmp_path / "b")
    spec = state2.species.numpy()
    for m in ("M1", "M2", "M3", "M4"):
        spec[sched2.chem.index[m], 4, 4] = 5000
    _set(state2, "species", spec)
    with pytest.raises(ValueError):
        place_seed(sched2, state2, seq, 4, 4)
    assert sched2.events.read_all() == []
    place_seed(sched2, state2, SEED_SEQ, 4, 4)              # a valid seed still works
    assert [r["event"] for r in sched2.events.read_all()] == ["seed_placed"]


# ------------------------------------------------------------- 8
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_stop_signal_writes_shutdown_snapshot_and_flushes(isolated_cwd, sig):
    cfg = tiny_cfg(n=16, cap=200, log_level="INFO")
    cfg.run.metrics_every_ticks = 20
    (isolated_cwd / "c.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    p = subprocess.Popen([sys.executable, "-m", "firmament.cli", "--device", "cpu", "run",
                          "--config", "c.yaml"], cwd=isolated_cwd, env=ENV,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 120
    while time.time() < deadline:
        logs = list(isolated_cwd.glob("runs/*/logs/sim.log"))
        if logs and "api serving" in logs[0].read_text():
            break
        time.sleep(0.2)
    time.sleep(4)
    p.send_signal(sig)
    rc = p.wait(timeout=120)
    run = next(isolated_cwd.glob("runs/*"))
    assert rc == 0, f"exit code {rc}"
    snaps = list((run / "snapshots").glob("tick_*.zarr"))
    assert snaps, "no shutdown snapshot"
    assert list((run / "metrics").glob("part_*.parquet")), "buffered metrics lost"


# ------------------------------------------------------------- 9 (offline event)
def test_offline_event_waits_for_pending_history_then_applies(isolated_cwd, device):
    from firmament import cli
    from firmament.io import rundir
    (isolated_cwd / "c.yaml").write_text(yaml.safe_dump(tiny_cfg(n=16, cap=100).model_dump(mode="json")))
    cli.main(["--device", device, "run", "--config", "c.yaml", "--ticks", "5"])
    rid = rundir.list_runs()[0]["run_id"]
    # a live process committed a rain at tick 7 and was killed before its next snapshot
    with open(f"runs/{rid}/events.jsonl", "a") as f:
        f.write(json.dumps({"run_id": rid, "tick": 7, "sim_time": "", "component": "operator",
                            "level": "INFO", "event": "rain", "n": 0,
                            "params": {"intensity_mm": 5.0}}) + "\n")
    cli.main(["--device", device, "event", "--run", rid, "--type", "drought",
              "--params", '{"strength": 0.3}'])
    events = [json.loads(x)["event"] for x in open(f"runs/{rid}/events.jsonl") if x.strip()]
    assert events == ["rain", "drought"]
    import zarr
    last = sorted((rundir.RUNS / rid / "snapshots").glob("tick_*.zarr"))[-1]
    assert zarr.open_group(str(last), mode="r").attrs["meta"]["causal_n"] == 2


# ------------------------------------------------------------- 12
def test_lineage_is_readable_live_and_flushed_with_snapshots(tmp_path, device):
    import sqlite3
    cfg, state, sched, _ = _life_sim(tmp_path, device)
    for _ in range(200):
        sched.tick_once()
    ids = state.p_id.numpy()
    kids = [int(ids[c]) for c in _completed(state) if int(ids[c]) > 1]
    assert kids
    path = sched.lineage.path_to_seed(max(kids))
    assert path[-1] == 1 and len(path) >= 2, f"live lineage path {path}"
    sched.snapshots.save(state, sched)
    rows = sqlite3.connect(tmp_path / "run" / "lineage.sqlite").execute(
        "SELECT COUNT(*) FROM copies").fetchone()[0]
    assert rows > 0, "lineage not on disk after a snapshot"


# ------------------------------------------------------------- 1
def test_binder_links_are_always_mutual_and_alive(tmp_path, device):
    binder = "M3M1M3M4M2M4"
    cfg, state, sched, _ = _life_sim(tmp_path, device, seq=REPLICASE + binder + PAD,
                                     pp=400000, monomers=6000)
    bad = []
    for t in range(600):
        sched.tick_once()
        if t % 10:
            continue
        st, pr, cell = state.p_state.numpy(), state.p_partner.numpy(), state.p_cell.numpy()
        for p in np.nonzero((st >= 1) & (st <= 3) & (pr >= 0))[0]:
            q = int(pr[p])
            if q != p and not (1 <= st[q] <= 3 and pr[q] == p and cell[q] == cell[p]):
                bad.append((t, int(p), q))
    assert not bad, f"{len(bad)} dangling partner links, e.g. {bad[:3]}"


# ------------------------------------------------------------- 10 (gates)
@pytest.mark.parametrize("gate_motif, species, on_value", [
    ("M3M3M1M2M4M4", None, None),          # photoactive: light
    ("M4M1M1M2M2M3", "A", 1_000_000),      # sense_A: signal A
])
def test_gates_switch_every_effect(tmp_path, device, gate_motif, species, on_value):
    membrane = "M1M1M3M4M2M2"
    stores = []
    for on in (False, True):
        cfg, state, sched, (y, x) = _life_sim(tmp_path / str(on), device,
                                             seq=gate_motif + membrane + PAD, pp=200000)
        spec = state.species.numpy()
        spec[sched.chem.index["L"], y, x] = 5000
        if species and on:
            spec[sched.chem.index[species], y, x] = on_value
        _set(state, "species", spec)
        if species is None and on:
            _set(state, "light_water", np.full(state.shape, 300.0, np.float32))
        for _ in range(20):
            sched.tick_once()
        stores.append(int(state.membrane_store.numpy()[y, x]))
    assert stores[0] == 0, f"gated-off membrane still accreted {stores[0]} L"
    assert stores[1] > 0


# ------------------------------------------------------------- untried: replay --verify
def _cadence_cfg(ticks_per_snapshot):
    cfg = tiny_cfg(n=16, cap=200)
    cfg.run.snapshot_every_sim_days = ticks_per_snapshot * 60 / 86400
    cfg.chemistry.overrides = {m: {"init_wet": 400} for m in ("M1", "M2", "M3", "M4")}
    return cfg


def test_replay_verify_refuses_to_compare_a_snapshot_with_itself(isolated_cwd, device):
    from firmament import cli
    from firmament.io import rundir
    (isolated_cwd / "c.yaml").write_text(yaml.safe_dump(tiny_cfg(n=16, cap=100).model_dump(mode="json")))
    cli.main(["--device", device, "run", "--config", "c.yaml", "--ticks", "20"])
    rid = rundir.list_runs()[0]["run_id"]
    with pytest.raises(SystemExit):                         # no snapshot BEFORE tick 20
        cli.main(["--device", device, "replay", "--run", rid, "--to-tick", "20", "--verify"])


def test_replay_verify_across_a_seed_placement(isolated_cwd, device, capsys):
    from firmament import cli
    from firmament.io import rundir
    cfg = _cadence_cfg(5)
    (isolated_cwd / "c.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    cli.main(["--device", device, "run", "--config", "c.yaml", "--ticks", "10"])
    rid = rundir.list_runs()[0]["run_id"]
    probe, _ = build_test_sim(cfg, "cpu", isolated_cwd / "probe")
    y, x = np.argwhere(probe.water_depth.numpy() > 0.3)[0]
    cli.main(["--device", device, "seed", "--run", rid, "--sequence", SEED_SEQ,
              "--cell", f"{int(x)},{int(y)}"])
    cli.main(["--device", device, "replay", "--run", rid, "--snapshot", "5",
              "--to-tick", "10", "--verify"])
    assert "VERIFY: identical" in capsys.readouterr().out


# ------------------------------------------------------------- untried: lease race
def _grab(run, barrier, q):
    from firmament.io import rundir
    barrier.wait()
    try:
        rundir.acquire_lease(run)
        q.put("won")
        time.sleep(2)
    except rundir.LeaseError:
        q.put("lost")


def test_lease_is_exclusive_when_many_processes_reclaim_a_stale_one(isolated_cwd):
    import multiprocessing as mp

    from firmament.io import rundir
    run = rundir.create(tiny_cfg())
    (run / ".lease").write_text(json.dumps({"pid": 999_999_999}))     # dead holder
    ctx = mp.get_context("fork")
    barrier, q = ctx.Barrier(8), ctx.Queue()
    procs = [ctx.Process(target=_grab, args=(run, barrier, q)) for _ in range(8)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(30)
    results = [q.get(timeout=5) for _ in procs]
    assert results.count("won") == 1, results


# ------------------------------------------------------------- lead: CUDA-graph pointers
def test_events_and_edits_mutate_fluid_arrays_in_place(tmp_path, device):
    """The fluid CUDA graph holds fixed array pointers; rebinding a state array to a new
    wp.array leaves the graph integrating stale memory on GPU."""
    from firmament.operator import console, developer
    cfg = tiny_cfg(n=16, cap=100)
    state, sched = build_test_sim(cfg, device, tmp_path / "w")
    sched.developer_allowed = True
    # exactly the state arrays the fluid substep CUDA graph captures by pointer
    # (temp/species are ping-ponged by their own modules outside the graph by design)
    before = {k: getattr(state, k) for k in ("water_depth", "water_u", "water_v",
                                             "elevation", "sediment")}
    for ev, prm in (("flood", {"rise_m": 0.2}), ("earthquake", {"magnitude_m": 0.3}),
                    ("volcano", {"radius": 2}), ("meteor", {"energy_J": 1e6}),
                    ("rain", {"intensity_mm": 1.0})):
        console.request(sched, ev, prm)
        sched.tick_once()
    developer.set_field(sched, "water_depth", 1, 1, 0.5)
    for k, arr in before.items():
        assert getattr(state, k) is arr, f"{k} was rebound to a new array"


# ------------------------------------------------------------- lead: event RNG collision
def test_event_rng_never_collides_across_ticks_and_sequence_numbers(tmp_path, device):
    from firmament.operator import console
    deltas = []
    for tick, n in ((1, 0), (0, 1000)):
        cfg = tiny_cfg(n=16, cap=10)
        state, sched = build_test_sim(cfg, device, tmp_path / f"{tick}-{n}")
        e0 = state.elevation.numpy().copy()
        console.apply_natural(sched, "earthquake", {"magnitude_m": 1.0}, n, tick)
        deltas.append(state.elevation.numpy() - e0)
    assert not np.array_equal(deltas[0], deltas[1])


# ------------------------------------------------------------- lead: restart counter
def test_restart_counter_resets_once_the_run_makes_progress(tmp_path, device):
    cfg = tiny_cfg(n=16, cap=10)
    state, sched = build_test_sim(cfg, device, tmp_path / "w")
    (tmp_path / "w" / ".restart_count").write_text("3")
    sched.tick_once()
    sched.snapshots.save(state, sched)
    assert not (tmp_path / "w" / ".restart_count").exists()


# ------------------------------------------------------------- lead: crash record
def test_crash_record_points_at_a_complete_snapshot(tmp_path):
    from firmament.cli import _last_good_snapshot
    snaps = tmp_path / "snapshots"
    (snaps / "tick_000000000005.zarr").mkdir(parents=True)
    (snaps / "tick_000000000009.zarr.tmp").mkdir()
    (snaps / "tick_000000000008.zarr.old").mkdir()
    assert _last_good_snapshot(tmp_path).endswith("tick_000000000005.zarr")


# ------------------------------------------------------------- leads: API robustness
def test_inspector_rejects_out_of_range_cells_and_viewers_cannot_crash_the_sim(tmp_path, device):
    from fastapi.testclient import TestClient

    from firmament.instruments.sampler import attach_instruments
    from firmament.server.api import make_app
    cfg = tiny_cfg(n=16, cap=100)
    state, sched = build_test_sim(cfg, device, tmp_path / "w")
    attach_instruments(sched, cfg, tmp_path / "w")
    (tmp_path / "w" / "meta.json").write_text(json.dumps(
        {"run_id": "w", "parent": None, "created": "x", "touched": False}))
    c = TestClient(make_app(sched, tmp_path / "w"))
    for _ in range(101):
        sched.tick_once()
    assert c.get("/api/inspect/cell?x=-1&y=0").status_code == 400
    assert c.get("/api/inspect/cell?x=999&y=0").status_code == 400
    with c.websocket_connect("/ws/state") as ws:
        ws.send_text(json.dumps({"layers": ["motif:abc", "species:NOPE", "bogus", "water"]}))
        time.sleep(0.8)
        for _ in range(3):
            sched.tick_once()                               # must not raise in the sim thread
        time.sleep(0.6)
        sched.tick_once()


# ------------------------------------------------------------- lead: dust law
def test_dust_settling_is_independent_of_dt(tmp_path, device):
    out = []
    for dt in (60.0, 120.0):
        cfg = tiny_cfg(n=8, cap=10, dt=dt)
        state, sched = build_test_sim(cfg, device, tmp_path / f"d{int(dt)}")
        sched.modules = [sched.modules[0]]                  # radiation only
        _set(state, "albedo_dust", np.full(state.shape, 0.5, np.float32))
        for _ in range(int(86400 / dt)):
            sched.tick_once()
        out.append(float(state.albedo_dust.numpy().mean()))
    assert abs(out[0] - out[1]) / out[0] < 1e-3, out


# ------------------------------------------------------------- lead: developer edits vs audit
def test_developer_edit_does_not_crash_the_next_audit(tmp_path, device):
    from firmament.operator import developer
    state, sched = _audited(tmp_path, device)
    for _ in range(6):
        sched.tick_once()
    developer.set_species(sched, "PP", 3, 3, 12345, "edit")
    for _ in range(12):
        sched.tick_once()                                   # AuditError = failure
