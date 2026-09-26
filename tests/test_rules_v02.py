"""Genesis v0.2 rules: thermodynamic energy economy, dt as a numerical knob (not a law),
strict configuration, and the single-writer run lease."""
from __future__ import annotations

import copy
import json
import subprocess

import numpy as np
import pytest
import warp as wp
import yaml
from pydantic import ValidationError

from tests.conftest import REPO, build_test_sim, tiny_cfg
from tests.test_phase5_polymers import SEED_SEQ, _completed, _copies, _life_sim

CHEM = REPO / "configs/chemistry_v0_2.yaml"


# ---------------------------------------------------------------- thermodynamics
def _dark_box(tmp_path, device, pp=0, pi=None, ticks=600):
    cfg = tiny_cfg(n=16, cap=100, solar=0.0, geo=0.0, vents=[])
    state, sched = build_test_sim(cfg, device, tmp_path / "box")
    sched.modules = [sched.modules[3]]                    # chemistry only
    spec = state.species.numpy()
    spec[sched.chem.index["PP"]] = pp
    if pi is not None:
        spec[sched.chem.index["Pi"]] = pi
    state.species = wp.array(spec, dtype=wp.int32, device=device)
    for _ in range(ticks):
        sched.tick_once()
    return sched, state


def test_no_energy_carrier_made_in_the_dark(tmp_path, device):
    """v0.1 free lunch: thermal condensation at ambient temperature filled every dark
    pool with P~P. Thermodynamic gating: none may form without light."""
    sched, state = _dark_box(tmp_path, device)
    assert int(state.species.numpy()[sched.chem.index["PP"]].sum()) == 0


def test_energy_carrier_still_decays_in_the_dark(tmp_path, device):
    sched, state = _dark_box(tmp_path, device, pp=5000)
    assert int(state.species.numpy()[sched.chem.index["PP"]].sum()) < 5000 * 16 * 16


@pytest.mark.parametrize("pi, should_copy", [(0, True), (2_000_000_000, False)])
def test_copying_needs_real_free_energy(tmp_path, device, pi, should_copy):
    """Same P~P count, different Pi: copying proceeds only while hydrolysing P~P at the
    cell's actual concentrations pays for the condensation. Swamped in Pi (P~P near
    equilibrium) a replicase must stall even though P~P >= the per-step cost."""
    cfg, state, sched, (y, x) = _life_sim(tmp_path, device, pp=2000)
    spec = state.species.numpy()
    spec[sched.chem.index["Pi"], y, x] = pi
    state.species = wp.array(spec, dtype=wp.int32, device=device)
    for _ in range(300):
        sched.tick_once()
    child = int(state.p_child.numpy()[0])
    grew = (child >= 0 and int(state.p_len.numpy()[child]) > 0) or _copies(sched) > 0
    assert grew is should_copy


def test_loader_rejects_an_ungated_route_to_the_energy_carrier():
    from firmament.core.chemistry import Chemistry
    doc = yaml.safe_load(open(CHEM))
    doc = copy.deepcopy(doc)
    for r in doc["reactions"]:
        if r["name"] == "vent_phosphorylation":
            del r["dg0"]
    with pytest.raises(ValueError, match="free lunch"):
        Chemistry(doc)


def test_copy_step_stoichiometry_is_exact_with_two_carriers(tmp_path, device):
    """M + 2 P~P + H2O -> unit + 4 Pi: element totals exact to the atom."""
    from firmament.core.audit import Audit
    cfg, state, sched, _ = _life_sim(tmp_path, device)
    assert cfg.polymers.copy_energy_per_monomer == 2
    aud = Audit(cfg, sched.chem, tmp_path)
    e0 = aud.element_totals(state)
    for _ in range(300):
        sched.tick_once()
    assert len(_completed(state)) > 5
    assert np.array_equal(aud.element_totals(state), e0)


# ---------------------------------------------------------------- dt is a knob
def _chem_change(tmp_path, device, dt, sim_seconds, label):
    cfg = tiny_cfg(n=16, cap=100, solar=0.0, geo=0.0, vents=[], dt=dt)
    state, sched = build_test_sim(cfg, device, tmp_path / label)
    sched.modules = [sched.modules[3]]
    s0 = state.species.numpy().astype(np.int64).sum(axis=(1, 2))
    for _ in range(int(sim_seconds / dt)):
        sched.tick_once()
    return state.species.numpy().astype(np.int64).sum(axis=(1, 2)) - s0


def test_chemistry_extent_is_independent_of_dt(tmp_path, device):
    """v0.1 rates were per tick, so doubling dt halved the physics per second."""
    a = _chem_change(tmp_path, device, 60.0, 4 * 3600, "dt60")
    b = _chem_change(tmp_path, device, 120.0, 4 * 3600, "dt120")
    big = np.abs(a) > 2000
    assert big.sum() >= 3, "too little chemistry to compare"
    rel = np.abs(a[big] - b[big]) / np.abs(a[big])
    assert rel.max() < 0.05, f"extent differs by {rel.max():.1%} between dt=60 and dt=120"


def test_copy_speed_is_independent_of_dt(tmp_path, device):
    """The first copy of a 30-mer takes the same SIMULATED time at dt=60 and dt=120."""
    times = []
    for dt in (60.0, 120.0):
        cfg = tiny_cfg(n=16, water=0.9, cap=3000, seed=4242, dt=dt)
        cfg.polymers.mutation_rate_per_monomer = 0.0
        state, sched = build_test_sim(cfg, device, tmp_path / f"dt{int(dt)}")
        sched.modules = [sched.modules[4]]
        n = 16
        state.water_depth = wp.array(np.full((n, n), 1.0), dtype=wp.float64, device=device)
        state.temp = wp.array(np.full((3, n, n), 288.0), dtype=wp.float64, device=device)
        spec = np.zeros_like(state.species.numpy())
        idx = sched.chem.index
        spec[idx["H2O"]] = 10_000_000
        for m in ("M1", "M2", "M3", "M4"):
            spec[idx[m], 8, 8] = 2000
        spec[idx["PP"], 8, 8] = 100000
        state.species = wp.array(spec, dtype=wp.int32, device=device)
        from firmament.operator.console import place_seed
        place_seed(sched, state, SEED_SEQ, 8, 8)
        while len(_completed(state)) < 2:
            sched.tick_once()
            assert state.tick * dt < 6 * 3600, f"no copy within 6 sim-hours at dt={dt}"
        times.append(state.tick * dt)
    assert abs(times[0] - times[1]) <= 120.0, times


# ---------------------------------------------------------------- strict config
def _cfg_dict():
    return tiny_cfg().model_dump(mode="json")


@pytest.mark.parametrize("mutate, match", [
    (lambda d: d["run"].update(typo_field=1), "extra"),
    (lambda d: d["run"].update(dt_seconds=90), "multiple of 60"),
    (lambda d: d["world"].update(vents=[[999, 5]]), "outside"),
    (lambda d: d["world"].update(cell_meters=2.0), "cell_meters"),
    (lambda d: d["world"].update(initial_water_fraction=1.5), "less than or equal"),
    (lambda d: d["polymers"].update(mutation_rate_per_monomer=1.2), "less than"),
    (lambda d: d["chemistry"].update(overrides={"M1": {"init_wtt": 5}}), "unknown keys"),
])
def test_config_rejects_bad_values(mutate, match):
    from firmament.config import Config
    d = _cfg_dict()
    mutate(d)
    with pytest.raises((ValidationError, ValueError), match=match):
        Config.model_validate(d)


def test_unknown_override_species_is_rejected(tmp_path, device):
    cfg = tiny_cfg()
    cfg.chemistry.overrides = {"UNOBTAINIUM": {"init_wet": 5}}
    with pytest.raises(ValueError, match="unknown species"):
        build_test_sim(cfg, device, tmp_path / "r")


def test_chemistry_file_rejects_unknown_reaction_keys():
    from firmament.core.chemistry import Chemistry
    doc = copy.deepcopy(yaml.safe_load(open(CHEM)))
    doc["reactions"][0]["k30O"] = 1.0                       # typo'd key
    with pytest.raises(ValueError, match="unknown keys"):
        Chemistry(doc)


def test_fluid_rejects_a_dt_that_breaks_stability():
    from firmament.core.fluid import Fluid
    with pytest.raises(ValueError, match="stability"):
        Fluid(tiny_cfg(dt=3000.0)).check_stability()


def test_shipped_world_configs_validate():
    from firmament.config import Config
    for f in sorted((REPO / "configs").glob("world_*.yaml")):
        Config.load(f)


# ---------------------------------------------------------------- run lease
def test_single_writer_lease(isolated_cwd):
    from firmament.io import rundir
    run = rundir.create(tiny_cfg())
    rundir.acquire_lease(run)
    rundir.acquire_lease(run)                               # same pid: re-entrant
    rundir.release_lease(run)
    other = _hold_lease_in_subprocess(run)
    try:
        with pytest.raises(rundir.LeaseError, match=str(other.pid)):
            rundir.acquire_lease(run)
    finally:
        other.kill()
        other.wait()
    rundir.acquire_lease(run)                               # holder died: lock released
    assert json.loads((run / ".lease").read_text())["pid"] == __import__("os").getpid()


def _hold_lease_in_subprocess(run):
    import sys
    import time as _t
    code = ("import sys, time; sys.path.insert(0, %r); from firmament.io import rundir; "
            "rundir.acquire_lease(%r); print('held', flush=True); time.sleep(60)" % (str(REPO), str(run)))
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
                         env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""})
    assert p.stdout.readline().strip() == "held"
    _t.sleep(0.1)
    return p


def test_cli_refuses_to_write_a_run_another_process_holds(isolated_cwd, device):
    from firmament import cli
    from firmament.io import rundir
    p = isolated_cwd / "c.yaml"
    p.write_text(yaml.safe_dump(tiny_cfg(n=16, cap=200).model_dump(mode="json")))
    cli.main(["--device", device, "run", "--config", str(p), "--ticks", "2"])
    rid = rundir.list_runs()[0]["run_id"]
    rundir.release_lease(rundir.RUNS / rid)
    other = _hold_lease_in_subprocess(rundir.RUNS / rid)
    try:
        with pytest.raises(rundir.LeaseError):
            cli.main(["--device", device, "event", "--run", rid, "--type", "rain"])
    finally:
        other.kill()
        other.wait()
