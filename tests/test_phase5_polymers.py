"""Phase 5 acceptance (Genesis v0.2 rules): templated copying gated by monomers AND
energy; errors happen only during real incorporation (a starved copy makes no progress
and no mistakes); nothing shorter than MIN_LEN units can exist; mutation rate measured
from SEQUENCES, not the kernel's own counter; motif scan vs an independent truth table;
element/energy books with polymers alive; replay bit-identical with polymers."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from tests.conftest import REPO, build_test_sim, tiny_cfg

REPLICASE = "M1M3M1M2M4M2"
PAD = "M1M3M2M4" * 6                      # neutral padding, no catalog motif inside
SEED_SEQ = REPLICASE + PAD                # length 30 monomers
COMP = {1: 2, 2: 1, 3: 4, 4: 3}


def _life_sim(tmp_path, device, seq=SEED_SEQ, pp=100000, monomers=2000, n=16,
              mutation=0.005):
    """A quiet pond: chemistry/fluid off so resources are exactly what we grant.
    Polymers module only -> copy gating is attributable to our inputs alone."""
    cfg = tiny_cfg(n=n, water=0.9, cap=3000, seed=4242)
    cfg.polymers.mutation_rate_per_monomer = mutation
    state, sched = build_test_sim(cfg, device, tmp_path / "run")
    sched.modules = [sched.modules[4]]                    # polymers only
    h = n
    state.water_depth = wp.array(np.full((h, h), 1.0), dtype=wp.float64, device=device)
    state.temp = wp.array(np.full((3, h, h), 288.0), dtype=wp.float64, device=device)
    spec = np.zeros_like(state.species.numpy())
    idx = sched.chem.index
    y = x = n // 2
    spec[idx["H2O"], :, :] = 10_000_000
    for m in ("M1", "M2", "M3", "M4"):
        spec[idx[m], y, x] = monomers
    spec[idx["PP"], y, x] = pp
    state.species = wp.array(spec, dtype=wp.int32, device=device)
    from firmament.operator.console import place_seed
    place_seed(sched, state, seq, x, y)
    return cfg, state, sched, (y, x)


def _completed(state):
    st = state.p_state.numpy()
    return np.nonzero((st >= 1) & (st <= 3))[0]


def _copies(sched):
    sched.lineage.flush()
    return sched.lineage.db.execute("SELECT COUNT(*) FROM copies").fetchone()[0]


class TestCopyGating:
    def test_copies_with_resources(self, tmp_path, device):
        cfg, state, sched, _ = _life_sim(tmp_path, device)
        for _ in range(200):
            sched.tick_once()
        assert len(_completed(state)) > 1
        assert _copies(sched) > 0

    def test_no_copy_without_energy(self, tmp_path, device):
        cfg, state, sched, _ = _life_sim(tmp_path, device, pp=0)
        for _ in range(500):
            sched.tick_once()
        assert len(_completed(state)) == 1, "a polymer was completed without energy"
        assert _copies(sched) == 0

    def test_no_copy_without_monomers(self, tmp_path, device):
        # the seed is built FROM cell monomers (conservation): grant exactly enough to
        # construct it, then strip the leftovers before ticking
        cfg, state, sched, (y, x) = _life_sim(tmp_path, device, monomers=10)
        spec = state.species.numpy()
        for m in ("M1", "M2", "M3", "M4"):
            spec[sched.chem.index[m], y, x] = 0
        state.species = wp.array(spec, dtype=wp.int32, device=state.device)
        for _ in range(500):
            sched.tick_once()
        assert len(_completed(state)) == 1, "a polymer was completed without monomers"
        assert _copies(sched) == 0

    def test_copy_stalls_and_resumes(self, tmp_path, device):
        """Starve mid-copy, then refeed: the SAME copy finishes (stall, not abort)."""
        cfg, state, sched, (y, x) = _life_sim(tmp_path, device, pp=10)
        for _ in range(60):
            sched.tick_once()
        idx = sched.chem.index
        spec = state.species.numpy()
        assert spec[idx["PP"], y, x] == 0, "energy not exhausted as expected"
        assert len(_completed(state)) == 1 and _copies(sched) == 0
        child = int(state.p_child.numpy()[0])
        assert child >= 0 and 0 < int(state.p_len.numpy()[child]) < 30
        spec[idx["PP"], y, x] = 100000
        state.species = wp.array(spec, dtype=wp.int32, device=state.device)
        for _ in range(120):
            sched.tick_once()
        assert int(state.p_state.numpy()[child]) in (1, 2, 3), "stalled copy did not finish"


def test_starved_copy_makes_no_progress_and_no_errors(tmp_path, device):
    """v0.1 bug regression: a copy starved of energy kept rolling free deletions every
    tick, advanced to the end, and produced short/empty children. With a huge error
    rate and energy for only 5 incorporations, v0.2 must never complete a child."""
    cfg, state, sched, _ = _life_sim(tmp_path, device, pp=5, mutation=0.5)
    for _ in range(3000):
        sched.tick_once()
    assert len(_completed(state)) == 1
    assert _copies(sched) == 0
    assert int(state.p_copy_pos.numpy()[0]) < 30


@pytest.mark.parametrize("short_len", [0, 1])
def test_sub_min_length_templates_never_yield_polymers(tmp_path, device, short_len):
    """v0.1 bug regression: copying an empty template completed instantly, for free, and
    produced an immortal zero-atom 'polymer'. Inject a length-0/1 template next to the
    seed: it gets copied, but the fragment must dissolve — no completed polymer below
    MIN_LEN, no copy event, element totals exact."""
    from firmament.core.audit import Audit
    MIN_LEN = 2          # the rule itself, NOT imported from the code under test

    cfg, state, sched, (y, x) = _life_sim(tmp_path, device)
    arr = {k: getattr(state, k).numpy().copy() for k in
           ("p_state", "p_len", "p_cell", "p_id", "p_partner", "p_child", "p_seq", "p_motifs")}
    arr["p_state"][1], arr["p_len"][1] = 1, short_len
    arr["p_cell"][1], arr["p_id"][1] = int(arr["p_cell"][0]), 999
    arr["p_partner"][1], arr["p_child"][1], arr["p_motifs"][1] = -1, -1, 0
    arr["p_seq"][1, :] = 0
    arr["p_seq"][1, :short_len] = 1
    spec = state.species.numpy()
    if short_len:
        spec[sched.chem.index["M1"], y, x] -= short_len   # its atoms come from the pool
        spec[sched.chem.index["H2O"], y, x] += short_len
    state.species = wp.array(spec, dtype=wp.int32, device=device)
    for k, a in arr.items():
        setattr(state, k, wp.array(a, dtype=getattr(state, k).dtype, device=device))
    aud = Audit(cfg, sched.chem, tmp_path)
    el0 = aud.element_totals(state)
    was_template = False
    for _ in range(600):
        sched.tick_once()
        was_template |= int(state.p_state.numpy()[1]) == 2
    assert was_template, "the short template was never copied — test did not exercise the rule"
    done = _completed(state)
    lens = state.p_len.numpy()[done]
    others = done[done != 1]
    assert state.p_len.numpy()[others].min() >= MIN_LEN
    sched.lineage.flush()
    assert sched.lineage.db.execute(
        "SELECT COUNT(*) FROM copies WHERE length < ?", (MIN_LEN,)).fetchone()[0] == 0
    assert np.array_equal(aud.element_totals(state), el0)
    assert lens.size > 1


def test_motif_scan_matches_independent_truth(tmp_path, device):
    """Truth comes from the motif strings in the YAML, not from the code's parsed table,
    and covers every catalog motif placed in a probe sequence."""
    import yaml
    code = yaml.safe_load(open(REPO / "configs/genetic_code_v0.yaml"))
    names = list(code["motifs"])
    cfg, state, sched, _ = _life_sim(tmp_path, device, mutation=0.0)
    for bit, name in enumerate(names):
        probe = "".join(code["motifs"][name]["seq"]) + "M1M3M2M4"
        sym = {"M1": 1, "M2": 2, "M3": 3, "M4": 4}
        seq = np.array([sym[probe[i:i + 2]] for i in range(0, len(probe), 2)], np.uint8)
        assert sched.poly.motif_mask_host(seq) & (1 << bit), f"{name} not detected"
    for _ in range(150):
        sched.tick_once()
    child = [c for c in _completed(state) if c != 0][0]
    ln = int(state.p_len.numpy()[child])
    seed = np.array([{"M1": 1, "M2": 2, "M3": 3, "M4": 4}[SEED_SEQ[i:i + 2]]
                     for i in range(0, len(SEED_SEQ), 2)], np.uint8)
    rc = np.array([COMP[int(v)] for v in seed], np.uint8)[::-1]
    assert np.array_equal(state.p_seq.numpy()[child, :ln], rc), "child is not the reverse complement"
    assert int(state.p_motifs.numpy()[child]) == sched.poly.repl_bit


@pytest.mark.slow
def test_mutation_rate_measured_from_sequences(tmp_path, device):
    """Compare each finished child to the reverse complement of its (still living)
    template. Among equal-length pairs (no net indel), the per-base substitution rate
    must be 0.8*eps (substitutions are 80% of errors and always change the base)."""
    eps = 0.03
    cfg, state, sched, _ = _life_sim(tmp_path, device, pp=500000, monomers=20000, mutation=eps)
    for _ in range(900):
        sched.tick_once()
    done = _completed(state)
    ids = state.p_id.numpy()
    by_id = {int(ids[s]): s for s in done}
    seqs, lens, parents = state.p_seq.numpy(), state.p_len.numpy(), state.p_parent.numpy()
    positions = subs = 0
    for c in done:
        t = by_id.get(int(parents[c]))
        if t is None or t == c or lens[c] != lens[t]:
            continue
        ln = int(lens[c])
        rc = np.array([COMP[int(v)] for v in seqs[t, :ln]], np.uint8)[::-1]
        d = int((seqs[c, :ln] != rc).sum())
        if d > 3:            # misaligned by an insertion+deletion pair: skip
            continue
        positions += ln
        subs += d
    assert positions > 8000, f"too few comparable positions ({positions})"
    p = 0.8 * eps
    expected, sigma = p * positions, np.sqrt(positions * p * (1 - p))
    assert abs(subs - expected) < 4 * sigma, f"{subs} substitutions vs {expected:.0f} ± {sigma:.0f}"


def test_audits_hold_with_life(tmp_path, device):
    """Element totals exact while polymers copy and die (chain unit = M − H2O); the
    thermal energy change equals the reaction heat released (ledger closure)."""
    from firmament.core.audit import Audit
    from firmament.core.chemistry import Chemistry

    cfg, state, sched, _ = _life_sim(tmp_path, device, pp=50000)
    chem = Chemistry.load(str(REPO / "configs/chemistry_v0.yaml"))
    aud = Audit(cfg, chem, tmp_path)
    el0 = aud.element_totals(state)
    e0 = aud.energy_stored(state)
    for _ in range(400):
        sched.tick_once()
    assert np.array_equal(el0, aud.element_totals(state))
    e_chem = float(state.e_chem.numpy().sum())
    assert e_chem > 0, "no reaction heat — nothing happened"
    assert abs((aud.energy_stored(state) - e0) - e_chem) / e_chem < 1e-6
    assert len(_completed(state)) > 1


@pytest.mark.slow
def test_replay_bit_identical_with_polymers(tmp_path, device):
    from tests.test_phase4_replay import run_replay_protocol
    run_replay_protocol(tmp_path, device, seeded=True)
