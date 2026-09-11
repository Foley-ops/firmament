"""Phase 5 acceptance: templated copying gated by monomers AND energy; measured
mutation rate ~ epsilon; motif scan matches hand-computed truth; element/energy
audits hold with polymers alive; replay stays bit-identical with polymers."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from tests.conftest import REPO, build_test_sim, tiny_cfg

REPLICASE = "M1M3M1M2M4M2"
PAD = "M1M3M2M4" * 6                      # neutral padding, no catalog motif inside
SEED_SEQ = REPLICASE + PAD                # length 30 monomers


def _life_sim(tmp_path, device, seq=SEED_SEQ, pp=100000, monomers=2000, n=16,
              mutation=0.005, feed=False):
    """A quiet pond: chemistry/fluid off so resources are exactly what we grant.
    Polymers module only -> copy gating is attributable to our inputs alone."""
    cfg = tiny_cfg(n=n, water=0.9, cap=3000, seed=4242)
    cfg.polymers.mutation_rate_per_monomer = mutation
    state, sched = build_test_sim(cfg, device, tmp_path / "run")
    sched.modules = [sched.modules[4]]                    # polymers only
    # flatten the world: uniform tepid water everywhere, no flow
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


def _alive_count(state):
    return int((state.p_state.numpy() > 0).sum())


class TestCopyGating:
    def test_copies_with_resources(self, tmp_path, device):
        cfg, state, sched, _ = _life_sim(tmp_path, device)
        for _ in range(200):                              # ~6 copy generations
            sched.tick_once()
        n = _alive_count(state)
        assert n > 1, "replicase with monomers and energy failed to copy"
        sched.lineage.flush()
        assert sched.lineage.db.execute("SELECT COUNT(*) FROM copies").fetchone()[0] > 0

    def test_no_copy_without_energy(self, tmp_path, device):
        cfg, state, sched, _ = _life_sim(tmp_path, device, pp=0)
        for _ in range(200):
            sched.tick_once()
        # copying may START (binding is free) but no monomer can ever be added
        alive = state.p_state.numpy() > 0
        lens = state.p_len.numpy()[alive]
        assert (lens > 0).sum() == 1, "a child grew without energy"

    def test_no_copy_without_monomers(self, tmp_path, device):
        # the seed is built FROM cell monomers (conservation), so grant exactly
        # enough to construct it, then strip the leftovers before ticking
        cfg, state, sched, (y, x) = _life_sim(tmp_path, device, monomers=10)
        spec = state.species.numpy()
        for m in ("M1", "M2", "M3", "M4"):
            spec[sched.chem.index[m], y, x] = 0
        state.species = wp.array(spec, dtype=wp.int32, device=state.device)
        for _ in range(200):
            sched.tick_once()
        alive = state.p_state.numpy() > 0
        lens = state.p_len.numpy()[alive]
        assert (lens > 0).sum() == 1, "a child grew without monomers"

    def test_copy_stalls_and_resumes(self, tmp_path, device):
        """Starve mid-copy, then refeed: the copy must finish (stall, not abort)."""
        cfg, state, sched, (y, x) = _life_sim(tmp_path, device, pp=10)  # 10 monomers' worth
        for _ in range(60):
            sched.tick_once()
        stalled = _alive_count(state)
        idx = sched.chem.index
        spec = state.species.numpy()
        assert spec[idx["PP"], y, x] == 0, "energy not exhausted as expected"
        spec[idx["PP"], y, x] = 100000                    # refeed
        state.species = wp.array(spec, dtype=wp.int32, device=state.device)
        for _ in range(120):
            sched.tick_once()
        assert _alive_count(state) > stalled, "stalled copy did not resume after refeed"


def test_motif_scan_matches_hand_computed(tmp_path, device):
    """Kernel-side motif mask of a completed child vs. a hand-built truth table.
    The child of a palindromic-motif seed carries the same motifs (complement!)."""
    cfg, state, sched, _ = _life_sim(tmp_path, device, mutation=0.0)
    poly = sched.poly
    # hand-computed: which catalog motifs sit in SEED_SEQ?
    sym = {"M1": 1, "M2": 2, "M3": 3, "M4": 4}
    seq = np.array([sym[SEED_SEQ[i:i + 2]] for i in range(0, len(SEED_SEQ), 2)], np.uint8)
    truth = 0
    for mi, row in enumerate(poly.motif_np):
        k = len(row)
        if any(np.array_equal(seq[p:p + k], row) for p in range(len(seq) - k + 1)):
            truth |= 1 << mi
    assert truth == poly.repl_bit, "seed should carry exactly the replicase motif"
    for _ in range(120):
        sched.tick_once()
    st = state.p_state.numpy()
    born = np.nonzero((st > 0) & (state.p_id.numpy() > 1))[0]
    assert len(born) > 0, "no child completed"
    for b in born:
        if st[b] in (1, 2, 3):                            # completed children only
            # complement of the template: motifs preserved because palindromic
            assert int(state.p_motifs.numpy()[b]) == truth, "kernel scan != hand-computed"
    # and the child really is the REVERSE complement (antiparallel synthesis),
    # not a byte copy and not the parallel complement
    child = born[0]
    ln = int(state.p_len.numpy()[child])
    if ln == len(seq):
        rc = np.array([{1: 2, 2: 1, 3: 4, 4: 3}[int(v)] for v in seq], np.uint8)[::-1]
        got = state.p_seq.numpy()[child, :ln]
        assert np.array_equal(got, rc), "child is not the template's reverse complement"


@pytest.mark.slow
def test_measured_mutation_rate_matches_epsilon(tmp_path, device):
    """Over thousands of copied monomers the observed mutation count must sit near
    epsilon * monomers (binomial), within 4 sigma."""
    eps = 0.02                                            # higher rate -> tighter stats fast
    cfg, state, sched, (y, x) = _life_sim(tmp_path, device, pp=500000, monomers=20000,
                                          mutation=eps)
    total_monomers = 0
    total_muts = 0
    for _ in range(1500):
        sched.tick_once()
        for (cid, pid, cell, ln, mu) in sched.poly.recent_events:
            total_monomers += ln
            total_muts += mu
        if total_monomers > 60000:
            break
    assert total_monomers > 20000, f"too few copies to measure ({total_monomers} monomers)"
    expected = eps * total_monomers
    sigma = np.sqrt(total_monomers * eps * (1 - eps))
    assert abs(total_muts - expected) < 4 * sigma, (
        f"mutation rate off: {total_muts} vs expected {expected:.0f} ± {sigma:.0f}")


def test_audits_hold_with_life(tmp_path, device):
    """Element totals exact and energy books closed while polymers copy and die.
    Monomers inside polymers count (chain unit = M − H2O)."""
    from firmament.core.audit import Audit
    from firmament.core.chemistry import Chemistry

    cfg, state, sched, _ = _life_sim(tmp_path, device, pp=50000)
    chem = Chemistry.load(str(REPO / "configs/chemistry_v0.yaml"))
    aud = Audit(cfg, chem, tmp_path)
    el0 = aud.element_totals(state)
    e0 = aud.energy_stored(state)
    for _ in range(400):
        sched.tick_once()
    el1 = aud.element_totals(state)
    assert np.array_equal(el0, el1), f"element drift with life: {(el1 - el0).tolist()}"
    # polymer effects/copying release heat tracked in e_chem — books must close
    e_chem = float(state.e_chem.numpy().sum())
    e1 = aud.energy_stored(state)
    assert abs((e1 - e0) - e_chem) / max(abs(e_chem), 1.0) < 1e-6, (
        f"energy books off with life: dStored={e1 - e0:.1f} vs e_chem={e_chem:.1f}")
    assert _alive_count(state) > 1


@pytest.mark.slow
def test_replay_bit_identical_with_polymers(tmp_path, device):
    from tests.test_phase4_replay import run_replay_protocol
    run_replay_protocol(tmp_path, device, seeded=True)
