"""Diversity instrument: distinct sequences, Shannon entropy, motif census,
compressed size of the living sequence set (structure proxy)."""
from __future__ import annotations

import zlib

import numpy as np


def seq_hashes(v) -> np.ndarray:
    alive = np.nonzero(v["p_state"] > 0)[0]
    if len(alive) == 0:
        return np.array([], dtype=np.uint64)
    seqs = v["p_seq"][alive]
    lens = v["p_len"][alive]
    # cheap rolling hash over the used prefix, vectorized per unique length
    out = np.zeros(len(alive), dtype=np.uint64)
    for ln in np.unique(lens):
        m = lens == ln
        if ln == 0:
            continue
        block = seqs[m, :ln].astype(np.uint64)
        h = np.zeros(block.shape[0], dtype=np.uint64)
        for x in range(int(ln)):
            h = h * np.uint64(1099511628211) + block[:, x] + np.uint64(1)
        out[m] = h
    return out


def sample(v) -> dict:
    hashes = seq_hashes(v)
    n = len(hashes)
    if n == 0:
        return {"distinct_seqs": 0, "seq_entropy_bits": 0.0, "compressed_bytes": 0,
                "motif_census": "{}", "mutants": 0}
    uniq, counts = np.unique(hashes, return_counts=True)
    p = counts / n
    entropy = float(-(p * np.log2(p)).sum())
    alive = np.nonzero(v["p_state"] > 0)[0]
    blob = v["p_seq"][alive].tobytes()
    comp = len(zlib.compress(blob, 6))
    masks = v["p_motifs"][alive]
    census = {}
    for b in range(16):
        c = int(((masks >> b) & 1).sum())
        if c:
            census[str(b)] = c
    return {"distinct_seqs": int(len(uniq)), "seq_entropy_bits": entropy,
            "compressed_bytes": comp, "motif_census": str(census),
            "mutants": int((v["p_mut"][alive] > 0).sum()) if "p_mut" in v else 0}
