"""Ecology instrument: energy accounting, spatial range, interaction signals."""
from __future__ import annotations

import numpy as np


def sample(v, cfg) -> dict:
    alive = np.nonzero(v["p_state"] > 0)[0]
    copies = v["recent_copies"]
    # P~P spent on copying this tick ~ monomers added; decay path is chemistry's
    pp_copy = sum(e[3] for e in copies) * cfg.polymers.copy_energy_per_monomer if copies else 0
    row = {"copies_per_sample": len(copies), "pp_copy_estimate": pp_copy}
    if len(alive) == 0:
        return row | {"range_cells": 0, "bound_pairs": 0}
    cells = np.unique(v["p_cell"][alive])
    partners = v["p_partner"][alive]
    row |= {"range_cells": int(len(cells)),
            "bound_pairs": int((partners >= 0).sum()) // 2}
    return row
