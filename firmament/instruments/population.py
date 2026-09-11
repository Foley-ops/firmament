"""Population instrument: counts, length distribution, compartments."""
from __future__ import annotations

import numpy as np


def sample(v) -> dict:
    alive = v["p_state"] > 0
    lens = v["p_len"][alive]
    n = int(alive.sum())
    comp = v.get("compartment_id")
    n_comp = int((comp > 0).sum()) if comp is not None else 0
    row = {"n_polymers": n, "n_compartments": n_comp,
           "membrane_L_total": int(v["membrane_store"].sum()) if "membrane_store" in v else 0}
    if n:
        row |= {"len_mean": float(lens.mean()), "len_max": int(lens.max()),
                "len_min": int(lens.min()), "len_p50": float(np.percentile(lens, 50))}
        # polymers resident in compartment cells vs free water
        if comp is not None:
            cells = v["p_cell"][alive]
            flat = comp.ravel()
            in_comp = int((flat[cells] > 0).sum())
            row["poly_in_compartments"] = in_comp
            row["poly_free"] = n - in_comp
    else:
        row |= {"len_mean": 0.0, "len_max": 0, "len_min": 0, "len_p50": 0.0,
                "poly_in_compartments": 0, "poly_free": 0}
    return row
