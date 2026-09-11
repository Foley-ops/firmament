"""Nightly summary: a readable markdown report per run (population, diversity,
milestones, novelty flags, disk usage). Reads only files — safe on live runs."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from firmament.io.metrics import MetricsWriter


def _disk_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20


def write_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    t = MetricsWriter(run_dir / "metrics").read_all()
    events = [json.loads(x) for x in open(run_dir / "events.jsonl") if x.strip()] \
        if (run_dir / "events.jsonl").exists() else []
    lines = [f"# {meta['run_id']} — daily report ({time.strftime('%Y-%m-%d %H:%M')})", ""]
    if meta.get("touched"):
        lines.append("**⚠ TOUCHED — this run has developer edits.**\n")
    if t is not None and t.num_rows:
        d = t.to_pydict()
        last = {k: v[-1] for k, v in d.items()}
        lines += [
            "## Population",
            f"- tick {last.get('tick')}, polymers **{last.get('n_polymers', 0)}**, "
            f"compartments {last.get('n_compartments', 0)}, "
            f"in-compartment share {last.get('poly_in_compartments', 0)}/{max(last.get('n_polymers', 0), 1)}",
            f"- length mean {last.get('len_mean', 0):.1f} (max {last.get('len_max', 0)})",
            "",
            "## Diversity",
            f"- distinct sequences {last.get('distinct_seqs', 0)}, "
            f"entropy {last.get('seq_entropy_bits', 0):.2f} bits, "
            f"compressed size {last.get('compressed_bytes', 0)} B, "
            f"mutant carriers {last.get('mutants', 0)}",
            f"- motif census {last.get('motif_census', '{}')}",
            "",
            "## Environment",
            f"- surface T mean {last.get('temp_surf_mean', 0):.1f} K "
            f"({last.get('temp_surf_min', 0):.1f}–{last.get('temp_surf_max', 0):.1f}); "
            f"wet fraction {last.get('wet_fraction', 0):.2f}; "
            f"water {last.get('water_total_m3', 0):.0f} m³ + vapor {last.get('vapor_total_kg', 0):.0f} kg",
            "",
        ]
    else:
        lines.append("_no metrics yet_\n")
    miles = [e for e in events if e.get("component") == "milestones"]
    nov = [e for e in events if e.get("component") == "novelty"]
    ops = [e for e in events if e.get("component") == "operator"]
    mlines = [f"- t{e['tick']}: **{e['event']}**" for e in miles] or ["- none yet"]
    lines += ["## Milestones"] + mlines
    nlines = [f"- t{e['tick']}: {e['event']}" for e in nov[-10:]] or ["- none"]
    lines += ["", "## Novelty flags"] + nlines
    olines = [f"- t{e['tick']}: {e['event']}" for e in ops[-10:]] or ["- none"]
    lines += ["", "## Operator actions"] + olines
    if (run_dir / "lineage.sqlite").exists():
        db = sqlite3.connect(run_dir / "lineage.sqlite")
        n = db.execute("SELECT COUNT(*) FROM copies").fetchone()[0]
        lines += ["", f"## Lineage\n- {n} copy events on record"]
        db.close()
    lines += ["", f"## Disk\n- {_disk_mb(run_dir):.1f} MB total "
              f"({_disk_mb(run_dir / 'snapshots'):.1f} MB snapshots)"]
    out = run_dir / "reports" / f"{time.strftime('%Y-%m-%d')}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out
