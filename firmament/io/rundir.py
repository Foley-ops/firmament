"""Run directory lifecycle: runs/<run_id>/ with frozen config, logs, snapshots, metrics."""
from __future__ import annotations

import json
import time
from pathlib import Path

RUNS = Path("runs")


def new_run_id(cfg) -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{cfg.run.name}-{cfg.hash()[:6]}"


def create(cfg, run_id: str | None = None, parent: str | None = None) -> Path:
    run_id = run_id or new_run_id(cfg)
    d = RUNS / run_id
    for sub in ("logs", "snapshots", "metrics", "reports"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(cfg.canonical_yaml())      # frozen copy
    (d / "events.jsonl").touch()
    meta = {"run_id": run_id, "parent": parent, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "touched": False}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return d


def read_meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "meta.json").read_text())


def write_meta(run_dir: Path, meta: dict) -> None:
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))


def mark_touched(run_dir: Path) -> None:
    m = read_meta(run_dir)
    m["touched"] = True
    write_meta(run_dir, m)


def list_runs() -> list[dict]:
    out = []
    if RUNS.exists():
        for d in sorted(RUNS.iterdir()):
            if (d / "meta.json").exists():
                out.append(read_meta(d) | {"dir": str(d)})
    return out
