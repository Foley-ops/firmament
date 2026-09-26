"""Shared helpers for the history-area repros. CPU only; every run lives in a fresh
temp dir under the scratch area (the CLI writes ./runs relative to cwd)."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = "/home/nick/Life/firmament"
sys.path.insert(0, REPO)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
SCRATCH = Path("/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/"
               "scratchpad/court/history")
PY = f"{REPO}/.venv/bin/python"


def workdir(tag: str) -> Path:
    d = Path(tempfile.mkdtemp(prefix=f"{tag}_", dir=SCRATCH))
    return d


def write_cfg(wd: Path, n=16, cap=500, snap_ticks=100000, metrics_every=100, overrides=None,
              seed=20260910) -> Path:
    import yaml
    from tests.conftest import tiny_cfg
    cfg = tiny_cfg(n=n, cap=cap, seed=seed)
    # snapshot cadence in ticks -> sim days (dt = 60 s)
    cfg.run.snapshot_every_sim_days = snap_ticks * 60.0 / 86400.0
    cfg.run.metrics_every_ticks = metrics_every
    if overrides:
        cfg.chemistry.overrides = overrides
    p = wd / "c.yaml"
    p.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    return p


def cli(wd: Path, *args, check=False, timeout=600):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=REPO)
    r = subprocess.run([PY, "-m", "firmament.cli", "--device", "cpu", *args], cwd=wd,
                       env=env, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"CLI failed: {args}\n{r.stdout}\n{r.stderr}")
    return r


def only_run(wd: Path) -> str:
    runs = sorted(p.name for p in (wd / "runs").iterdir() if (p / "meta.json").exists())
    return runs[-1]


def events(wd: Path, rid: str, name="events.jsonl"):
    p = wd / "runs" / rid / name
    return [json.loads(x) for x in open(p) if x.strip()]


def last_line(s: str) -> str:
    lines = [x for x in s.strip().splitlines() if x.strip()]
    return lines[-1] if lines else ""
