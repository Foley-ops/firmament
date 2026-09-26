"""Run directory lifecycle: runs/<run_id>/ with frozen config, logs, snapshots, metrics."""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

RUNS = Path("runs")


def provenance(cfg) -> dict:
    """Everything needed to say exactly which code and rules produced a run."""
    from firmament.config import file_sha256
    from firmament.core.polymers import RULESET

    def git(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True,
                                  cwd=Path(__file__).resolve().parents[2]).stdout.strip()
        except OSError:
            return None
    import numpy
    import warp
    return {
        "ruleset": RULESET,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "rule_files_sha256": {k: file_sha256(p) for k, p in cfg.rule_files().items()},
        "config_hash": cfg.hash(),
        "python": platform.python_version(), "numpy": numpy.__version__,
        "warp": warp.config.version, "platform": platform.platform(),
    }


def new_run_id(cfg) -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{cfg.run.name}-{cfg.hash()[:6]}"


def create(cfg, run_id: str | None = None, parent: str | None = None) -> Path:
    run_id = run_id or new_run_id(cfg)
    # same-second fork of the same config must not collide with its parent
    if run_id and (RUNS / run_id / "meta.json").exists():
        k = 2
        while (RUNS / f"{run_id}.{k}" / "meta.json").exists():
            k += 1
        run_id = f"{run_id}.{k}"
    d = RUNS / run_id
    for sub in ("logs", "snapshots", "metrics", "reports", "rules"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    # freeze the rule files inside the run; the run's config points at these copies so
    # resume/replay never read whatever happens to be in the workspace later
    frozen = cfg.model_copy(deep=True)
    for key, src in cfg.rule_files().items():
        dst = d / "rules" / f"{key}.yaml"
        if Path(src).resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        if key == "chemistry":
            frozen.chemistry.file = str(dst)
        else:
            frozen.polymers.genetic_code = str(dst)
    (d / "config.yaml").write_text(frozen.canonical_yaml())
    (d / "events.jsonl").touch()
    (d / "analysis.jsonl").touch()
    meta = {"run_id": run_id, "parent": parent, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "touched": False, "provenance": provenance(cfg)}
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


class LeaseError(RuntimeError):
    pass


def acquire_lease(run_dir: Path) -> Path:
    """Single-writer lease: one process may write a run directory at a time. A lease
    whose pid is dead is reclaimed (and logged); a lease held by this same pid (e.g.
    an auto-restart re-exec) is re-entered."""
    import atexit
    import os

    from firmament.io.logging import get
    lease = Path(run_dir) / ".lease"
    me = os.getpid()
    while True:
        try:
            fd = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({"pid": me, "since": time.strftime("%Y-%m-%dT%H:%M:%S")}).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                holder = json.loads(lease.read_text())["pid"]
            except (OSError, ValueError, KeyError):
                holder = None
            if holder == me:
                break
            if holder is not None and _pid_alive(holder):
                raise LeaseError(f"{run_dir} is being written by pid {holder}; "
                                 "stop that process first (single-writer rule)")
            get("rundir").warning("reclaiming stale run lease", extra={"stale_pid": holder})
            lease.unlink(missing_ok=True)

    def _release():
        try:
            if json.loads(lease.read_text())["pid"] == me:
                lease.unlink()
        except (OSError, ValueError, KeyError):
            pass
    atexit.register(_release)
    return lease


def release_lease(run_dir: Path) -> None:
    import os
    lease = Path(run_dir) / ".lease"
    try:
        if json.loads(lease.read_text())["pid"] == os.getpid():
            lease.unlink()
    except (OSError, ValueError, KeyError):
        pass


def _pid_alive(pid: int) -> bool:
    import os
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
