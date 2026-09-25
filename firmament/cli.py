"""firmament run|resume|fork|replay|inspect|seed — headless CLI entry point."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from firmament.config import Config
from firmament.io import rundir
from firmament.io.logging import CLOCK, get, setup


def build_sim(cfg: Config, run_dir: Path, device: str):
    """Assemble state + scheduler with all modules registered (grows per phase)."""
    import warp as wp

    wp.init()
    from firmament.core.chemistry import Chemistry
    from firmament.core.scheduler import Scheduler
    from firmament.core.state import State

    chem = Chemistry.load(cfg.chemistry.file)
    state = State(cfg, n_species=chem.n_species, device=device)
    sched = Scheduler(state, cfg, run_dir)
    from firmament.core import audit, fluid, polymers, radiation, terrain, thermal

    terrain.generate(state, cfg)
    rad = radiation.Radiation(cfg)
    sched.add(rad.step)
    sched.add(thermal.Thermal(cfg).step)
    flu = fluid.Fluid(cfg)
    sched.add(flu.step)
    chem.bind(state, cfg)
    chem.init_species(state)      # resume/replay overwrite this from the snapshot
    sched.add(chem.step)
    poly = polymers.Polymers(cfg, chem)
    poly.bind(state)
    poly.fluid = flu              # polymers drift with the tick's water fluxes
    sched.add(poly.step)
    sched.poly = poly
    sched.chem = chem
    aud = audit.Audit(cfg, chem, run_dir)
    sched.add(aud.step)
    sched.audit = aud
    return state, sched


def cmd_run(args):
    cfg = Config.load(args.config)
    run_dir = rundir.create(cfg)
    CLOCK.run_id = run_dir.name
    log = setup(cfg.logging.dir or run_dir / "logs", cfg.logging.level)
    log.info("run created", extra={"config": str(args.config), "device": args.device})
    state, sched = build_sim(cfg, run_dir, args.device)
    _attach_io(cfg, run_dir, state, sched)
    sched.developer_allowed = bool(args.developer)
    _serve(sched, run_dir, args)      # always on: the GUI attaches on demand
    _guarded_loop(sched, cfg, run_dir, state, args.ticks)


def _attach_io(cfg, run_dir, state, sched):
    from firmament.instruments.sampler import attach_instruments
    from firmament.io.events import EventLog
    from firmament.io.lineage import LineageDB
    from firmament.io.snapshot import SnapshotManager

    sched.events = EventLog(run_dir / "events.jsonl")        # causal history
    sched.analysis = EventLog(run_dir / "analysis.jsonl")    # derived observations
    sched.lineage = LineageDB(run_dir / "lineage.sqlite")
    sched.poly.lineage = sched.lineage
    snaps = SnapshotManager(run_dir / "snapshots", cfg)
    sched.snapshots = snaps
    every = max(1, int(cfg.run.snapshot_every_sim_days * 86400 / cfg.run.dt_seconds))
    attach_instruments(sched, cfg, run_dir)       # before the snapshot hook: a snapshot
                                                  # at tick T includes tick T's analysis

    def maybe_snapshot(s, tick):
        if tick > 0 and tick % every == 0:
            snaps.save(s, sched)
    sched.instruments.append(maybe_snapshot)


def _guarded_loop(sched, cfg, run_dir, state, ticks):
    import json
    import traceback
    log = get("cli")
    try:
        sched.loop(max_ticks=ticks)
        sched.snapshots.save(state, sched)          # shutdown snapshot
        _flush(sched)
    except Exception as e:  # crash record then re-raise
        last = sorted((run_dir / "snapshots").glob("tick_*"))
        (run_dir / "crash.json").write_text(json.dumps({
            "exception": repr(e), "traceback": traceback.format_exc(),
            "tick": state.tick, "last_good_snapshot": str(last[-1]) if last else None}))
        log.error("crash", extra={"exception": repr(e), "tick": state.tick})
        raise


def _flush(sched):
    for obj in (getattr(sched, "lineage", None), getattr(sched, "events", None),
                getattr(sched, "metrics", None)):
        if obj:
            obj.flush()


def _load_from_snapshot(args, run_dir: Path, snap=None):
    """Load a snapshot and reconstruct everything after it deterministically: analysis
    state is restored, derived logs written after the snapshot are dropped (the resumed
    process regenerates them identically), and logged causal history after the snapshot
    is queued for replay at its original ticks."""
    from firmament.io import events as ev
    from firmament.io.snapshot import SnapshotManager
    cfg = Config.load(run_dir / "config.yaml")
    CLOCK.run_id = run_dir.name
    setup(run_dir / "logs", cfg.logging.level)
    state, sched = build_sim(cfg, run_dir, args.device)
    snaps = SnapshotManager(run_dir / "snapshots", cfg)
    snaps.load(state, sched, snap)
    _attach_io(cfg, run_dir, state, sched)
    sched.sampler.load_state(snaps.loaded_meta.get("analysis_state", {}))
    dropped = sched.analysis.truncate_after(state.tick)
    dropped_rows = sched.metrics.truncate_after(state.tick)
    n = ev.replay_events(sched, run_dir / "events.jsonl")
    get("cli").info("snapshot loaded for continuation", extra={
        "tick": state.tick, "causal_n": state.causal_n, "causal_to_replay": n,
        "analysis_dropped": dropped, "metric_rows_dropped": dropped_rows})
    sched.developer_allowed = bool(getattr(args, "developer", False))
    return cfg, state, sched


def cmd_resume(args):
    run_dir = rundir.RUNS / args.run
    attempts = 0
    while True:
        cfg, state, sched = _load_from_snapshot(args, run_dir)
        get("cli").info("resumed", extra={"tick": state.tick, "attempt": attempts})
        if attempts == 0:
            _serve(sched, run_dir, args)
        try:
            _guarded_loop(sched, cfg, run_dir, state, args.ticks)
            return
        except Exception as e:
            attempts += 1
            if not getattr(args, "auto_restart", False) or not _is_transient(e):
                raise                 # deterministic crashes replay identically: never retry
            # a CUDA-level error poisons the GPU context: in-process retry is futile.
            # Re-exec a fresh process (marker file counts attempts across execs).
            marker = run_dir / ".restart_count"
            n = int(marker.read_text()) if marker.exists() else 0
            if n >= 3:
                marker.unlink(missing_ok=True)
                raise
            marker.write_text(str(n + 1))
            get("cli").warning("auto-restart: re-exec fresh process", extra={"attempt": n + 1})
            import os as _os
            import sys as _sys
            _os.execv(_sys.executable, [_sys.executable, "-m", "firmament.cli"] + _sys.argv[1:])


def _is_transient(e: Exception) -> bool:
    """Only device/driver faults can differ on retry. Everything else (AuditError,
    capacity exhaustion, bugs) is a deterministic function of the snapshot and would
    reproduce exactly, so retrying just burns hours."""
    msg = repr(e)
    return "Warp error" in msg or "CUDA" in msg or "cuda" in msg


def cmd_fork(args):
    from firmament.io.snapshot import fork_run
    new_dir = fork_run(rundir.RUNS / args.run, args.snapshot)
    print(new_dir.name)


def cmd_replay(args):
    """Re-derive a later state from a snapshot + the causal log, writing only to
    runs/_replays/<run>@<tick>/ (the original run is never touched). With --verify,
    compare byte-for-byte against the run's own snapshot at that tick."""
    import numpy as np
    import zarr

    from firmament.io.events import replay_events
    from firmament.io.snapshot import SnapshotManager
    run_dir = rundir.RUNS / args.run
    cfg = Config.load(run_dir / "config.yaml")
    state, sched = build_sim(cfg, run_dir, args.device)
    SnapshotManager(run_dir / "snapshots", cfg).load(
        state, sched, int(args.snapshot) if args.snapshot else None)
    out = rundir.RUNS / "_replays" / f"{args.run}@{args.to_tick}"
    out.mkdir(parents=True, exist_ok=True)
    sched.events = None                      # replay applies history; it never writes it
    replay_events(sched, run_dir / "events.jsonl")
    while state.tick < args.to_tick:
        sched.tick_once()
    path = SnapshotManager(out, cfg).save(state)
    print(f"replayed to tick {state.tick} -> {path}")
    if args.verify:
        ref = run_dir / "snapshots" / path.name
        a, b = zarr.open_group(str(ref), mode="r"), zarr.open_group(str(path), mode="r")
        bad = [k for k in a.array_keys() if not np.array_equal(a[k][:], b[k][:])]
        print("VERIFY: identical" if not bad else f"VERIFY: MISMATCH in {bad}")
        if bad:
            sys.exit(1)


def cmd_seed(args):
    from firmament.operator.console import place_seed
    run_dir = rundir.RUNS / args.run
    cfg, state, sched = _load_from_snapshot(args, run_dir)
    x, y = (int(v) for v in args.cell.split(","))
    place_seed(sched, state, args.sequence, x, y)
    sched.snapshots.save(state, sched)
    _flush(sched)
    print(f"seed placed at {x},{y}; event #0 logged")


def cmd_event(args):
    """Apply a natural event to a resumable run (offline path; GUI is the live path)."""
    import json

    from firmament.operator import console
    run_dir = rundir.RUNS / args.run
    cfg, state, sched = _load_from_snapshot(args, run_dir)
    console.request(sched, args.type, json.loads(args.params))
    sched.tick_once()                     # events apply on the next tick boundary
    sched.snapshots.save(state, sched)
    _flush(sched)
    print(f"{args.type} applied at tick {state.tick}")


def cmd_report(args):
    from firmament.io.report import write_report
    path = write_report(rundir.RUNS / args.run)
    print(path)


def cmd_inspect(args):
    run_dir = rundir.RUNS / args.run
    meta = rundir.read_meta(run_dir)
    snaps = sorted(p.name for p in (run_dir / "snapshots").glob("tick_*"))
    print(f"run: {meta['run_id']}  touched: {meta['touched']}  parent: {meta['parent']}")
    print(f"snapshots: {snaps[-5:]} ({len(snaps)} total)")


def _serve(sched, run_dir, args):
    """Always-on read-only bridge (writeup 4.10: sims run headless 24/7 and the GUI
    attaches to ANY of them without touching the sim). An idle socket costs nothing;
    view work only happens while a viewer is attached."""
    import atexit
    import os
    import socket
    import threading

    import uvicorn

    from firmament.server.api import make_app

    port = getattr(args, "port", 8000)
    with socket.socket() as probe:                 # preferred port taken? use ephemeral
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            with socket.socket() as s2:
                s2.bind(("", 0))
                port = s2.getsockname()[1]
    app = make_app(sched, run_dir)
    host = "0.0.0.0" if getattr(args, "lan", False) else "127.0.0.1"   # LAN is opt-in
    t = threading.Thread(
        target=uvicorn.run, kwargs=dict(app=app, host=host, port=port, log_level="warning"),
        daemon=True)
    t.start()
    run_dir_abs = Path(run_dir).resolve()
    meta = rundir.read_meta(run_dir_abs)
    meta["port"] = port
    meta["pid"] = os.getpid()
    rundir.write_meta(run_dir_abs, meta)

    def _clear():
        try:
            m = rundir.read_meta(run_dir_abs)
            m.pop("port", None)
            m.pop("pid", None)
            rundir.write_meta(run_dir_abs, m)
        except OSError:
            pass                      # best-effort exit cleanup; liveness probe covers stale pids
    atexit.register(_clear)
    get("cli").info("api serving", extra={"port": port, "host": host})


def main(argv=None):
    p = argparse.ArgumentParser(prog="firmament")
    p.add_argument("--device", default="cuda:0")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--ticks", type=int, default=None)
    r.add_argument("--port", type=int, default=8000)
    r.add_argument("--lan", action="store_true", help="expose the GUI on the LAN (default: localhost)")
    r.add_argument("--developer", action="store_true")
    r.set_defaults(fn=cmd_run)
    for name, fn in (("resume", cmd_resume),):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--ticks", type=int, default=None)
        s.add_argument("--port", type=int, default=8000)
        s.add_argument("--lan", action="store_true")
        s.add_argument("--developer", action="store_true")
        s.add_argument("--auto-restart", action="store_true", dest="auto_restart")
        s.set_defaults(fn=fn)
    f = sub.add_parser("fork")
    f.add_argument("--run", required=True)
    f.add_argument("--snapshot", default=None)
    f.set_defaults(fn=cmd_fork)
    rp = sub.add_parser("replay")
    rp.add_argument("--run", required=True)
    rp.add_argument("--snapshot", default=None)
    rp.add_argument("--to-tick", type=int, required=True)
    rp.add_argument("--verify", action="store_true")
    rp.set_defaults(fn=cmd_replay)
    sd = sub.add_parser("seed")
    sd.add_argument("--run", required=True)
    sd.add_argument("--sequence", required=True)
    sd.add_argument("--cell", required=True)
    sd.set_defaults(fn=cmd_seed)
    rp2 = sub.add_parser("report")
    rp2.add_argument("--run", required=True)
    rp2.set_defaults(fn=cmd_report)
    ev = sub.add_parser("event")
    ev.add_argument("--run", required=True)
    ev.add_argument("--type", required=True)
    ev.add_argument("--params", default="{}")
    ev.set_defaults(fn=cmd_event)
    i = sub.add_parser("inspect")
    i.add_argument("--run", required=True)
    i.set_defaults(fn=cmd_inspect)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
