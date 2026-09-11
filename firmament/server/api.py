"""Sim <-> GUI bridge. Read-only state stream + console commands.

Design: the sim never waits on the GUI. A view cache (downsampled layers + tails)
is refreshed by the scheduler loop ONLY while viewers are attached; websockets
stream the cache. A disconnecting viewer leaves zero residue.
"""
from __future__ import annotations

import asyncio
import base64
import json
import secrets
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from firmament.io import rundir
from firmament.io.logging import get, set_level

log = get("server")


def _b64(arr: np.ndarray, ds: int = 4) -> dict:
    a = np.asarray(arr[::ds, ::ds], dtype=np.float32)
    return {"shape": list(a.shape), "min": float(np.nanmin(a)), "max": float(np.nanmax(a)),
            "data": base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()}


def build_view(sched, layers: list[str], ds: int) -> dict:
    """Called from the sim loop thread while viewers > 0. Uses instrument-style
    host copies only; never touches live GPU state beyond a read."""
    s = sched.state
    out = {"tick": s.tick, "sim_seconds": s.tick * sched.dt, "mode": sched.mode,
           "layers": {}, "tps": getattr(sched, "tps", 0.0)}
    chem = sched.chem
    for name in layers:
        if name == "elevation":
            out["layers"][name] = _b64(s.elevation.numpy(), ds)
        elif name == "water":
            out["layers"][name] = _b64(s.water_depth.numpy(), ds)
        elif name == "temp":
            out["layers"][name] = _b64(s.temp.numpy()[1], ds)
        elif name == "light":
            out["layers"][name] = _b64(s.light.numpy(), ds)
        elif name == "compartments":
            out["layers"][name] = _b64((s.compartment_id.numpy() > 0).astype(np.float32), ds)
        elif name == "polymer_density":
            h, w = s.shape
            alive = s.p_state.numpy() > 0
            dens = np.bincount(s.p_cell.numpy()[alive], minlength=h * w).reshape(h, w)
            out["layers"][name] = _b64(dens.astype(np.float32), ds)
        elif name == "lineage":
            h, w = s.shape
            alive = s.p_state.numpy() > 0
            par = s.p_parent.numpy()[alive].astype(np.float64)
            cells = s.p_cell.numpy()[alive]
            img = np.zeros(h * w, dtype=np.float32)
            if len(cells):
                img[cells] = (par % 97).astype(np.float32) + 1.0
            out["layers"][name] = _b64(img.reshape(h, w), ds)
        elif name.startswith("species:"):
            sp = name.split(":", 1)[1]
            if sp in chem.index:
                out["layers"][name] = _b64(s.species.numpy()[chem.index[sp]].astype(np.float32), ds)
        elif name.startswith("motif:"):
            b = int(name.split(":", 1)[1])
            h, w = s.shape
            alive = (s.p_state.numpy() > 0) & ((s.p_motifs.numpy() & (1 << b)) != 0)
            img = np.bincount(s.p_cell.numpy()[alive], minlength=h * w).reshape(h, w)
            out["layers"][name] = _b64(img.astype(np.float32), ds)
    return out


def make_app(sched, run_dir: Path) -> FastAPI:
    app = FastAPI(title="FIRMAMENT")
    sched.viewers = 0
    sched.view_cache = None
    sched.view_layers = ["elevation", "water", "temp", "light", "polymer_density"]
    sched.view_ds = max(1, sched.state.shape[0] // 256)
    tokens: dict[str, dict] = {}
    developer_enabled = {"on": False}

    def refresh_view(s, tick):
        if sched.viewers > 0:
            now = time.monotonic()
            if now - getattr(sched, "_last_view", 0) > 0.5:
                sched.view_cache = build_view(sched, sched.view_layers, sched.view_ds)
                sched._last_view = now
    sched.instruments.append(refresh_view)

    # ---------- run picker / manager ----------
    @app.get("/api/runs")
    def runs():
        import os
        out = []
        for m in rundir.list_runs():
            d = Path(m["dir"])
            snaps = sorted(p.name for p in (d / "snapshots").glob("tick_*.zarr"))
            ev = []
            evp = d / "events.jsonl"
            if evp.exists():
                lines = evp.read_text().strip().splitlines()
                ev = [json.loads(x) for x in lines[-3:]] if lines else []
            mine = m["run_id"] == run_dir.name
            alive = mine
            if not mine and m.get("pid"):
                try:
                    os.kill(m["pid"], 0)          # liveness probe, no signal sent
                    alive = True
                except (OSError, ProcessLookupError):
                    alive = False
            status = "running" if alive else ("resumable" if snaps else "empty")
            out.append({"run_id": m["run_id"], "status": status, "touched": m["touched"],
                        "parent": m["parent"], "snapshots": len(snaps),
                        "port": m.get("port") if alive else None, "mine": mine,
                        "last_milestone": next((e["event"] for e in reversed(ev)
                                                if e.get("component") == "milestones"), None),
                        "tick": sched.state.tick if mine else None,
                        "tps": round(getattr(sched, "tps", 0.0), 1) if mine else None})
        return out

    @app.get("/api/runs/{rid}/snapshots")
    def snapshots(rid: str):
        return sorted(p.name for p in (rundir.RUNS / rid / "snapshots").glob("tick_*.zarr"))

    @app.post("/api/runs/{rid}/fork")
    def fork(rid: str, snapshot: str | None = None):
        from firmament.io.snapshot import fork_run
        new = fork_run(rundir.RUNS / rid, snapshot)
        return {"run_id": new.name}

    @app.get("/api/runs/{rid}/metrics")
    def run_metrics(rid: str, names: str = "", tail: int = 2000):
        from firmament.io.metrics import MetricsWriter
        mw = sched.metrics if rid == run_dir.name else MetricsWriter(rundir.RUNS / rid / "metrics")
        if rid == run_dir.name:
            mw.flush()
        t = mw.read_all()
        if t is None:
            return {}
        d = t.to_pydict()
        keep = ["tick"] + ([n for n in names.split(",") if n] or
                           [c for c in d if c != "tick"])
        return {k: d[k][-tail:] for k in keep if k in d}

    # ---------- time control ----------
    @app.post("/api/time")
    async def time_ctl(body: dict):
        mode = body.get("mode")
        if mode in ("unbounded", "throttled", "paused"):
            sched.mode = mode
        if mode == "step":
            sched.step_budget = int(body.get("steps", 1))
            sched.mode = "step"
        if "target_tps" in body:
            sched.target_tps = float(body["target_tps"])
        return {"mode": sched.mode, "target_tps": sched.target_tps}

    @app.post("/api/loglevel")
    def loglevel(body: dict):
        set_level(body["level"])
        return {"ok": True}

    # ---------- inspector ----------
    def _view():
        # Build-guide 8: the GUI never reads live GPU memory — read-only copies only.
        # The instrument sampler refreshes this host-side view every metrics interval.
        v = getattr(sched, "latest_view", None)
        if v is None:
            raise HTTPException(status_code=503, detail="view warming up (first sample pending)")
        return v

    @app.get("/api/inspect/cell")
    def inspect_cell(x: int, y: int):
        v = _view()
        chem = sched.chem
        w = v["shape"][1]
        spec = v["species"][:, y, x]
        alive = v["p_state"] > 0
        here = np.nonzero(alive & (v["p_cell"] == y * w + x))[0]
        return {
            "cell": [x, y], "as_of_tick": v["tick"],
            "elevation": float(v["elevation"][y, x]),
            "water_depth": float(v["water_depth"][y, x]),
            "temp": [float(t) for t in v["temp"][:, y, x]],
            "light": float(v["light"][y, x]),
            "compartment": int(v["compartment_id"][y, x]),
            "membrane_L": int(v["membrane_store"][y, x]) if "membrane_store" in v else 0,
            "species": {n: int(spec[chem.index[n]]) for n in chem.names},
            "polymers": [int(v["p_id"][p]) for p in here[:50]],
        }

    @app.get("/api/inspect/polymer")
    def inspect_polymer(id: int):
        v = _view()
        slots = np.nonzero((v["p_id"] == id) & (v["p_state"] > 0))[0]
        if len(slots) == 0:
            return JSONResponse({"error": "not alive"}, status_code=404)
        p = int(slots[0])
        ln = int(v["p_len"][p])
        seq = v["p_seq"][p, :ln]
        names = ["", "M1", "M2", "M3", "M4"]
        poly = sched.poly
        motifs = []
        for mi, row in enumerate(poly.motif_np):
            k = len(row)
            for pos in range(ln - k + 1):
                if (seq[pos:pos + k] == row).all():
                    motifs.append({"name": poly.motif_names[mi], "pos": pos, "len": k})
        lineage_path = sched.lineage.path_to_seed(id) if sched.lineage else []
        return {"id": id, "parent": int(v["p_parent"][p]),
                "cell": int(v["p_cell"][p]), "length": ln,
                "state": int(v["p_state"][p]), "born_tick": int(v["p_born"][p]),
                "mutations": int(v["p_mut"][p]) if "p_mut" in v else 0,
                "sequence": "".join(names[m] for m in seq),
                "motifs": motifs, "lineage_path": lineage_path}

    @app.get("/api/lineage/tree")
    def lineage_tree(limit: int = 2000):
        v = _view()
        alive = np.nonzero(v["p_state"] > 0)[0][:limit]
        return [{"id": int(v["p_id"][p]), "parent": int(v["p_parent"][p]),
                 "len": int(v["p_len"][p]), "cell": int(v["p_cell"][p]),
                 "motifs": int(v["p_motifs"][p])} for p in alive]

    @app.get("/api/events")
    def events(tail: int = 200):
        return sched.events.tail(tail) if getattr(sched, "events", None) else []

    # ---------- god console (confirm step) & developer mode ----------
    @app.post("/api/console/{event}")
    def console_cmd(event: str, body: dict):
        from firmament.operator import console
        tok = body.pop("confirm_token", None)
        if tok is None:
            t = secrets.token_hex(8)
            tokens[t] = {"event": event, "params": body, "ts": time.time()}
            return {"confirm_token": t, "event": event, "params": body,
                    "message": "repeat request with confirm_token to execute"}
        held = tokens.pop(tok, None)
        if held is None or held["event"] != event or time.time() - held["ts"] > 300:
            return JSONResponse({"error": "invalid or expired confirm token"}, status_code=400)
        rec = console.request(sched, event, held["params"])
        return {"applied_at_tick": rec["tick"], "event_n": rec["n"]}

    @app.post("/api/developer/enable")
    def dev_enable(body: dict):
        developer_enabled["on"] = bool(body.get("on", False))
        return {"developer": developer_enabled["on"],
                "warning": "edits will permanently mark this run TOUCHED"}

    @app.post("/api/developer/{action}")
    def dev_cmd(action: str, body: dict):
        if not developer_enabled["on"]:
            return JSONResponse({"error": "developer mode not enabled"}, status_code=403)
        from firmament.operator import developer as dev
        fn = {"set_field": dev.set_field, "set_species": dev.set_species,
              "edit_polymer": dev.edit_polymer}.get(action)
        if fn is None:
            return JSONResponse({"error": "unknown action"}, status_code=404)
        def run():
            fn(sched, **body)
        sched.pending_events.append(lambda s, t: run())
        return {"queued": action}

    @app.get("/api/meta")
    def meta():
        m = rundir.read_meta(run_dir)
        return {"run_id": run_dir.name, "touched": m["touched"],
                "world": list(sched.state.shape), "dt": sched.dt,
                "species": sched.chem.names,
                "motifs": sched.poly.motif_names,
                "n_reactions": sched.chem.n_react}

    # ---------- live stream ----------
    @app.websocket("/ws/state")
    async def ws_state(ws: WebSocket):
        await ws.accept()
        sched.viewers += 1
        log.info("viewer attached", extra={"viewers": sched.viewers})
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(ws.receive_text(), timeout=0.5)
                    req = json.loads(msg)
                    if "layers" in req:
                        sched.view_layers = req["layers"]
                except asyncio.TimeoutError:
                    pass
                if sched.view_cache is not None:
                    await ws.send_text(json.dumps(sched.view_cache))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            sched.viewers -= 1
            if sched.viewers == 0:
                sched.view_cache = None       # zero residue
            log.info("viewer detached", extra={"viewers": sched.viewers})

    gui_dir = Path(__file__).resolve().parents[2] / "gui"
    if gui_dir.exists():
        app.mount("/", StaticFiles(directory=str(gui_dir), html=True), name="gui")
    return app
