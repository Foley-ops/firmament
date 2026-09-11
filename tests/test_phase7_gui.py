"""Phase 7 acceptance: every GUI panel's data source works on a DEAD world (zero
polymers) without errors; the live stream attaches/detaches leaving no residue;
sim throughput with a viewer attached differs < 5% from detached."""
from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.conftest import build_test_sim, tiny_cfg


@pytest.fixture
def dead_world(tmp_path, device):
    from firmament.instruments.sampler import attach_instruments

    cfg = tiny_cfg(n=32, cap=1000)
    run_dir = tmp_path / "runs" / "dead-世界-000000-abcdef"
    run_dir.mkdir(parents=True)
    state, sched = build_test_sim(cfg, device, run_dir)
    attach_instruments(sched, cfg, run_dir)
    (run_dir / "meta.json").write_text(json.dumps(
        {"run_id": run_dir.name, "parent": None, "created": "now", "touched": False}))
    (run_dir / "config.yaml").write_text(cfg.canonical_yaml())
    import firmament.io.rundir as rd
    old_runs = rd.RUNS
    rd.RUNS = tmp_path / "runs"
    for _ in range(120):                       # a few samples land in metrics
        sched.tick_once()
    sched.metrics.flush()
    from firmament.server.api import make_app
    app = make_app(sched, run_dir)
    yield TestClient(app), sched, state
    rd.RUNS = old_runs


class TestPanelsOnDeadWorld:
    def test_run_picker(self, dead_world):
        client, sched, _ = dead_world
        runs = client.get("/api/runs").json()
        assert len(runs) == 1
        assert runs[0]["status"] == "running"
        assert runs[0]["touched"] is False

    def test_meta(self, dead_world):
        client, _, _ = dead_world
        m = client.get("/api/meta").json()
        assert m["world"] == [32, 32]
        assert "replicase" in m["motifs"]
        assert len(m["species"]) == 22

    def test_map_layers_render(self, dead_world):
        client, sched, _ = dead_world
        from firmament.server.api import build_view
        layers = ["elevation", "water", "temp", "light", "polymer_density",
                  "compartments", "lineage", "species:PP", "motif:0"]
        view = build_view(sched, layers, ds=1)
        assert set(view["layers"]) == set(layers)
        for name, l in view["layers"].items():
            assert l["shape"] == [32, 32]
            import base64
            arr = np.frombuffer(base64.b64decode(l["data"]), dtype=np.float32)
            assert arr.size == 32 * 32
            assert np.isfinite(arr).all(), f"layer {name} has non-finite values"

    def test_inspector_cell_and_missing_polymer(self, dead_world):
        client, _, _ = dead_world
        d = client.get("/api/inspect/cell?x=5&y=5").json()
        assert "species" in d and "temp" in d and d["polymers"] == []
        r = client.get("/api/inspect/polymer?id=1")
        assert r.status_code == 404              # dead world: no polymer, no crash

    def test_charts_metrics(self, dead_world):
        client, _, _ = dead_world
        d = client.get("/api/runs/dead-世界-000000-abcdef/metrics?names=n_polymers").json()
        assert "tick" in d and "n_polymers" in d
        assert all(v == 0 for v in d["n_polymers"])

    def test_lineage_tree_empty(self, dead_world):
        client, _, _ = dead_world
        assert client.get("/api/lineage/tree").json() == []

    def test_event_timeline(self, dead_world):
        client, _, _ = dead_world
        assert isinstance(client.get("/api/events").json(), list)

    def test_console_confirm_flow(self, dead_world):
        client, sched, _ = dead_world
        r = client.post("/api/console/rain", json={"intensity_mm": 5.0, "radius": 10}).json()
        assert "confirm_token" in r              # step 1: no token -> challenge
        assert not sched.pending_events
        r2 = client.post("/api/console/rain",
                         json={"intensity_mm": 5.0, "radius": 10,
                               "confirm_token": r["confirm_token"]}).json()
        assert "applied_at_tick" in r2
        assert sched.pending_events              # queued for the tick boundary
        sched.tick_once()                        # applies without error on dead world
        assert not sched.pending_events

    def test_developer_gate(self, dead_world):
        client, _, _ = dead_world
        r = client.post("/api/developer/set_species",
                        json={"species": "PP", "x": 1, "y": 1, "count": 5})
        assert r.status_code == 403              # locked until explicitly enabled

    def test_time_control(self, dead_world):
        client, sched, _ = dead_world
        client.post("/api/time", json={"mode": "paused"})
        assert sched.mode == "paused"
        client.post("/api/time", json={"mode": "throttled", "target_tps": 12})
        assert sched.mode == "throttled" and sched.target_tps == 12
        client.post("/api/time", json={"mode": "unbounded"})

    def test_websocket_attach_stream_detach_no_residue(self, dead_world):
        client, sched, _ = dead_world
        assert sched.viewers == 0
        with client.websocket_connect("/ws/state") as ws:
            assert sched.viewers == 1
            for _ in range(3):                   # stream needs a view cache: tick it
                sched.tick_once()
            ws.send_text(json.dumps({"layers": ["elevation", "water"]}))
            frame = json.loads(ws.receive_text())
            assert "tick" in frame and "layers" in frame
        deadline = time.time() + 2
        while sched.viewers != 0 and time.time() < deadline:
            time.sleep(0.05)
        assert sched.viewers == 0
        assert sched.view_cache is None          # zero residue

    def test_gui_files_served(self, dead_world):
        client, _, _ = dead_world
        page = client.get("/").text
        assert "FIRMAMENT" in page and "app.js" in page
        assert client.get("/app.js").status_code == 200
        assert client.get("/style.css").status_code == 200


@pytest.mark.slow
def test_throughput_with_viewer_within_5pct(tmp_path, device):
    """The sim must not slow more than 5% while a viewer streams."""
    cfg = tiny_cfg(n=32, cap=1000)
    state, sched = build_test_sim(cfg, device, tmp_path / "run")
    from firmament.server.api import make_app
    app = make_app(sched, tmp_path / "run")
    (tmp_path / "run" / "meta.json").write_text(json.dumps(
        {"run_id": "x", "parent": None, "created": "n", "touched": False}))
    client = TestClient(app)

    def measure(seconds):
        t0, tick0 = time.perf_counter(), state.tick
        while time.perf_counter() - t0 < seconds:
            sched.tick_once()
        return (state.tick - tick0) / (time.perf_counter() - t0)

    measure(3)                                   # warmup
    base = measure(10)
    stop = threading.Event()

    def viewer():
        with client.websocket_connect("/ws/state") as ws:
            while not stop.is_set():
                try:
                    ws.receive_text()
                except Exception:
                    break

    t = threading.Thread(target=viewer, daemon=True)
    t.start()
    time.sleep(0.3)
    assert sched.viewers == 1
    attached = measure(10)
    stop.set()
    slowdown = (base - attached) / base
    assert slowdown < 0.05, f"viewer slows sim by {slowdown:.1%} (base {base:.0f} vs {attached:.0f} tps)"
