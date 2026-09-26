"""Natural path: exact tests/test_phase8_console.py EVENTS schedule, unmodified audit cadence,
simply run past tick 400. Variant 'no meteor' drops only the meteor event."""
import sys, os, tempfile, logging, time
sys.path.insert(0, "/home/nick/Life/firmament")
from pathlib import Path
from tests.conftest import tiny_cfg, build_test_sim
from tests.test_phase8_console import EVENTS
from firmament.core.audit import AuditError
from firmament.operator import console
SCR = "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/defense-env-meteor-energy-ledger-audit-crash"
logging.getLogger("firmament.fluid").setLevel(logging.ERROR)

def go(label, every, events, stop):
    tmp = Path(tempfile.mkdtemp(dir=SCR)); os.chdir(tmp)
    cfg = tiny_cfg(n=32, cap=1000)
    if every is not None:
        cfg.run.audit_every_ticks = every
    s, sched = build_test_sim(cfg, "cpu", tmp / "run")
    todo = list(events)
    try:
        while s.tick < stop:
            if todo and s.tick + 1 == todo[0][0]:
                _, ev, p = todo.pop(0); console.request(sched, ev, p)
            sched.tick_once()
        print(f"[{label}] audit_every={cfg.run.audit_every_ticks}: ran to tick {s.tick}, all audits PASSED")
    except AuditError as e:
        print(f"[{label}] audit_every={cfg.run.audit_every_ticks}: AuditError: {e}")

t = time.time()
go("test_phase8 schedule", None, EVENTS, 1001)            # tiny_cfg cadence = 500
go("test_phase8 schedule minus meteor", None, [e for e in EVENTS if e[1] != "meteor"], 1001)
go("test_phase8 schedule", 1000, EVENTS, 1001)            # shipped default cadence
go("test_phase8 schedule minus meteor", 1000, [e for e in EVENTS if e[1] != "meteor"], 1001)
print(f"elapsed {time.time()-t:.1f}s")
