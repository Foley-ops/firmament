"""Real SIGINT (what Ctrl-C sends) to a process inside cli._guarded_loop at tick 230,
after a tick-200 snapshot. RUNBOOK: 'Ctrl-C (or kill). A shutdown snapshot is written on
clean exit'. Does lineage.sqlite get the buffered copy events?"""
import sys
sys.path.insert(0, "/home/nick/Life/firmament")
sys.path.insert(0, "/tmp/claude-1000/-home-nick-Life/e538cd8c-d5ab-4b90-9b16-07d0ee1b070e/scratchpad/court/defense-observe-lineage-buffer-invisible-and-lost")
import json, os, shutil, signal, sqlite3, subprocess, tempfile
from pathlib import Path


def child(workdir):
    import defense_crash_path as d
    from firmament import cli
    orig_guard = cli._guarded_loop

    def guarded_with_sigint(sched, cfg, run_dir, state, ticks):
        def ctrl_c(s, tick):
            if tick == 230:
                os.kill(os.getpid(), signal.SIGINT)   # exactly what a terminal Ctrl-C sends
        sched.instruments.append(ctrl_c)
        return orig_guard(sched, cfg, run_dir, state, ticks)
    cli._guarded_loop = guarded_with_sigint
    d.child(workdir, 3000)


def main():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    w = Path(tempfile.mkdtemp(prefix="defsigint_"))
    try:
        p = subprocess.run([sys.executable, __file__, "child", str(w)], env=env,
                           capture_output=True, text=True, timeout=600)
        last = [l for l in p.stderr.strip().splitlines() if l.strip()][-1]
        truth = [json.loads(l) for l in open(w / "run" / "truth.jsonl")]
        snaps = sorted(x.name for x in (w / "run" / "snapshots").glob("tick_*.zarr"))
        c = sqlite3.connect(w / "run" / "lineage.sqlite")
        rows = c.execute("SELECT COUNT(*) FROM copies").fetchone()[0]
        c.close()
        print(f"child exit code {p.returncode}; last stderr line: {last}")
        print(f"snapshots on disk: {snaps}  crash.json exists: {(w/'run'/'crash.json').exists()}")
        print(f"copy events that happened: {len(truth)} (tick<=200: {sum(1 for t in truth if t[0]<=200)})")
        print(f"rows in lineage.sqlite after Ctrl-C: {rows}")
    finally:
        shutil.rmtree(w, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2])
    else:
        main()
