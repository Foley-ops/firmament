import json
from pathlib import Path
H = Path(__file__).resolve().parent
ref = json.loads((H / "result_ref.json").read_text())
rm, rl = ref["final"]["metric_ticks"], {tuple(r) for r in ref["final"]["lineage"]}
print(f"REF (uninterrupted, clean --ticks exit): final {ref['final']['snapshots_tail']}, "
      f"{len(rm)} metric rows [{rm[0]}..{rm[-1]}], {len(rl)} lineage rows")
for mode in ("SIGINT_PG", "SIGTERM", "SIGKILL", "CRASH"):
    r = json.loads((H / f"result_{mode}.json").read_text())
    m, l = r["final"]["metric_ticks"], {tuple(x) for x in r["final"]["lineage"]}
    miss_m = sorted(set(rm) - set(m))
    miss_l = sorted(rl - l)
    extra_l = sorted(l - rl)
    ids = {c for c, _, _ in miss_l}
    orphans = [(c, p) for c, p, _ in l if p in ids]
    print(f"\n{mode}: rc={r['stop_rc']} stopped at tick {r['tick_reached']}; after stop: {r['after_stop']}")
    print(f"  after resume -> final {r['final']['snapshots_tail']}, {len(m)} metric rows, {len(l)} lineage rows")
    print(f"  metric ticks MISSING vs ref: {len(miss_m)}"
          + (f" [{miss_m[0]}..{miss_m[-1]}] (all <= snapshot tick 300: {all(t <= 300 for t in miss_m)})" if miss_m else ""))
    print(f"  lineage copy rows MISSING vs ref: {len(miss_l)}"
          + (f" ticks {miss_l[0][2]}..{max(t for *_, t in miss_l)} (all <= 300: {all(t <= 300 for *_, t in miss_l)})" if miss_l else "")
          + f"; extra rows not in ref: {len(extra_l)}; surviving rows whose parent record is gone: {len(orphans)}")
