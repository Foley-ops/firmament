"""Metrics writer: buffered Parquet, one row per sample, never rotated away."""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from firmament.io.logging import get

log = get("metrics")


class MetricsWriter:
    def __init__(self, metrics_dir: Path, flush_rows: int = 200) -> None:
        self.dir = Path(metrics_dir)
        self.dir.mkdir(exist_ok=True)
        self.buf: list[dict] = []
        self.flush_rows = flush_rows
        self.part = len(list(self.dir.glob("part_*.parquet")))

    def add(self, row: dict) -> None:
        self.buf.append(row)
        if len(self.buf) >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        keys = sorted({k for r in self.buf for k in r})
        table = pa.table({k: [r.get(k) for r in self.buf] for k in keys})
        pq.write_table(table, self.dir / f"part_{self.part:06d}.parquet")
        self.part += 1
        self.buf.clear()

    def truncate_after(self, tick: int) -> int:
        """Drop rows newer than `tick` (resume regenerates them identically)."""
        self.buf = [r for r in self.buf if r["tick"] <= tick]
        t = self.read_all()
        if t is None:
            return 0
        rows = t.to_pylist()
        keep = [r for r in rows if r["tick"] <= tick]
        if len(keep) == len(rows):
            return 0
        for f in self.dir.glob("part_*.parquet"):
            f.unlink()
        self.part = 0
        if keep:
            pq.write_table(pa.Table.from_pylist(keep), self.dir / "part_000000.parquet")
            self.part = 1
        return len(rows) - len(keep)

    def read_all(self):
        import pyarrow.dataset as ds
        files = sorted(self.dir.glob("part_*.parquet"))
        if not files:
            return None
        return ds.dataset([str(f) for f in files]).to_table()
