"""Lineage DB: sqlite of copy events, buffered writes, pruning of extinct branches.

Pruning policy (Phase 6): keep all ancestors of the living; extinct branches are
summarized (count, tick span, max depth) into `extinct_summary`.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from firmament.io.logging import get

log = get("lineage")


class LineageDB:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.db = sqlite3.connect(self.path)
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS copies(
            child_id INTEGER PRIMARY KEY, parent_id INTEGER, tick INTEGER,
            cell INTEGER, length INTEGER, mutations INTEGER);
        CREATE INDEX IF NOT EXISTS idx_parent ON copies(parent_id);
        CREATE TABLE IF NOT EXISTS extinct_summary(
            root_id INTEGER PRIMARY KEY, n_descendants INTEGER,
            first_tick INTEGER, last_tick INTEGER, max_depth INTEGER);
        """)
        self.buf: list[tuple] = []

    def add_copies(self, tick: int, events: list[tuple]) -> None:
        # events: (child_id, parent_id, cell, length, mutations) sorted by child_id
        self.buf.extend((c, p, tick, cell, ln, mu) for c, p, cell, ln, mu in events)
        if len(self.buf) >= 10000:
            self.flush()

    def flush(self) -> None:
        if self.buf:
            self.db.executemany("INSERT OR REPLACE INTO copies VALUES(?,?,?,?,?,?)", self.buf)
            self.db.commit()
            self.buf.clear()

    def depth_of(self, pid: int) -> int:
        d = 0
        cur = pid
        while cur and d < 100000:
            row = self.db.execute("SELECT parent_id FROM copies WHERE child_id=?", (cur,)).fetchone()
            if row is None:
                break
            cur = row[0]
            d += 1
        return d

    def path_to_seed(self, pid: int, limit: int = 200) -> list[int]:
        out = [pid]
        cur = pid
        while cur and len(out) < limit:
            row = self.db.execute("SELECT parent_id FROM copies WHERE child_id=?", (cur,)).fetchone()
            if row is None or row[0] == 0:
                break
            cur = row[0]
            out.append(cur)
        return out

    def prune(self, living_ids: set[int]) -> None:
        """Summarize and drop branches with no living descendants."""
        self.flush()
        keep: set[int] = set()
        for pid in living_ids:
            for a in self.path_to_seed(pid, limit=100000):
                if a in keep:
                    break
                keep.add(a)
        rows = self.db.execute("SELECT child_id, parent_id, tick FROM copies").fetchall()
        dead = [(c, p, t) for c, p, t in rows if c not in keep]
        if not dead:
            return
        # summarize extinct subtrees rooted at children of kept nodes
        by_root: dict[int, list[tuple]] = {}
        for c, p, t in dead:
            root = c if p in keep or p == 0 else None
            if root is not None:
                by_root.setdefault(root, []).append((c, p, t))
        for root, items in by_root.items():
            ticks = [t for _, _, t in items]
            self.db.execute("INSERT OR REPLACE INTO extinct_summary VALUES(?,?,?,?,?)",
                            (root, len(items), min(ticks), max(ticks), 0))
        self.db.executemany("DELETE FROM copies WHERE child_id=?", [(c,) for c, _, _ in dead])
        self.db.commit()
        log.info("lineage pruned", extra={"removed": len(dead), "kept": len(keep)})

    def close(self) -> None:
        self.flush()
        self.db.close()
