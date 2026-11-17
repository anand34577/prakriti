"""Crash-safe building blocks used by every pipeline stage.

Guarantees:
- A stage marked done is never re-run (unless forced).
- A per-item ledger lets a half-finished stage skip work already completed.
- Files and checkpoints are written atomically (tmp + fsync + rename), so a
  power cut or killed process never leaves a corrupt file behind.
- Training always resumes from the newest *valid* checkpoint.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path: str | os.PathLike, obj) -> None:
    atomic_write_bytes(path, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))


def atomic_save(save_fn: Callable[[str], None], path: str | os.PathLike) -> None:
    """Atomic wrapper for savers that take a filename (e.g. torch.save(obj, p))."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    save_fn(str(tmp))
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


class StageState:
    """Done-markers for whole stages, stored under <root>/state/."""

    def __init__(self, root: str | os.PathLike):
        self.dir = Path(root) / "state"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _marker(self, stage: str) -> Path:
        return self.dir / f"{stage}.done"

    def is_done(self, stage: str) -> bool:
        return self._marker(stage).exists()

    def mark_done(self, stage: str, info: dict | None = None) -> None:
        atomic_write_json(self._marker(stage), {"finished_at": time.time(), **(info or {})})

    def reset(self, stage: str) -> None:
        self._marker(stage).unlink(missing_ok=True)


class ItemLedger:
    """Append-only JSONL record of finished items inside a stage.

    Each line is flushed + fsynced, so after a crash at most the item being
    processed at that moment is redone. A torn last line is ignored.
    """

    def __init__(self, path: str | os.PathLike, fsync_every: int = 1):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync_every = max(1, fsync_every)
        self._done: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self._done[rec["id"]] = rec
                    except (json.JSONDecodeError, KeyError):
                        continue  # torn write from a crash
        self._fh = open(self.path, "a", encoding="utf-8")
        self._pending = 0

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._done

    def __len__(self) -> int:
        return len(self._done)

    def records(self) -> list[dict]:
        return list(self._done.values())

    def add(self, item_id: str, **fields) -> None:
        rec = {"id": item_id, **fields}
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._pending += 1
        if self._pending >= self.fsync_every:
            self.flush()
        self._done[item_id] = rec

    def flush(self) -> None:
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._pending = 0

    def pending(self, ids: Iterable[str]) -> Iterator[str]:
        return (i for i in ids if i not in self._done)

    def close(self) -> None:
        self.flush()
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


_STEP_RE = re.compile(r"step_(\d+)\.pt$")


class CheckpointManager:
    """Keeps the last N checkpoints; resumes from the newest one that loads."""

    def __init__(self, ckpt_dir: str | os.PathLike, keep_last: int = 3):
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

    def list(self) -> list[tuple[int, Path]]:
        found = []
        for p in self.dir.glob("step_*.pt"):
            m = _STEP_RE.search(p.name)
            if m:
                found.append((int(m.group(1)), p))
        return sorted(found)

    def save(self, step: int, save_fn: Callable[[str], None]) -> Path:
        path = self.dir / f"step_{step:08d}.pt"
        atomic_save(save_fn, path)
        atomic_write_json(self.dir / "latest.json", {"step": step, "path": path.name})
        for _, old in self.list()[: -self.keep_last]:
            old.unlink(missing_ok=True)
        return path

    def load_latest(self, load_fn: Callable[[str], object]):
        """Returns (step, obj) from the newest loadable checkpoint, or (0, None)."""
        for step, path in reversed(self.list()):
            try:
                return step, load_fn(str(path))
            except Exception as e:  # corrupt file: fall back to the previous one
                print(f"[resume] skipping unreadable checkpoint {path.name}: {e}")
        return 0, None
