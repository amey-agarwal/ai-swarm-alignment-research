"""Append-only, hash-chained JSONL log.

Each line is a JSON object:
    {"seq": i, ...record..., "prev_hash": <hex>, "hash": <hex>}
where hash = SHA-256 of the canonical JSON of the object without "hash".
The first record's prev_hash is GENESIS. `verify()` recomputes the chain end to end.

The writer keeps its own byte counter (`bytes_written`) which the governor reads,
so the run directory never has to be polled for log growth.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GENESIS = "0" * 64
RESERVED_KEYS = frozenset({"seq", "prev_hash", "hash"})


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def record_hash(entry_without_hash: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(entry_without_hash).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    n_records: int
    last_hash: str
    error: str | None = None
    line_no: int | None = None


def verify(path: str | Path) -> VerifyResult:
    """Verify the whole chain. Returns ok=False with the first failing line on any problem."""
    path = Path(path)
    prev = GENESIS
    n = 0
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return VerifyResult(False, 0, GENESIS, f"cannot read log: {exc}")
    if raw and not raw.endswith(b"\n"):
        return VerifyResult(False, 0, GENESIS, "log does not end with a newline (truncated?)")
    for i, line in enumerate(raw.decode("utf-8", errors="strict").splitlines(), start=1):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            return VerifyResult(False, n, prev, f"invalid JSON: {exc}", i)
        if not isinstance(entry, dict):
            return VerifyResult(False, n, prev, "record is not an object", i)
        if line != canonical(entry):
            return VerifyResult(False, n, prev, "record is not in canonical form", i)
        h = entry.pop("hash", None)
        if entry.get("seq") != n:
            return VerifyResult(False, n, prev, f"seq {entry.get('seq')!r} != expected {n}", i)
        if entry.get("prev_hash") != prev:
            return VerifyResult(False, n, prev, "prev_hash does not link to previous record", i)
        if h != record_hash(entry):
            return VerifyResult(False, n, prev, "hash mismatch (record modified)", i)
        prev = h
        n += 1
    return VerifyResult(True, n, prev)


class HashChainWriter:
    """Append-only writer. Re-opening an existing log verifies it and continues the chain."""

    def __init__(self, path: str | Path, fsync: bool = False):
        self.path = Path(path)
        self._fsync = fsync
        self._lock = threading.Lock()
        if self.path.exists():
            res = verify(self.path)
            if not res.ok:
                raise ValueError(f"refusing to append to invalid chain {self.path}: {res.error}")
            self._seq, self._prev = res.n_records, res.last_hash
            self._bytes = self.path.stat().st_size
        else:
            self._seq, self._prev, self._bytes = 0, GENESIS, 0
        self._fh = open(self.path, "ab")

    @property
    def bytes_written(self) -> int:
        """Total size of the log in bytes, from the writer's own counter."""
        with self._lock:
            return self._bytes

    @property
    def n_records(self) -> int:
        with self._lock:
            return self._seq

    @property
    def last_hash(self) -> str:
        with self._lock:
            return self._prev

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(record, dict):
            raise TypeError("record must be a dict")
        clash = RESERVED_KEYS & record.keys()
        if clash:
            raise ValueError(f"record uses reserved keys: {sorted(clash)}")
        with self._lock:
            if self._fh.closed:
                raise ValueError("writer is closed")
            entry = {"seq": self._seq, **record, "prev_hash": self._prev}
            entry["hash"] = record_hash(entry)
            data = (canonical(entry) + "\n").encode("utf-8")
            self._fh.write(data)
            self._fh.flush()
            if self._fsync:
                os.fsync(self._fh.fileno())
            self._seq += 1
            self._prev = entry["hash"]
            self._bytes += len(data)
            return entry

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> "HashChainWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
