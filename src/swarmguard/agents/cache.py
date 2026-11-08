"""SQLite response cache (CLAUDE.md §4.7, §10).

One file, `cache.sqlite`, inside the run directory (addressed via RunDir, so it
cannot be placed elsewhere). All queries are parameterised. The journal is kept
in memory so SQLite never creates extra files (-journal, -wal, -shm) that would
violate the run-dir allowlist. Inserts beyond `max_bytes` raise CacheFullError
(fail closed; the episode stops).
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from swarmguard.rundir import RunDir

CACHE_FILE = "cache.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    text TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    finish_reason TEXT NOT NULL
)
"""


class CacheFullError(RuntimeError):
    pass


@dataclass(frozen=True)
class CachedResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


class ResponseCache:
    def __init__(self, rd: RunDir, max_bytes: int, *, read_only: bool = False):
        self.path: Path = rd.file(CACHE_FILE)
        self.max_bytes = max_bytes
        self.read_only = read_only
        if read_only:
            if not self.path.exists():
                raise FileNotFoundError(f"no cache at {self.path}")
            uri = f"file:{self.path}?mode=ro"
            self._db = sqlite3.connect(uri, uri=True, check_same_thread=False)
        else:
            self._db = sqlite3.connect(str(self.path), check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=MEMORY")
            self._db.execute(_SCHEMA)
            self._db.commit()
        self._db.execute("PRAGMA temp_store=MEMORY")
        self._lock = threading.Lock()

    def size_bytes(self) -> int:
        with self._lock:
            pages = self._db.execute("PRAGMA page_count").fetchone()[0]
            size = self._db.execute("PRAGMA page_size").fetchone()[0]
        return int(pages) * int(size)

    def get(self, key: str) -> CachedResponse | None:
        with self._lock:
            row = self._db.execute(
                "SELECT text, prompt_tokens, completion_tokens, finish_reason FROM responses WHERE key = ?",
                (key,),
            ).fetchone()
        return CachedResponse(*row) if row else None

    def put(self, key: str, model_id: str, resp: CachedResponse) -> None:
        if self.read_only:
            raise PermissionError("cache opened read-only")
        incoming = len(resp.text.encode("utf-8")) + len(key) + len(model_id) + 64
        if self.size_bytes() + incoming > self.max_bytes:
            raise CacheFullError(f"response cache would exceed {self.max_bytes} bytes")
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO responses (key, model_id, text, prompt_tokens, completion_tokens, "
                "finish_reason) VALUES (?, ?, ?, ?, ?, ?)",
                (key, model_id, resp.text, int(resp.prompt_tokens), int(resp.completion_tokens), resp.finish_reason),
            )
            self._db.commit()

    def __len__(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM responses").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> "ResponseCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
