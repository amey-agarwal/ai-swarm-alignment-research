"""Kill switches (CLAUDE.md §4.9).

- STOP file in the run directory, checked every round -> graceful stop.
- SIGUSR1 (Slurm --signal=B:USR1@120) and SIGTERM set a flag; the episode loop
  checkpoints at the next round boundary and exits cleanly.
- The governor's hard threshold is surfaced through `check(governor=...)`.
"""

from __future__ import annotations

import json
import signal
import threading
from typing import Any, Protocol

from swarmguard.rundir import RunDir

STOP_FILE = "STOP"
CHECKPOINT_FILE = "checkpoint.json"
HANDLED_SIGNALS = (signal.SIGUSR1, signal.SIGTERM)


class _StopSource(Protocol):
    def should_stop(self) -> str | None: ...


class KillSwitch:
    def __init__(self, rd: RunDir):
        self.rd = rd
        self._signal: str | None = None
        self._previous: dict[int, Any] = {}

    def install_signal_handlers(self) -> bool:
        """Install handlers (main thread only). Returns False if not on the main thread."""
        if threading.current_thread() is not threading.main_thread():
            return False
        for sig in HANDLED_SIGNALS:
            self._previous[sig] = signal.signal(sig, self._on_signal)
        return True

    def uninstall_signal_handlers(self) -> None:
        for sig, prev in self._previous.items():
            signal.signal(sig, prev)
        self._previous.clear()

    def _on_signal(self, signum, _frame) -> None:
        self._signal = signal.Signals(signum).name

    def check(self, governor: _StopSource | None = None) -> str | None:
        """Return a stop reason, or None to continue. Call once per round."""
        if self.rd.file(STOP_FILE).exists():
            return "killswitch:stop_file"
        if self._signal is not None:
            return f"signal:{self._signal}"
        if governor is not None:
            r = governor.should_stop()
            if r:
                return f"governor:{r}"
        return None


def request_stop(rd: RunDir, reason: str = "user") -> None:
    """Create the STOP file (used by `swarm stop`)."""
    with rd.open(STOP_FILE, "w", encoding="utf-8") as fh:
        fh.write(f"stop: {reason}\n")


def write_checkpoint(rd: RunDir, state: dict[str, Any]) -> None:
    with rd.open(CHECKPOINT_FILE, "w", encoding="utf-8") as fh:
        json.dump(state, fh, sort_keys=True, indent=2)
        fh.write("\n")


def read_checkpoint(rd: RunDir) -> dict[str, Any] | None:
    p = rd.file(CHECKPOINT_FILE)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)
