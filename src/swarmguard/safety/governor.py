"""Resource governor (CLAUDE.md §4.5).

CPI = max(power / power_cap, gpu_mem / gpu_mem_cap, token_rate / token_rate_cap,
          msg_rate / msg_rate_cap, rundir_bytes / rundir_cap)
  CPI >= soft            -> throttle (lower concurrency, pause between rounds)
  CPI >= hard, or any budget exhausted, disk low, run-dir violation -> stop (latched)

dev mode uses FakeMetricsProvider. cluster mode requires NVML (pynvml) and psutil;
if either is unavailable the governor refuses to start (fail closed).
Throttling concurrency is the control lever; GPU power limits are never set.
"""

from __future__ import annotations

import shutil
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Literal, Protocol

from swarmguard.config import Config
from swarmguard.rundir import RunDir
from swarmguard.safety.integrity import check_rundir

Action = Literal["ok", "throttle", "stop"]


class GovernorError(Exception):
    pass


@dataclass(frozen=True)
class HardwareSample:
    gpu_util: float | None = None  # 0..1
    gpu_mem_frac: float | None = None  # 0..1
    power_w: float | None = None
    power_rated_w: float | None = None
    temp_c: float | None = None
    cpu_pct: float | None = None
    rss_bytes: int | None = None
    disk_free_frac: float | None = None


class MetricsProvider(Protocol):
    name: str

    def sample(self) -> HardwareSample: ...


class FakeMetricsProvider:
    """Deterministic, settable metrics for dev mode and tests."""

    name = "fake"

    def __init__(self, **values):
        defaults = dict(gpu_util=0.0, gpu_mem_frac=0.1, power_w=50.0, power_rated_w=400.0,
                        temp_c=40.0, cpu_pct=5.0, rss_bytes=100 * 1024**2, disk_free_frac=0.5)
        defaults.update(values)
        self._s = HardwareSample(**defaults)
        self._lock = threading.Lock()

    def set(self, **values) -> None:
        with self._lock:
            self._s = replace(self._s, **values)

    def sample(self) -> HardwareSample:
        with self._lock:
            return self._s


class NvmlMetricsProvider:
    """Real metrics via pynvml + psutil. Raises GovernorError if unavailable."""

    name = "nvml"

    def __init__(self, disk_path: str | Path = ".", device_index: int = 0):
        try:
            import psutil  # noqa: F401
            import pynvml
        except ImportError as exc:
            raise GovernorError(f"NVML/psutil unavailable: {exc}") from exc
        try:
            pynvml.nvmlInit()
            self._h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as exc:  # noqa: BLE001
            raise GovernorError(f"NVML init failed: {exc}") from exc
        self._nvml = pynvml
        self._psutil = psutil
        self._proc = psutil.Process()
        self._disk = str(disk_path)

    def sample(self) -> HardwareSample:
        n, h = self._nvml, self._h
        mem = n.nvmlDeviceGetMemoryInfo(h)
        du = shutil.disk_usage(self._disk)
        return HardwareSample(
            gpu_util=n.nvmlDeviceGetUtilizationRates(h).gpu / 100.0,
            gpu_mem_frac=mem.used / mem.total,
            power_w=n.nvmlDeviceGetPowerUsage(h) / 1000.0,
            power_rated_w=n.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0,
            temp_c=float(n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)),
            cpu_pct=self._psutil.cpu_percent(interval=None),
            rss_bytes=self._proc.memory_info().rss,
            disk_free_frac=du.free / du.total,
        )


def make_provider(mode: str, disk_path: str | Path = ".") -> MetricsProvider:
    if mode == "dev":
        return FakeMetricsProvider()
    if mode == "cluster":
        return NvmlMetricsProvider(disk_path)
    raise GovernorError(f"unknown mode {mode!r}")


@dataclass
class GovernorState:
    action: Action = "ok"
    reason: str | None = None
    cpi: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    peak_cpi: float = 0.0
    energy_wh: float = 0.0
    tokens_total: int = 0
    messages_total: int = 0
    heartbeats: int = 0
    last_tick: float | None = None
    last_sample: HardwareSample | None = None
    throttle_events: int = 0


class Governor:
    def __init__(
        self,
        cfg: Config,
        provider: MetricsProvider,
        *,
        rundir: RunDir | None = None,
        rundir_bytes: Callable[[], int] | None = None,
        clock: Callable[[], float] = time.monotonic,
        interval_s: float | None = None,
        max_concurrency: int = 8,
    ):
        self.cfg = cfg
        self.provider = provider
        self.rundir = rundir
        self._rundir_bytes = rundir_bytes
        self._clock = clock
        self.interval_s = interval_s if interval_s is not None else cfg.governor.interval_s
        self.max_concurrency = max(1, max_concurrency)
        self._lock = threading.RLock()
        self._state = GovernorState()
        self._pending_tokens = 0
        self._pending_msgs = 0
        self._last_power: float | None = None
        self._last_rundir_check: float | None = None
        self._thread: threading.Thread | None = None
        self._halt = threading.Event()

    # -- counters fed by the orchestrator -------------------------------------------------
    def record_tokens(self, n: int) -> None:
        with self._lock:
            self._pending_tokens += int(n)
            self._state.tokens_total += int(n)
            if self._state.tokens_total >= self.cfg.budget.tokens:
                self._latch_stop(f"budget:tokens ({self._state.tokens_total} >= {self.cfg.budget.tokens})")

    def record_messages(self, n: int) -> None:
        with self._lock:
            self._pending_msgs += int(n)
            self._state.messages_total += int(n)

    # -- evaluation -----------------------------------------------------------------------
    def _latch_stop(self, reason: str) -> None:
        if self._state.action != "stop":
            self._state.action = "stop"
            self._state.reason = reason

    def tick(self) -> GovernorState:
        """Take one sample and update the decision. Called by the thread or directly in tests."""
        g = self.cfg.governor
        now = self._clock()
        try:
            s = self.provider.sample()
        except Exception as exc:  # noqa: BLE001 - metrics failure fails closed
            with self._lock:
                self._latch_stop(f"governor:metrics_error ({exc!r})")
                self._state.heartbeats += 1
                self._state.last_tick = now
                return self.state()
        with self._lock:
            st = self._state
            dt = 0.0 if st.last_tick is None else max(0.0, now - st.last_tick)
            token_rate = self._pending_tokens / dt if dt > 0 else 0.0
            msg_rate = self._pending_msgs / dt if dt > 0 else 0.0
            self._pending_tokens = self._pending_msgs = 0

            if s.power_w is not None:
                if self._last_power is not None and dt > 0:
                    st.energy_wh += (self._last_power + s.power_w) / 2.0 * dt / 3600.0
                self._last_power = s.power_w

            comps: dict[str, float] = {}
            if s.power_w is not None and s.power_rated_w:
                comps["power"] = s.power_w / (s.power_rated_w * g.power_frac_cap)
            if s.gpu_mem_frac is not None:
                comps["gpu_mem"] = s.gpu_mem_frac / g.gpu_mem_frac_cap
            comps["token_rate"] = token_rate / g.token_rate_cap
            comps["msg_rate"] = msg_rate / g.msg_rate_cap
            if self._rundir_bytes is not None:
                comps["rundir_bytes"] = self._rundir_bytes() / self.cfg.rundir.max_bytes
            cpi = max(comps.values()) if comps else 0.0

            st.components, st.cpi = comps, cpi
            st.peak_cpi = max(st.peak_cpi, cpi)
            st.last_sample = s
            st.last_tick = now
            st.heartbeats += 1

            if cpi >= g.hard:
                top = max(comps, key=comps.get)
                self._latch_stop(f"cpi:hard ({cpi:.3f} >= {g.hard}, {top})")
            if st.energy_wh >= self.cfg.budget.energy_wh:
                self._latch_stop(f"budget:energy ({st.energy_wh:.3f} Wh >= {self.cfg.budget.energy_wh})")
            if st.tokens_total >= self.cfg.budget.tokens:
                self._latch_stop(f"budget:tokens ({st.tokens_total} >= {self.cfg.budget.tokens})")
            if s.disk_free_frac is not None and s.disk_free_frac < self.cfg.disk.min_free_frac:
                self._latch_stop(f"disk:free ({s.disk_free_frac:.3f} < {self.cfg.disk.min_free_frac})")
            if self.rundir is not None and (
                self._last_rundir_check is None
                or now - self._last_rundir_check >= self.cfg.rundir.check_interval_s
            ):
                self._last_rundir_check = now
                rc = check_rundir(self.rundir, self.cfg.rundir)
                if not rc.ok:
                    self._latch_stop(f"rundir: {'; '.join(rc.reasons)}")

            if st.action != "stop":
                new = "throttle" if cpi >= g.soft else "ok"
                if new == "throttle" and st.action != "throttle":
                    st.throttle_events += 1
                st.action = new
                st.reason = f"cpi:soft ({cpi:.3f} >= {g.soft})" if new == "throttle" else None
            return self.state()

    # -- queries --------------------------------------------------------------------------
    def state(self) -> GovernorState:
        with self._lock:
            return replace(self._state, components=dict(self._state.components))

    def should_stop(self) -> str | None:
        with self._lock:
            return self._state.reason if self._state.action == "stop" else None

    def advice(self) -> tuple[int, float]:
        """(max concurrent requests, pause seconds between rounds)."""
        with self._lock:
            a = self._state.action
        if a == "stop":
            return 0, 0.0
        if a == "throttle":
            return max(1, self.max_concurrency // 2), float(self.interval_s)
        return self.max_concurrency, 0.0

    def summary(self) -> dict:
        st = self.state()
        return {
            "peak_cpi": st.peak_cpi,
            "energy_wh": st.energy_wh,
            "tokens": st.tokens_total,
            "messages": st.messages_total,
            "stop_reason": st.reason if st.action == "stop" else None,
            "throttle_events": st.throttle_events,
            "heartbeats": st.heartbeats,
            "provider": self.provider.name,
        }

    # -- background thread ----------------------------------------------------------------
    def _run(self) -> None:
        while not self._halt.is_set():
            self.tick()
            self._halt.wait(self.interval_s)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._halt.clear()
        self._thread = threading.Thread(target=self._run, name="swarmguard-governor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._halt.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, 2 * self.interval_s))
            self._thread = None

    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self) -> "Governor":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
