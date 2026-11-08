import time

import pytest

from swarmguard.config import load_config
from swarmguard.rundir import RunDir
from swarmguard.safety.governor import FakeMetricsProvider, Governor, GovernorError, make_provider
from swarmguard.safety.killswitch import KillSwitch, read_checkpoint, request_stop, write_checkpoint


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _gov(overrides=None, **prov):
    cfg = load_config(overrides=overrides)
    clock = Clock()
    p = FakeMetricsProvider(**prov)
    return Governor(cfg, p, clock=clock), p, clock


def test_low_pressure_ok():
    g, _, _ = _gov()
    st = g.tick()
    assert st.action == "ok" and st.cpi < 0.8 and g.advice() == (8, 0.0)


def test_crossing_soft_throttles_then_recovers():
    g, p, c = _gov(gpu_mem_frac=0.9 * 0.85)  # cpi = 0.85
    st = g.tick()
    assert st.action == "throttle" and st.cpi == pytest.approx(0.85)
    conc, pause = g.advice()
    assert conc < 8 and pause > 0
    p.set(gpu_mem_frac=0.1)
    c.t += 2
    assert g.tick().action == "ok"
    assert g.summary()["throttle_events"] == 1


def test_crossing_hard_stops_and_latches():
    g, p, c = _gov(gpu_mem_frac=0.9 * 0.96)  # cpi = 0.96
    st = g.tick()
    assert st.action == "stop" and "cpi:hard" in st.reason
    p.set(gpu_mem_frac=0.1)
    c.t += 2
    assert g.tick().action == "stop"
    assert g.should_stop() and g.advice()[0] == 0


def test_power_component():
    g, _, _ = _gov(power_w=400 * 0.85 * 0.82, power_rated_w=400)
    st = g.tick()
    assert st.components["power"] == pytest.approx(0.82) and st.action == "throttle"


def test_token_rate_component():
    g, _, c = _gov()
    g.tick()
    c.t += 1.0
    g.record_tokens(int(20000 * 0.97))
    st = g.tick()
    assert st.components["token_rate"] == pytest.approx(0.97) and st.action == "stop"


def test_rundir_bytes_component():
    cfg = load_config()
    g = Governor(cfg, FakeMetricsProvider(), rundir_bytes=lambda: int(cfg.rundir.max_bytes * 0.99), clock=Clock())
    assert g.tick().action == "stop"


def test_token_budget_exhausted():
    g, _, _ = _gov(overrides={"budget": {"tokens": 100}})
    g.record_tokens(100)
    assert g.should_stop() and "budget:tokens" in g.should_stop()


def test_energy_integration_and_budget():
    g, _, c = _gov(overrides={"budget": {"energy_wh": 0.5}}, power_w=100, power_rated_w=1000)
    g.tick()
    c.t += 18.0  # 100 W * 18 s = 0.5 Wh
    st = g.tick()
    assert st.energy_wh == pytest.approx(0.5)
    assert st.action == "stop" and "budget:energy" in st.reason


def test_low_disk_stops():
    g, _, _ = _gov(disk_free_frac=0.05)
    assert "disk" in g.tick().reason


def test_metrics_error_fails_closed():
    class Broken:
        name = "broken"

        def sample(self):
            raise RuntimeError("nvml lost")

    g = Governor(load_config(), Broken(), clock=Clock())
    assert g.tick().action == "stop"


def test_rundir_violation_stops(tmp_path):
    rd = RunDir.create(tmp_path)
    (rd.path / "payload.sh").write_text("x")
    g = Governor(load_config(), FakeMetricsProvider(), rundir=rd, clock=Clock())
    st = g.tick()
    assert st.action == "stop" and "rundir" in st.reason


def test_cluster_provider_requires_nvml():
    try:
        import pynvml  # noqa: F401
        pytest.skip("pynvml installed")
    except ImportError:
        pass
    with pytest.raises(GovernorError):
        make_provider("cluster")
    assert make_provider("dev").name == "fake"


def test_background_thread_heartbeat():
    g = Governor(load_config(), FakeMetricsProvider(), interval_s=0.05)
    with g:
        time.sleep(0.3)
        assert g.alive() and g.state().heartbeats >= 2
    assert not g.alive()


# --- kill switch ----------------------------------------------------------------


def test_killswitch_stop_file_and_governor(tmp_path):
    rd = RunDir.create(tmp_path)
    ks = KillSwitch(rd)
    g, p, _ = _gov()
    assert ks.check(g) is None
    p.set(gpu_mem_frac=0.89)
    g.tick()
    assert ks.check(g).startswith("governor:cpi:hard")
    request_stop(rd)
    assert ks.check(g) == "killswitch:stop_file"


def test_killswitch_signal(tmp_path):
    import os
    import signal

    rd = RunDir.create(tmp_path)
    ks = KillSwitch(rd)
    assert ks.install_signal_handlers()
    try:
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(0.05)
        assert ks.check() == "signal:SIGUSR1"
    finally:
        ks.uninstall_signal_handlers()


def test_checkpoint_roundtrip(tmp_path):
    rd = RunDir.create(tmp_path)
    assert read_checkpoint(rd) is None
    write_checkpoint(rd, {"round": 3, "reason": "signal:SIGUSR1"})
    assert read_checkpoint(rd) == {"round": 3, "reason": "signal:SIGUSR1"}
    assert rd.unexpected() == []
