import os
import stat

import pytest

from swarmguard.config import load_config
from swarmguard.safety import probes
from swarmguard.safety.governor import FakeMetricsProvider


JOB = {"SLURM_JOB_ID": "1"}


@pytest.fixture
def cfg():
    return load_config()


def _boom(*a, **k):
    raise OSError("network unreachable")


class _Conn:
    def close(self):
        pass


def test_dev_pass2_ok_without_network(cfg, monkeypatch):
    # In dev, network probes must not even be attempted.
    monkeypatch.setattr(probes.socket, "create_connection", lambda *a, **k: pytest.fail("network attempted"))
    monkeypatch.setattr(probes.socket, "getaddrinfo", lambda *a, **k: pytest.fail("dns attempted"))
    res = probes.run_pass2(cfg, "dev")
    assert res.ok, [(c.name, c.status, c.detail) for c in res.checks]
    assert res.get("net_tcp_external").status == "skipped"
    assert res.get("governor_heartbeat").status == "pass"
    assert res.get("stop_file_shutdown").status == "pass"


def test_tcp_blocked_passes(cfg):
    assert probes.probe_tcp(cfg, "cluster", connect=_boom, env=JOB).status == "pass"


def test_tcp_connected_fails(cfg):
    assert probes.probe_tcp(cfg, "cluster", connect=lambda *a, **k: _Conn(), env=JOB).status == "fail"


def test_dns_resolved_fails_and_blocked_passes(cfg):
    assert probes.probe_dns(cfg, "cluster", resolve=lambda *a, **k: [("x",)], env=JOB).status == "fail"
    assert probes.probe_dns(cfg, "cluster", resolve=_boom, env=JOB).status == "pass"


def test_network_waived_only_with_explicit_setting():
    cfg = load_config(overrides={"sandbox": {
        "network_isolation": "unavailable",
        "network_isolation_justification": "Test platform cannot isolate network namespaces.",
    }})
    r = probes.probe_tcp(cfg, "cluster", connect=lambda *a, **k: _Conn(), env=JOB)
    assert r.status == "waived" and "WAIVED" in r.detail


def test_write_outside_writable_target_fails_and_cleans_up(tmp_path):
    r = probes.probe_write_outside("cluster", targets=[tmp_path], env=JOB)
    assert r.status == "fail"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
def test_write_outside_readonly_target_passes(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        assert probes.probe_write_outside("cluster", targets=[ro], env=JOB).status == "pass"
    finally:
        ro.chmod(stat.S_IRWXU)


def test_cluster_missing_modules_fail():
    # parser/bus are built in Steps 5/6; until then cluster mode must fail closed.
    for r in (probes.probe_parser("cluster"), probes.probe_bus("cluster")):
        assert r.status in ("fail", "pass")
        if "not available" in r.detail:
            assert r.status == "fail"


class _ParseError(Exception):
    pass


def _strict_parse(raw, *, allowed_recipients, max_tokens):
    import json
    if len(raw) > 10_000:
        raise _ParseError("oversized")
    try:
        d = json.loads(raw)
    except json.JSONDecodeError as e:
        raise _ParseError(str(e))
    if not isinstance(d, dict) or set(d) != {"to", "content"} or not isinstance(d["content"], str):
        raise _ParseError("schema")
    if d["to"] not in allowed_recipients:
        raise _ParseError("recipient")
    return d


def test_parser_probe_with_strict_parser():
    assert probes.probe_parser("cluster", parse=_strict_parse, error_cls=_ParseError).status == "pass"


def test_parser_probe_with_lenient_parser_fails():
    r = probes.probe_parser("cluster", parse=lambda raw, **k: raw, error_cls=_ParseError)
    assert r.status == "fail" and "ACCEPTED" in r.detail


def test_parser_probe_with_reject_all_parser_fails():
    def reject(raw, **k):
        raise _ParseError("no")
    assert probes.probe_parser("cluster", parse=reject, error_cls=_ParseError).status == "fail"


class _RoutingError(Exception):
    pass


class _StrictBus:
    def __init__(self, allowlist):
        self.allow = set(allowlist)

    def send(self, sender, recipient, content):
        if (sender, recipient) not in self.allow:
            raise _RoutingError(f"{sender}->{recipient}")


class _OpenBus(_StrictBus):
    def send(self, sender, recipient, content):
        pass


def test_bus_probe():
    assert probes.probe_bus("cluster", _StrictBus, _RoutingError).status == "pass"
    r = probes.probe_bus("cluster", _OpenBus, _RoutingError)
    assert r.status == "fail" and "ACCEPTED" in r.detail


def test_governor_probe_fails_without_nvml_in_cluster(cfg):
    try:
        import pynvml  # noqa: F401
        pytest.skip("pynvml installed")
    except ImportError:
        pass
    assert probes.probe_governor(cfg, "cluster").status == "fail"


def test_governor_probe_with_fake(cfg):
    assert probes.probe_governor(cfg, "cluster", provider=FakeMetricsProvider()).status == "pass"


def test_stop_file_probe():
    assert probes.probe_stop_file().status == "pass"


def test_cluster_probes_not_attempted_outside_slurm_job(cfg, tmp_path):
    def never(*a, **k):
        pytest.fail("probe attempted outside a Slurm job")
    for r in (probes.probe_tcp(cfg, "cluster", connect=never, env={}),
              probes.probe_dns(cfg, "cluster", resolve=never, env={}),
              probes.probe_write_outside("cluster", targets=[tmp_path], env={})):
        assert r.status == "fail" and "not attempted" in r.detail
    assert list(tmp_path.iterdir()) == []
