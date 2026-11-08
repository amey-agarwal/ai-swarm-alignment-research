import pytest

from swarmguard.config import load_config
from swarmguard.orchestrator.bus import Bus, RoutingError
from swarmguard.orchestrator.topology import IMPOSTOR_ID, LEADER_ID, Topology, build_allowlist, resolve_contact
from swarmguard.safety.preflight import check_routing, reference_allowlist
from swarmguard.safety.probes import probe_bus


@pytest.mark.parametrize("n", [2, 3, 5, 8, 16, 32])
def test_allowlist_matches_independent_reference(n):
    assert build_allowlist(n) == reference_allowlist(n)
    assert build_allowlist(n, impostor_contact="leader") == reference_allowlist(n, impostor_contact="leader")
    for k in (1, n - 1):
        c = f"worker:{k}"
        assert build_allowlist(n, impostor_contact=c) == reference_allowlist(n, impostor_contact=c)


def test_star_has_no_worker_to_worker_edges():
    e = build_allowlist(8)
    assert all(LEADER_ID in pair for pair in e) and len(e) == 14


def test_tree_not_implemented_and_bad_contact():
    with pytest.raises(NotImplementedError):
        build_allowlist(8, "tree")
    for bad in ("worker:0", "worker:8", "boss"):
        with pytest.raises(ValueError):
            resolve_contact(8, bad)


def test_topology_names_and_contacts():
    t = Topology.star(4)
    assert t.names == {"a00": "Leader", "a01": "W1", "a02": "W2", "a03": "W3"}
    assert t.contacts("a00") == ["a01", "a02", "a03"] and t.contacts("a02") == ["a00"]
    t2 = Topology.star(4, impostor_contact="worker:2")
    assert t2.contacts("a02") == ["a00", IMPOSTOR_ID] and t2.contacts(IMPOSTOR_ID) == ["a02"]
    assert t2.names[IMPOSTOR_ID] == "W4"


def test_bus_routes_allowed_and_rejects_disallowed():
    bus = Bus(build_allowlist(4))
    bus.send("a01", "a00", "clue")
    bus.send("a00", "a02", "ask")
    for s, r in [("a01", "a02"), ("a01", "a01"), ("x00", "a00"), ("a00", "x00"), ("a09", "a00")]:
        with pytest.raises(RoutingError):
            bus.send(s, r, "nope")
    out = bus.flush()
    assert [(e.sender, e.recipient) for e in out] == [("a01", "a00"), ("a00", "a02")]
    assert bus.flush() == [] and bus.rejected == 5 and bus.delivered == 2


def test_bus_rejects_non_text():
    with pytest.raises(TypeError):
        Bus(build_allowlist(3)).send("a01", "a00", {"cmd": "x"})


@pytest.mark.parametrize("mode", ["dev", "cluster"])
def test_safety_probe_passes_with_real_bus(mode):
    r = probe_bus(mode)
    assert r.status == "pass", r.detail


def test_preflight_routing_check_now_passes():
    r = check_routing(load_config())
    assert r.status == "pass", r.detail
