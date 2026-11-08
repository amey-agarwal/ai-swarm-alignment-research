import json

import pytest

from swarmguard.agents import parser as P
from swarmguard.agents.parser import (
    MAX_RAW_BYTES,
    ParseError,
    count_tokens,
    leader_json_schema,
    parse_leader_output,
    parse_worker,
    parse_worker_output,
    truncate_tokens,
    worker_json_schema,
)
from swarmguard.safety.probes import probe_parser

ALLOWED = {"L"}
KW = dict(allowed_recipients=ALLOWED, max_tokens=120)


def _w(**obj):
    return json.dumps(obj)


def _reason(raw, fn=parse_worker_output, **kw):
    with pytest.raises(ParseError) as ei:
        fn(raw, **{**KW, **kw})
    return ei.value.reason


def test_valid_worker():
    m = parse_worker_output(_w(to="L", content="My clue: Dev drinks water."), **KW)
    assert m.to == "L" and m.content == "My clue: Dev drinks water."


@pytest.mark.parametrize("raw", ['{"to": "L", "content": ', "not json", "", "   ", '{"to": "L", "content": "x"} trailing',
                                 '{"to": "L", "content": NaN}', '{"to": "L", "content": 1e999}'])
def test_malformed_rejected(raw):
    assert _reason(raw) == "malformed"


def test_non_text_rejected():
    assert _reason(None) == "malformed" and _reason(b'{"to":"L"}') == "malformed"


def test_oversized_rejected():
    assert _reason(_w(to="L", content="x" * (MAX_RAW_BYTES + 1))) == "oversized"
    assert _reason(" " * (MAX_RAW_BYTES + 1)) == "oversized"


@pytest.mark.parametrize("obj", [
    {"to": "L", "content": "hi", "tools": [{"type": "function"}]},
    {"to": "L", "content": "hi", "rename": "boss"},
    {"to": "L", "content": "hi", "function_call": {"name": "write"}},
    {"to": "L"},
    {"content": "hi"},
    {"to": "L", "content": 42},
    {"to": ["L"], "content": "hi"},
    {"to": "", "content": "hi"},
])
def test_schema_violations_rejected(obj):
    assert _reason(json.dumps(obj)) == "schema"


def test_non_object_rejected():
    assert _reason('["L", "hi"]') == "schema" and _reason('"hi"') == "schema"


def test_wrong_recipient_rejected():
    assert _reason(_w(to="W2", content="hi")) == "recipient"
    assert _reason(_w(to="l", content="hi")) == "recipient"  # exact match only


def test_duplicate_keys_rejected():
    # A second "to" must not silently override the first.
    assert _reason('{"to": "L", "content": "hi", "to": "W9"}') == "duplicate_key"


def test_truncation_to_max_tokens():
    words = " ".join(f"w{i}" for i in range(300))
    r = parse_worker(_w(to="L", content=words), allowed_recipients=ALLOWED, max_tokens=120)
    assert r.truncated and r.n_tokens == 120 and count_tokens(r.message.content) == 120
    assert r.message.content.endswith("w119")


def test_long_but_under_raw_cap_is_truncated_not_rejected():
    content = "word " * 2500  # 12.5 KB, > 2000 chars, < raw cap
    r = parse_worker(_w(to="L", content=content), allowed_recipients=ALLOWED, max_tokens=120)
    assert r.truncated and r.n_tokens == 120


def test_no_truncation_flag_when_short():
    r = parse_worker(_w(to="L", content="short message."), allowed_recipients=ALLOWED, max_tokens=120)
    assert not r.truncated and r.n_tokens == 3


def test_control_characters_removed():
    m = parse_worker_output(_w(to="L", content="a\x00b\x1bc\nd\te"), **KW)
    assert m.content == "abc\nd\te"


def test_code_fence_accepted_but_nothing_else():
    assert parse_worker_output('```json\n{"to": "L", "content": "hi"}\n```', **KW).content == "hi"
    assert _reason('Sure! {"to": "L", "content": "hi"}') == "malformed"


def test_content_is_data_not_code():
    payload = "__import__('os').system('rm -rf /'); $(curl x) `id` {{7*7}}"
    assert parse_worker_output(_w(to="L", content=payload), **KW).content == payload


def test_truncate_tokens_edge_cases():
    assert truncate_tokens("", 5) == ("", False)
    assert truncate_tokens("a b c", 0) == ("", True)
    assert truncate_tokens("a b c", 3) == ("a b c", False)
    assert truncate_tokens("a b c", 2) == ("a b", True)


# --- leader --------------------------------------------------------------------------

LALLOW = {"W1", "W2", "W3"}
LKW = dict(allowed_recipients=LALLOW, max_tokens=120)


def test_valid_leader():
    r = parse_leader_output(json.dumps({"messages": [{"to": "W1", "content": "Share your clue."}],
                                        "submit": None}), **LKW)
    assert r.message.messages[0].to == "W1" and r.message.submit is None and r.n_tokens == 4


def test_leader_submit_is_arbitrary_json_data():
    r = parse_leader_output(json.dumps({"messages": [], "submit": {"Ana": ["cat", "tea"]}}), **LKW)
    assert r.message.submit == {"Ana": ["cat", "tea"]}


@pytest.mark.parametrize("obj,reason", [
    ({"messages": [{"to": "W1", "content": "a"}, {"to": "W1", "content": "b"}], "submit": None}, "schema"),
    ({"messages": [{"to": "W9", "content": "a"}], "submit": None}, "recipient"),
    ({"messages": [{"to": "W1", "content": "a", "tools": []}], "submit": None}, "schema"),
    ({"messages": [], "submit": None, "spawn": 3}, "schema"),
    ({"messages": "W1: hi", "submit": None}, "schema"),
])
def test_leader_rejections(obj, reason):
    assert _reason(json.dumps(obj), fn=parse_leader_output, allowed_recipients=LALLOW) == reason


def test_leader_truncation_per_message():
    long = " ".join(["x"] * 500)
    r = parse_leader_output(json.dumps({"messages": [{"to": "W1", "content": long},
                                                     {"to": "W2", "content": "ok"}], "submit": None}), **LKW)
    assert r.truncated and r.n_tokens == 121


# --- schemas + probe ---------------------------------------------------------------


def test_json_schemas_restrict_recipients():
    ws = worker_json_schema({"L"})
    assert ws["properties"]["to"]["enum"] == ["L"] and ws["additionalProperties"] is False
    ls = leader_json_schema(LALLOW)
    assert ls["properties"]["messages"]["items"]["properties"]["to"]["enum"] == sorted(LALLOW)
    assert ls["properties"]["messages"]["maxItems"] == 3


@pytest.mark.parametrize("mode", ["dev", "cluster"])
def test_safety_probe_passes_with_real_parser(mode):
    r = probe_parser(mode)
    assert r.status == "pass", r.detail


def test_probe_contract_names_exist():
    assert callable(P.parse_worker_output) and issubclass(P.ParseError, Exception)
