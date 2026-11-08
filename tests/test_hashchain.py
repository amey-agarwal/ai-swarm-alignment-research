import json

import pytest

from swarmguard.audit.hashchain import GENESIS, HashChainWriter, canonical, record_hash, verify


def _write(path, n=5):
    with HashChainWriter(path) as w:
        for i in range(n):
            w.append({"type": "msg", "round": i, "from": "a01", "content": f"clue {i}"})
    return path


def _lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def _put(path, lines):
    path.write_text("".join(l + "\n" for l in lines), encoding="utf-8")


def test_valid_chain_verifies(tmp_path):
    p = _write(tmp_path / "log.jsonl")
    res = verify(p)
    assert res.ok and res.n_records == 5
    first = json.loads(_lines(p)[0])
    assert first["prev_hash"] == GENESIS and first["seq"] == 0


def test_empty_log_verifies(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_bytes(b"")
    assert verify(p).ok


def test_modified_content_detected(tmp_path):
    p = _write(tmp_path / "log.jsonl")
    lines = _lines(p)
    rec = json.loads(lines[2])
    rec["content"] = "tampered"
    lines[2] = canonical(rec)
    _put(p, lines)
    res = verify(p)
    assert not res.ok and res.line_no == 3 and "hash mismatch" in res.error


def test_forged_rehash_breaks_next_link(tmp_path):
    p = _write(tmp_path / "log.jsonl")
    lines = _lines(p)
    rec = json.loads(lines[2])
    rec["content"] = "tampered"
    rec.pop("hash")
    rec["hash"] = record_hash(rec)
    lines[2] = canonical(rec)
    _put(p, lines)
    res = verify(p)
    assert not res.ok and res.line_no == 4 and "prev_hash" in res.error


def test_deleted_line_detected(tmp_path):
    p = _write(tmp_path / "log.jsonl")
    lines = _lines(p)
    del lines[1]
    _put(p, lines)
    assert not verify(p).ok


def test_reordered_lines_detected(tmp_path):
    p = _write(tmp_path / "log.jsonl")
    lines = _lines(p)
    lines[1], lines[2] = lines[2], lines[1]
    _put(p, lines)
    assert not verify(p).ok


def test_truncation_detected(tmp_path):
    p = _write(tmp_path / "log.jsonl")
    p.write_bytes(p.read_bytes()[:-10])
    assert not verify(p).ok


def test_noncanonical_whitespace_detected(tmp_path):
    p = _write(tmp_path / "log.jsonl")
    lines = _lines(p)
    lines[0] = lines[0].replace(",", ", ", 1)
    _put(p, lines)
    assert not verify(p).ok


def test_resume_continues_chain(tmp_path):
    p = _write(tmp_path / "log.jsonl", n=3)
    with HashChainWriter(p) as w:
        assert w.n_records == 3
        w.append({"type": "msg", "round": 3})
    res = verify(p)
    assert res.ok and res.n_records == 4


def test_refuses_to_append_to_tampered_log(tmp_path):
    p = _write(tmp_path / "log.jsonl")
    lines = _lines(p)
    del lines[0]
    _put(p, lines)
    with pytest.raises(ValueError):
        HashChainWriter(p)


def test_byte_counter_matches_file_size(tmp_path):
    p = tmp_path / "log.jsonl"
    with HashChainWriter(p) as w:
        for i in range(10):
            w.append({"i": i, "text": "é" * i})
        assert w.bytes_written == p.stat().st_size
    with HashChainWriter(p) as w:
        assert w.bytes_written == p.stat().st_size


@pytest.mark.parametrize("key", ["seq", "prev_hash", "hash"])
def test_reserved_keys_rejected(tmp_path, key):
    with HashChainWriter(tmp_path / "log.jsonl") as w:
        with pytest.raises(ValueError):
            w.append({key: "x"})


def test_deterministic_output(tmp_path):
    a = _write(tmp_path / "a.jsonl")
    b = _write(tmp_path / "b.jsonl")
    assert a.read_bytes() == b.read_bytes()
