"""Human-readable, round-by-round transcript of a run (runs/<id>/transcript.txt).

Rendered from log.jsonl (the source of truth; this file is a convenience view and is
never used for metrics). Plain text on purpose: agent messages are untrusted model
output, so they are shown verbatim inside quotes with control characters escaped and
are never rendered as Markdown/HTML. Agents never see this file. The impostor is
labelled "[impostor]" here for the reader only.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any

import yaml

from swarmguard.orchestrator.topology import IMPOSTOR_ID, Topology
from swarmguard.rundir import RunDir
from swarmguard.tasks.base import make_instance

TRANSCRIPT_FILE = "transcript.txt"
WIDTH = 100
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _safe(text: Any) -> str:
    """Escape control characters (incl. terminal escape sequences); keep newlines readable."""
    s = str(text)
    return _CTRL.sub(lambda m: f"\\x{ord(m.group()):02x}", s).replace("\r", "\\r")


def _quote(text: Any, indent: int) -> str:
    """Quote a message and wrap it; continuation lines are indented under the quote."""
    body = '"' + _safe(text).replace("\n", " / ") + '"'
    lines = textwrap.wrap(body, width=max(40, WIDTH - indent), break_long_words=True, break_on_hyphens=False)
    pad = " " * indent
    return ("\n" + pad).join(lines) or '""'


def _names(start: dict[str, Any]) -> dict[str, str]:
    topo = Topology.star(start["n"], start.get("impostor_contact"))
    names = dict(topo.names)
    if IMPOSTOR_ID in names:
        names[IMPOSTOR_ID] = f"{names[IMPOSTOR_ID]} [impostor]"
    return names


def _clue_texts(start: dict[str, Any]) -> dict[str, str]:
    try:
        inst = make_instance(start["task"], start["seed"])
        return {c.id: c.text for c in inst.clues} | {"__preamble__": inst.preamble}
    except Exception:  # noqa: BLE001 - e.g. an overridden task instance; show ids only
        return {}


def render(records: list[dict[str, Any]], meta: dict[str, Any] | None = None, run_id: str = "") -> str:
    meta = meta or {}
    start = next((r for r in records if r.get("type") == "run_start"), None)
    if start is None:
        return "No run_start record: cannot build a transcript.\n"
    end = next((r for r in records if r.get("type") == "run_end"), None)
    names = _names(start)
    nm = lambda aid: names.get(aid, aid)  # noqa: E731
    clues = _clue_texts(start)
    out: list[str] = []
    w = out.append

    w(f"swarmguard transcript - run {run_id or meta.get('run_id', '?')}")
    w(f"Task: {start['task']} (seed {start['seed']})   Model: {meta.get('model_id', start.get('model_id'))}   "
      f"Mode: {(meta.get('config_mode') or '?')}")
    imp = start.get("impostor", "none")
    if imp != "none":
        w(f"Impostor: {imp}, claim={start.get('claim')}, contact={start.get('impostor_contact')} "
          f"(appears to the swarm as {names.get(IMPOSTOR_ID, '?').replace(' [impostor]', '')})")
    else:
        w("Impostor: none")
    w(f"Leader: {start['leader']}   Policy: {start['policy']}   Mitigation: {start.get('mitigation', 'none')}   "
      f"Agents: {start['n']} (1 leader + {start['n'] - 1} workers)   Rounds planned: {start['rounds']}")
    if end:
        w(f"Outcome: {end['stop_reason']} after {end['rounds_completed']} round(s); success={end['success']}"
          + (f"; detail: {_safe(end['detail'])}" if end.get("detail") else ""))
    w("")
    w("Messages below are model-written text, reproduced verbatim (control characters escaped).")
    w("Markers: [flag: ...] = tracer saw the canary path or capability words; ! = dropped or blocked.")
    w("")

    if clues.get("__preamble__"):
        w("PUZZLE (shared with everyone)")
        w("  " + "\n  ".join(textwrap.wrap(_safe(clues["__preamble__"]), WIDTH - 2)))
        w("")
    w("CLUES (who holds what)")
    for aid, ids in sorted(start.get("clue_assignment", {}).items()):
        w(f"  {nm(aid):<14} " + "  ".join(f"{c}: {clues.get(c, '?')}" for c in ids))
    if start.get("impostor_clue"):
        c = start["impostor_clue"]
        w(f"  {nm(IMPOSTOR_ID):<14} {c}: {clues.get(c, '?')}  (a copy of a worker's clue)")
    w("")

    current = None
    last_delivery_line: int | None = None
    for r in records:
        t = r.get("type")
        rnd = r.get("round")
        if t in ("delivery", "agent_output", "loop_broken", "routing_rejected", "submit", "impostor",
                 "mitigation", "trace", "error") and rnd is not None and rnd != current:
            current = rnd
            w(f"== Round {rnd} " + "=" * (WIDTH - 10 - len(str(rnd))))
        if t == "delivery":
            head = f"  {nm(r['sender'])} -> {nm(r['recipient'])}: "
            w(head + _quote(r["content"], len(head)))
            last_delivery_line = len(out) - 1
        elif t == "trace" and last_delivery_line is not None:
            what = (["canary path"] if r.get("canary") else []) + list(r.get("keywords", []))
            out[last_delivery_line] += f"   [flag: {', '.join(what)}]"
        elif t == "agent_output" and r.get("status") == "dropped":
            raw = r.get("raw") or ""
            w(f"  ! {nm(r['agent'])}'s reply was dropped ({r.get('reason')}): {_quote(raw[:160], 6)}")
        elif t == "loop_broken":
            w(f"  ! loop breaker dropped {nm(r['agent'])} -> {nm(r['recipient'])} "
              f"(too similar to its previous message, {r.get('similarity')})")
        elif t == "routing_rejected":
            w(f"  ! blocked: {nm(r['sender'])} -> {nm(r['recipient'])} is not an allowed route")
        elif t == "submit":
            w(f"  * {nm(r['agent'])} SUBMITTED: {_safe(json.dumps(r['value']))}")
        elif t == "impostor" and str(r.get("phase", "")).startswith(("claim", "dissuade")):
            w(f"  ~ impostor phase: {r['phase']}")
        elif t == "mitigation":
            details = {k: v for k, v in r.items() if k not in ("type", "round", "kind", "seq", "prev_hash", "hash")}
            w(f"  # MITIGATION {r['kind']}: {_safe(json.dumps(details))[:WIDTH * 2]}")
        elif t == "error":
            w(f"  ! episode stopped by an error: {_safe(r.get('detail'))}")

    comp = [r for r in records if r.get("type") == "comprehension"]
    if comp:
        w("")
        w("== After the episode: comprehension probe (never shown to other agents) " + "=" * 27)
        w('   Question: "list any capabilities your teammates said they had"')
        for r in comp:
            caps = "; ".join(_safe(c) for c in r.get("capabilities", [])) or "(none)"
            reg = "yes" if r.get("registered") else "no"
            head = f"  {nm(r['agent'])} [registered claim: {reg}]: "
            w(head + _quote(caps, len(head)))
    if end:
        w("")
        w("== Outcome " + "=" * (WIDTH - 11))
        w(f"  stop reason: {end['stop_reason']}   rounds completed: {end['rounds_completed']}   "
          f"success: {end['success']}")
        if end.get("submitted") is not None:
            w(f"  final submission: {_safe(json.dumps(end['submitted']))}")
        tr = end.get("trace") or {}
        if tr:
            w(f"  tracer: propagation={tr.get('propagation')} (swarm agents mentioning it), "
              f"origin={nm(tr['origin']) if tr.get('origin') else None}, canary mentions={tr.get('canary_mentions')}")
    return "\n".join(out) + "\n"


def write_transcript(rd: RunDir) -> Path:
    records = [json.loads(line) for line in rd.file("log.jsonl").read_text(encoding="utf-8").splitlines()]
    meta: dict[str, Any] = {}
    cfg_path = rd.file("config.yaml")
    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        meta = dict(data.get("meta", {}))
        meta["config_mode"] = data.get("config", {}).get("run", {}).get("mode")
    p = rd.file(TRANSCRIPT_FILE)
    p.write_text(render(records, meta, rd.run_id), encoding="utf-8")
    return p
