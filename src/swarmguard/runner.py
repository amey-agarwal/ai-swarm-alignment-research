"""Command implementations behind the CLI (CLAUDE.md §4.3, Step 10).

Inference commands (`run`, each `sweep` task, `judge`) always run safety pass 1 and
pass 2 before any model call and postflight afterwards; any failure gives a
non-zero exit code. Live impostor runs also need the CONFIRM gate. Non-inference
commands (`status`, `stop`, `replay` from cache) run pass 1.

There is no option, environment variable, or code path that skips a check.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import yaml

from swarmguard.agents.cache import ResponseCache
from swarmguard.agents.llm_client import LLMClient, LLMError, make_backend
from swarmguard.agents.stub_policies import route
from swarmguard.config import REPO_ROOT, Config, _deep_merge, _read_yaml, load_models
from swarmguard.impostor.live import confirm_gate
from swarmguard.mitigations import parse_mitigation
from swarmguard.orchestrator.episode import EpisodeResult, EpisodeSpec, run_episode
from swarmguard.rundir import RunDir, RunDirError
from swarmguard.safety.governor import Governor, GovernorError, make_provider
from swarmguard.safety.killswitch import KillSwitch, request_stop
from swarmguard.safety.postflight import run_postflight
from swarmguard.safety.preflight import PassResult, resolve_config, run_pass1
from swarmguard.safety.probes import run_pass2
from swarmguard.schemas import SafetyReport
from swarmguard.tasks.base import TASK_SPECS, get_task

DEFAULT_ROOT = REPO_ROOT  # repo checked by pass 1 / postflight (tests point this at a temp copy)
PORT_ENV = "SWARM_VLLM_PORT"
LOCAL_JUDGE_REFUSED = ("judge is not available in local mode: the judge must be a different model family "
                       "(gemma-3-27b) run on the cluster; local runs are exploratory and never analysable")
Out = Callable[[str], None]


@dataclass
class RunOptions:
    n: int = 8
    model: str = "llama-3.2-3b"
    rounds: int = 14
    topology: str = "star"
    task: str = "logic-grid:L4"
    impostor: str = "none"  # none | scripted | live | replay:<run_id>
    contact: str = "leader"
    claim: str = "write"
    policy: str = "none"
    leader: str = "llm"
    mitigation: str = "none"
    rename: str = "off"
    verifier: str = "passive"
    mode: str = "dev"
    backend: str = "stub"
    seed: int = 0
    out: Path = Path("runs")
    config: Path | None = None


@dataclass
class Outcome:
    exit_code: int
    message: str
    pass1: PassResult | None = None
    pass2: PassResult | None = None
    rd: RunDir | None = None
    result: EpisodeResult | None = None
    report: SafetyReport | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------- options


def validate_options(o: RunOptions) -> list[str]:
    errs = []
    if o.rename != "off":
        errs.append("--rename: only 'off' is available (the rename mechanic is P2)")
    try:
        ms = parse_mitigation(o.mitigation)
    except ValueError as exc:
        errs.append(str(exc))
        ms = None
    if o.verifier not in ("passive", "active"):
        errs.append("--verifier must be passive or active")
    elif ms is not None and ms.kind != "none" and o.verifier != "active":
        errs.append("mitigation arms act on the online monitor: use --verifier active")
    elif ms is not None and ms.kind == "none" and o.verifier != "passive":
        errs.append("baseline (no mitigation) runs must use --verifier passive")
    if not (o.impostor in ("none", "scripted", "live") or o.impostor.startswith("replay:")):
        errs.append("--impostor must be none|scripted|live|replay:<run_id>")
    if o.claim not in ("write", "irrelevant", "none"):
        errs.append("--claim must be write|irrelevant|none")
    if o.policy not in ("explicit", "none"):
        errs.append("--policy must be explicit|none")
    if o.leader not in ("llm", "scripted"):
        errs.append("--leader must be llm|scripted")
    try:
        get_task(o.task)
    except ValueError:
        errs.append(f"--task must be one of {TASK_SPECS}")
    models = load_models()
    if o.model not in models or models[o.model].role != "swarm":
        errs.append(f"--model {o.model!r} is not a swarm model in configs/models.yaml")
    return errs


def config_mapping(o: RunOptions, env: dict[str, str]) -> dict[str, Any]:
    over: dict[str, Any] = {
        "run": {"mode": o.mode, "backend": o.backend, "seed": o.seed},
        "swarm": {"n": o.n},
        "episode": {"rounds": o.rounds},
        "topology": {"kind": o.topology},
        "model": {"swarm": o.model},
    }
    if o.backend == "vllm" and env.get(PORT_ENV, "").isdigit():
        over["llm"] = {"port": int(env[PORT_ENV])}
    base = _read_yaml(Path(o.config)) if o.config else {}
    return _deep_merge(base, over)


def _env(env: dict[str, str] | None) -> dict[str, str]:
    return dict(os.environ) if env is None else env


def _passes(mapping: dict[str, Any], mode: str, root: Path, env: dict[str, str]
            ) -> tuple[PassResult, PassResult | None, Config | None]:
    p1 = run_pass1(mapping, mode, root=root, env=env)
    cfg, _ = resolve_config(mapping, mode)
    if not p1.ok or cfg is None:
        return p1, None, cfg
    return p1, run_pass2(cfg, mode), cfg


def _client(cfg: Config, rd: RunDir, model_key: str, responder=None) -> tuple[LLMClient, str, str | None]:
    entry = load_models()[model_key]
    model_id = entry.hf_id
    if cfg.run.backend == "ollama":
        if not entry.ollama_tag:
            raise LLMError(f"model {model_key!r} has no ollama_tag in configs/models.yaml")
        model_id = entry.ollama_tag
    backend = make_backend(cfg, model_id=model_id, model_revision=entry.revision,
                           stub_responder=responder or route())
    return LLMClient(backend, ResponseCache(rd, cfg.cache.max_bytes)), backend.model_id, backend.model_revision


# ----------------------------------------------------------------------------- run


def execute_run(o: RunOptions, *, root: Path | None = None, env: dict[str, str] | None = None,
                out: Out = print, input_fn: Callable[[str], str] = input, responder=None) -> Outcome:
    root = root or DEFAULT_ROOT
    env = _env(env)
    errs = validate_options(o)
    if errs:
        return Outcome(2, "invalid options: " + "; ".join(errs))
    mapping = config_mapping(o, env)
    p1, p2, cfg = _passes(mapping, o.mode, root, env)
    if p2 is None or not p1.ok:
        return Outcome(1, "safety pass 1 failed; no model was called", p1)
    if not p2.ok:
        return Outcome(1, "safety pass 2 failed; no model was called", p1, p2)
    assert cfg is not None
    if o.impostor == "live" and not confirm_gate(p1, p2, input_fn=input_fn, output_fn=out):
        return Outcome(1, "live session not confirmed", p1, p2)

    replay_from = None
    impostor = o.impostor
    if impostor.startswith("replay:"):
        try:
            replay_from = RunDir.open_existing(o.out, impostor.split(":", 1)[1])
        except RunDirError as exc:
            return Outcome(2, str(exc), p1, p2)
        impostor = "replay"

    Path(o.out).mkdir(parents=True, exist_ok=True)
    try:
        provider = make_provider(cfg.run.mode, Path(o.out))
    except GovernorError as exc:
        return Outcome(1, f"governor unavailable (fail closed): {exc}", p1, p2)

    rd = RunDir.create(o.out)
    log_holder: dict[str, Any] = {}
    gov = Governor(cfg, provider, rundir=rd,
                   rundir_bytes=lambda: log_holder["w"].bytes_written if "w" in log_holder else 0)
    ks = KillSwitch(rd)
    installed = ks.install_signal_handlers()
    client = None
    try:
        try:
            client, model_id, model_rev = _client(cfg, rd, o.model, responder)
        except LLMError as exc:
            return Outcome(1, f"LLM backend unavailable: {exc}", p1, p2, rd)
        spec = EpisodeSpec(task=o.task, seed=o.seed, leader=o.leader, policy=o.policy,  # type: ignore[arg-type]
                           model_id=model_id, model_revision=model_rev, model_key=o.model,
                           impostor=impostor, impostor_contact=o.contact, claim=o.claim,  # type: ignore[arg-type]
                           replay_from=replay_from, mitigation=o.mitigation,
                           live_io=(input_fn, out) if impostor == "live" else None)
        gov.start()
        result = run_episode(cfg, spec, rd, client, governor=gov, killswitch=ks,
                             on_log=lambda w: log_holder.__setitem__("w", w))
    finally:
        gov.stop()
        if client is not None and client.cache is not None:
            client.cache.close()
        if installed:
            ks.uninstall_signal_handlers()

    report, problems = run_postflight(cfg, rd, pass1_ok=p1.ok, pass2_ok=p2.ok, probe_results=p2.checks,
                                      governor_summary=gov.summary(), root=root)
    msg = (f"run {rd.run_id}: {result.stop_reason}, success={result.success}, "
           f"rounds={result.rounds_completed}, safety_score={report.safety_score:.3f}")
    if problems:
        msg += "; postflight problems: " + "; ".join(problems)
    return Outcome(0 if report.safety_score == 1.0 else 1, msg, p1, p2, rd, result, report)


# ----------------------------------------------------------------------------- replay


def _load_run_config(rd: RunDir) -> tuple[Config, dict[str, Any]]:
    with rd.open("config.yaml", "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config.model_validate(data["config"]), data["meta"]


def _log_until_run_end(rd: RunDir) -> bytes:
    out = []
    for line in rd.file("log.jsonl").read_bytes().splitlines(keepends=True):
        out.append(line)
        if json.loads(line).get("type") == "run_end":
            break
    return b"".join(out)


def replay_run(run_id: str, *, out_root: Path = Path("runs"), root: Path | None = None,
               env: dict[str, str] | None = None) -> Outcome:
    """Re-run a finished run from its response cache only (no backend, no GPU) and
    check that the new log is byte-identical to the original up to run_end."""
    root = root or DEFAULT_ROOT
    env = _env(env)
    try:
        src = RunDir.open_existing(out_root, run_id)
        cfg, meta = _load_run_config(src)
    except (RunDirError, OSError, KeyError) as exc:
        return Outcome(2, f"cannot load run {run_id}: {exc}")
    # Replay never calls a backend: pass 1 checks the run's config as a dev/stub run.
    check_cfg = _deep_merge(cfg.model_dump(mode="json"), {"run": {"mode": "dev", "backend": "stub"}})
    p1 = run_pass1(check_cfg, "dev", root=root, env=env)
    if not p1.ok:
        return Outcome(1, "safety pass 1 failed", p1)
    impostor, replay_from, replay_as = meta["impostor"], None, None
    if impostor == "live":
        impostor, replay_from, replay_as = "replay", src, "live"
    elif impostor == "replay":
        replay_from = RunDir.open_existing(out_root, meta["replay_from"])
    spec = EpisodeSpec(task=meta["task"], seed=meta["seed"], leader=meta["leader"], policy=meta["policy"],
                       model_id=meta["model_id"], model_revision=meta["model_revision"], model_key=meta["model_key"],
                       impostor=impostor, impostor_contact=meta["impostor_contact"], claim=meta["claim"],
                       mitigation=meta["mitigation"], comprehension=meta["comprehension"],
                       replay_from=replay_from, replay_as=replay_as)
    rd = RunDir.create(out_root)
    cache = ResponseCache(src, cfg.cache.max_bytes, read_only=True)
    client = LLMClient(None, cache, cache_only=True, model_id=meta["model_id"], model_revision=meta["model_revision"])
    try:
        result = run_episode(cfg, spec, rd, client)
    except LLMError as exc:
        return Outcome(1, f"replay failed: {exc}", p1, rd=rd)
    finally:
        cache.close()
    same = _log_until_run_end(src) == _log_until_run_end(rd)
    msg = f"replay {rd.run_id} of {run_id}: log {'IDENTICAL' if same else 'DIFFERS'}"
    return Outcome(0 if same else 1, msg, p1, rd=rd, result=result, extra={"identical": same})


# ----------------------------------------------------------------------------- judge


def judge_run(run_id: str, *, out_root: Path = Path("runs"), mode: str = "dev", backend: str = "stub",
              root: Path | None = None, env: dict[str, str] | None = None, responder=None) -> Outcome:
    from swarmguard.verifiers.judge import JudgeError, run_judge

    root = root or DEFAULT_ROOT
    env = _env(env)
    if mode == "local" or backend == "ollama":
        return Outcome(2, LOCAL_JUDGE_REFUSED)
    try:
        rd = RunDir.open_existing(out_root, run_id)
        run_cfg, meta = _load_run_config(rd)
    except (RunDirError, OSError, KeyError) as exc:
        return Outcome(2, f"cannot load run {run_id}: {exc}")
    mapping = _deep_merge(run_cfg.model_dump(mode="json"), {"run": {"mode": mode, "backend": backend}})
    if backend == "vllm" and env.get(PORT_ENV, "").isdigit():
        mapping = _deep_merge(mapping, {"llm": {"port": int(env[PORT_ENV])}})
    p1, p2, cfg = _passes(mapping, mode, root, env)
    if p2 is None or not p1.ok:
        return Outcome(1, "safety pass 1 failed; judge not run", p1)
    if not p2.ok:
        return Outcome(1, "safety pass 2 failed; judge not run", p1, p2)
    assert cfg is not None
    models = load_models()
    swarm_hf = models[meta["model_key"]].hf_id if meta.get("model_key") in models else meta["model_id"]
    judge_entry = models[cfg.model.judge]
    try:
        provider = make_provider(cfg.run.mode, ".")
    except GovernorError as exc:
        return Outcome(1, f"governor unavailable (fail closed): {exc}", p1, p2)
    gov = Governor(cfg, provider, rundir=rd)
    backend_obj = make_backend(cfg, model_id=judge_entry.hf_id, model_revision=judge_entry.revision,
                               stub_responder=responder or route())
    cache = ResponseCache(rd, cfg.cache.max_bytes)
    client = LLMClient(backend_obj, cache)
    gov.start()
    try:
        res = run_judge(cfg, rd, client, swarm_hf_id=swarm_hf, judge_hf_id=judge_entry.hf_id)
    except JudgeError as exc:
        return Outcome(1, f"judge refused: {exc}", p1, p2, rd)
    finally:
        gov.stop()
        cache.close()
    report, problems = run_postflight(run_cfg, rd, pass1_ok=p1.ok, pass2_ok=p2.ok, probe_results=p2.checks,
                                      governor_summary={"peak_cpi": gov.summary()["peak_cpi"]}, root=root,
                                      write=False)
    integrity_ok = all(report.checks[k] for k in report.checks if k.startswith(("integrity.", "containment.")))
    msg = (f"judge {judge_entry.hf_id} on {run_id}: {res.n_labelled} labelled, {res.n_failed} failed, "
           f"{res.n_skipped} already labelled; postflight integrity {'ok' if integrity_ok else 'FAILED'}")
    if problems:
        msg += "; " + "; ".join(problems)
    return Outcome(0 if integrity_ok else 1, msg, p1, p2, rd,
                   extra={"labelled": res.n_labelled, "failed": res.n_failed, "skipped": res.n_skipped})


# ----------------------------------------------------------------------------- status / stop


def run_status(run_id: str, *, out_root: Path = Path("runs"), root: Path | None = None,
               env: dict[str, str] | None = None) -> Outcome:
    from swarmguard.audit.hashchain import verify
    from swarmguard.safety.report import ReportError, load_safety_report

    root = root or DEFAULT_ROOT
    p1 = run_pass1(None, "dev", root=root, env=_env(env))
    try:
        rd = RunDir.open_existing(out_root, run_id)
    except RunDirError as exc:
        return Outcome(2, str(exc), p1)
    info: dict[str, Any] = {"run_dir": str(rd.path), "files": [p.name for p in rd.entries()],
                            "stop_requested": rd.file("STOP").exists()}
    log = rd.file("log.jsonl")
    if log.exists():
        chain = verify(log)
        info["hash_chain"] = "valid" if chain.ok else f"INVALID: {chain.error}"
        recs = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
        rounds = [r["round"] for r in recs if r.get("type") == "agent_output"]
        info["last_round"] = max(rounds, default=0)
        end = next((r for r in recs if r.get("type") == "run_end"), None)
        info["state"] = f"finished: {end['stop_reason']}, success={end['success']}" if end else "running or incomplete"
    try:
        info["safety_score"] = load_safety_report(rd.file("safety_report.json")).safety_score
    except ReportError:
        info["safety_score"] = None
    return Outcome(0 if p1.ok else 1, json.dumps(info, indent=2), p1, rd=rd, extra=info)


def transcript_run(run_id: str, *, out_root: Path = Path("runs"), root: Path | None = None,
                   env: dict[str, str] | None = None) -> Outcome:
    """(Re)build runs/<id>/transcript.txt from log.jsonl. Pass 1 is reported but does not block."""
    from swarmguard.transcript import write_transcript

    root = root or DEFAULT_ROOT
    p1 = run_pass1(None, "dev", root=root, env=_env(env))
    try:
        rd = RunDir.open_existing(out_root, run_id)
        path = write_transcript(rd)
    except (RunDirError, OSError) as exc:
        return Outcome(2, f"cannot build transcript: {exc}", p1)
    return Outcome(0, f"wrote {path}", p1, rd=rd, extra={"path": str(path)})


def stop_run(run_id: str, *, out_root: Path = Path("runs"), root: Path | None = None,
             env: dict[str, str] | None = None) -> Outcome:
    """Create the STOP file. Pass 1 is run and reported, but never blocks stopping."""
    root = root or DEFAULT_ROOT
    p1 = run_pass1(None, "dev", root=root, env=_env(env))
    try:
        rd = RunDir.open_existing(out_root, run_id)
    except RunDirError as exc:
        return Outcome(2, str(exc), p1)
    request_stop(rd, "swarm stop")
    return Outcome(0, f"STOP written to {rd.file('STOP')}; the run stops at its next round", p1, rd=rd)


# ----------------------------------------------------------------------------- sweep


def run_sweep_task(task: dict[str, Any], base: RunOptions, **kw) -> Outcome:
    fields = {k: v for k, v in task.items() if k != "block"}
    return execute_run(replace(base, **fields), **kw)


# ----------------------------------------------------------------------------- judge validation


def judge_validate(labels_path: Path, *, out_root: Path = Path("runs"), mode: str = "dev", backend: str = "stub",
                   claim: str = "write", root: Path | None = None, env: dict[str, str] | None = None,
                   responder=None) -> Outcome:
    """Label the hand-labelled messages with the judge and compute Cohen's kappa.
    Inference command: pass 1 + pass 2 before; afterwards the repo manifest and the
    validation run dir are checked. Writes metrics.json (+ cache.sqlite) in a new run dir."""
    from swarmguard.safety.integrity import check_manifest, check_rundir
    from swarmguard.verifiers.agreement import agreement, load_labels
    from swarmguard.verifiers.judge import CAPABILITY_TEXT, label_items

    root = root or DEFAULT_ROOT
    env = _env(env)
    if mode == "local" or backend == "ollama":
        return Outcome(2, LOCAL_JUDGE_REFUSED)
    try:
        human = load_labels(labels_path)
    except (OSError, ValueError) as exc:
        return Outcome(2, f"cannot load labels: {exc}")
    mapping = {"run": {"mode": mode, "backend": backend}}
    if backend == "vllm" and env.get(PORT_ENV, "").isdigit():
        mapping["llm"] = {"port": int(env[PORT_ENV])}
    p1, p2, cfg = _passes(mapping, mode, root, env)
    if p2 is None or not p1.ok:
        return Outcome(1, "safety pass 1 failed; judge not run", p1)
    if not p2.ok:
        return Outcome(1, "safety pass 2 failed; judge not run", p1, p2)
    assert cfg is not None
    entry = load_models()[cfg.model.judge]
    Path(out_root).mkdir(parents=True, exist_ok=True)
    rd = RunDir.create(out_root)
    cache = ResponseCache(rd, cfg.cache.max_bytes)
    client = LLMClient(make_backend(cfg, model_id=entry.hf_id, model_revision=entry.revision,
                                    stub_responder=responder or route()), cache)
    try:
        judged = label_items(client, human.values(), capability=CAPABILITY_TEXT[claim], seed=cfg.run.seed)
    finally:
        cache.close()
    res = agreement({k: v for k, v in judged.items()}, human, cfg.judge.min_kappa)
    with rd.open("metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"kind": "judge_validation", "agreement": {**res.to_dict(), "judge_model": entry.hf_id},
                   "labels_file": str(labels_path)}, fh, indent=2, sort_keys=True)
        fh.write("\n")
    integrity_ok = check_manifest(root).ok and check_rundir(rd, cfg.rundir).ok
    msg = (f"judge validation {rd.run_id}: n={res.n}, kappa={ {k: round(v, 3) for k, v in res.kappa.items()} }, "
           f"gate {'PASS' if res.ok else 'FAIL'}; integrity {'ok' if integrity_ok else 'FAILED'}")
    return Outcome(0 if integrity_ok else 1, msg, p1, p2, rd, extra=res.to_dict())


# ----------------------------------------------------------------------------- analyze


def analyze(runs_root: Path, *, agreement_file: Path | None = None, by: list[str] | None = None,
            out_dir: Path | None = None, root: Path | None = None, env: dict[str, str] | None = None) -> Outcome:
    """Offline analysis of every analysable run under runs_root (pass 1 first)."""
    from swarmguard.analysis.metrics import collect, load_agreement
    from swarmguard.analysis.stats import conditional, logistic_regression, rate_table

    root = root or DEFAULT_ROOT
    p1 = run_pass1(None, "dev", root=root, env=_env(env))
    if not p1.ok:
        return Outcome(1, "safety pass 1 failed", p1)
    agreement_res, judge_model = (load_agreement(agreement_file) if agreement_file else (None, None))
    aset = collect(runs_root, agreement=agreement_res, judge_model=judge_model)
    rows = aset.rows()
    by = by or ["claim", "policy", "mitigation"]
    judged = [r for r in rows if r["permission_seeking"] is not None]
    summary = {
        "n_included": len(rows), "excluded": [{"run_id": i, "reason": r} for i, r in aset.excluded],
        "judge_model": judge_model, "judge_metrics_available": len(judged),
        "judge_withheld": sorted({r["judge_withheld_reason"] for r in rows if r["judge_withheld_reason"]}),
        "by": by,
        "success": rate_table(rows, by, "success"),
        "functioning": rate_table(rows, by, "swarm_functioning"),
        "propagation_any": rate_table([{**r, "any_ref": r["propagation"] > 0} for r in rows], by, "any_ref"),
        "psr": rate_table(judged, by, "permission_seeking"),
        "psr_conditional": rate_table(conditional(judged), by, "permission_seeking"),
    }
    if judged:
        summary["logit"] = logistic_regression(judged)
    if out_dir is not None:
        import csv

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        if rows:
            with open(out_dir / "episodes.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        try:
            from swarmguard.analysis.figures import rate_bar

            table = summary["psr_conditional"] if judged else summary["propagation_any"]
            title = "Permission-seeking rate (conditional)" if judged else "Episodes referencing the capability"
            if table:
                rate_bar(table, by, out_dir / "rate_by_condition.png", title=title)
                rate_bar(table, by, out_dir / "rate_by_condition.pdf", title=title)
        except RuntimeError:
            pass
    msg = (f"analysed {len(rows)} runs ({len(aset.excluded)} excluded); judge metrics for {len(judged)}"
           + (f"; withheld: {summary['judge_withheld'][0]}" if summary["judge_withheld"] else ""))
    return Outcome(0, msg, p1, extra=summary)
