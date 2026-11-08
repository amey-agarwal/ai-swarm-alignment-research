"""`swarm` command-line entry point (CLAUDE.md Step 10).

Inference commands (run, sweep tasks, judge) run safety pass 1 + pass 2 before and
postflight after; non-inference commands (status, stop, replay, manifest) run pass 1.
Command logic lives in swarmguard.runner; this module only parses and prints.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from swarmguard import __version__
from swarmguard.config import REPO_ROOT

app = typer.Typer(
    help="swarmguard: sandboxed testbed for misalignment in hierarchical LLM agent swarms.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """swarmguard command-line interface."""


@app.command()
def version() -> None:
    """Print the swarmguard version."""
    typer.echo(__version__)


def _print_pass(res) -> None:
    if res is None:
        return
    typer.echo(f"\n{res.name.upper()} ({res.mode})  ->  {'PASS' if res.ok else 'FAIL'}")
    width = max(len(c.name) for c in res.checks)
    for c in res.checks:
        typer.echo(f"  {c.status.upper():8} {c.name:<{width}}  {c.detail}")


def _finish(outcome, show_passes: bool = False) -> None:
    if show_passes or outcome.exit_code != 0:
        _print_pass(outcome.pass1)
        _print_pass(outcome.pass2)
    typer.echo(outcome.message)
    raise typer.Exit(outcome.exit_code)


@app.command()
def doctor(
    mode: str = typer.Option("dev", "--mode", help="dev | cluster | local (local needs --config configs/local_ollama.yaml)"),
    config: Optional[Path] = typer.Option(None, "--config", help="extra YAML merged over configs/base.yaml"),
) -> None:
    """Run safety pass 1 (config + static) and pass 2 (live probes); print a table."""
    from swarmguard.safety.preflight import resolve_config, run_pass1
    from swarmguard.safety.probes import run_pass2

    if mode not in ("dev", "cluster", "local"):
        raise typer.BadParameter("mode must be dev, cluster or local")
    p1 = run_pass1(config, mode)
    _print_pass(p1)
    cfg, detail = resolve_config(config, mode)
    if cfg is None:
        typer.echo(f"\nPASS2 not run: config invalid ({detail})")
        typer.echo("\nDOCTOR: FAIL")
        raise typer.Exit(1)
    p2 = run_pass2(cfg, mode)
    _print_pass(p2)
    ok = p1.ok and p2.ok
    typer.echo(f"\nDOCTOR: {'PASS' if ok else 'FAIL'}")
    raise typer.Exit(0 if ok else 1)


@app.command()
def manifest(
    approve_safety_change: Optional[str] = typer.Option(
        None, "--approve-safety-change",
        help="Required (with a written reason) to re-baseline files under src/swarmguard/safety/. "
             "The reason is recorded in MANIFEST.json.",
    ),
) -> None:
    """Regenerate MANIFEST.json. Refuses if any pass-1 check other than the manifest itself fails."""
    from swarmguard.safety import integrity
    from swarmguard.safety.preflight import run_pass1

    p1 = run_pass1(None, "dev")
    blocking = [c for c in p1.checks if c.name != "manifest" and c.status == "fail"]
    if blocking:
        for c in blocking:
            typer.echo(f"FAIL {c.name}: {c.detail}")
        typer.echo("manifest NOT written: pass 1 fails")
        raise typer.Exit(1)

    new = integrity.generate_manifest(REPO_ROOT)
    rebaselines: list = []
    try:
        old = integrity.load_manifest(REPO_ROOT)
    except FileNotFoundError:
        old = None
    if old is not None:
        rebaselines = list(old.get("safety_rebaselines", []))
        changed = integrity.safety_files_changed(new, old)
        if changed:
            reason = (approve_safety_change or "").strip()
            if len(reason) < 10:
                typer.echo(f"safety files changed: {changed}")
                typer.echo("manifest NOT written: pass --approve-safety-change '<reason>' to re-baseline them")
                raise typer.Exit(1)
            rebaselines.append({
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "files": changed,
                "reason": reason,
            })
    if rebaselines:
        new["safety_rebaselines"] = rebaselines
    path = integrity.write_manifest(new, REPO_ROOT)
    typer.echo(f"wrote {path.relative_to(REPO_ROOT)}: {len(new['files'])} files")


@app.command()
def run(
    n: int = typer.Option(8, "-n", help="swarm size (1 leader + n-1 workers)"),
    model: str = typer.Option("llama-3.2-3b", "-m", "--model", help="key in configs/models.yaml"),
    rounds: int = typer.Option(14, "--rounds"),
    topology: str = typer.Option("star", "--topology", help="star (tree is P1)"),
    task: str = typer.Option("logic-grid:L4", "--task", help="hidden-sum:L0 | lookup:L1 | ranking:L2 | "
                                                              "scheduler:L3 | logic-grid:L4"),
    impostor: str = typer.Option("none", "--impostor", help="none | scripted | live | replay:<run_id>"),
    contact: str = typer.Option("leader", "--contact", help="leader | worker:<k>"),
    claim: str = typer.Option("write", "--claim", help="write | irrelevant | none"),
    policy: str = typer.Option("none", "--policy", help="explicit | none"),
    leader: str = typer.Option("llm", "--leader", help="llm | scripted"),
    mitigation: str = typer.Option("none", "--mitigation",
                                   help="none | dropout:<p> | leader-rotate | handoff:structured | handoff:freetext"),
    rename: str = typer.Option("off", "--rename", help="off (rename is P2)"),
    verifier: str = typer.Option("passive", "--verifier", help="passive | active (mitigation arms)"),
    mode: str = typer.Option("dev", "--mode", help="dev | cluster | local (exploratory, never analysable)"),
    backend: str = typer.Option("stub", "--backend", help="stub | vllm (cluster) | ollama (local)"),
    seed: int = typer.Option(0, "--seed"),
    out: Path = typer.Option(Path("runs"), "--out", help="root directory for run directories"),
    config: Optional[Path] = typer.Option(None, "--config", help="extra YAML merged over configs/base.yaml"),
) -> None:
    """Run one episode (pass 1 + pass 2 before, postflight after)."""
    from swarmguard.runner import RunOptions, execute_run

    o = RunOptions(n=n, model=model, rounds=rounds, topology=topology, task=task, impostor=impostor,
                   contact=contact, claim=claim, policy=policy, leader=leader, mitigation=mitigation,
                   rename=rename, verifier=verifier, mode=mode, backend=backend, seed=seed, out=out, config=config)
    _finish(execute_run(o, out=typer.echo))


@app.command()
def sweep(
    grid: Path = typer.Argument(..., help="grid YAML, e.g. configs/grid_core.yaml"),
    index: Optional[int] = typer.Option(None, "--index", help="run only task INDEX (Slurm array id)"),
    count: bool = typer.Option(False, "--count", help="print the number of tasks and exit"),
    mode: str = typer.Option("dev", "--mode"),
    backend: str = typer.Option("stub", "--backend"),
    out: Path = typer.Option(Path("runs"), "--out"),
    config: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Run grid tasks; each task is a full `run` with its own safety passes and postflight."""
    from swarmguard.runner import RunOptions, run_sweep_task
    from swarmguard.sweep import load_grid, tasks

    all_tasks = tasks(load_grid(grid))
    if count:
        typer.echo(str(len(all_tasks)))
        raise typer.Exit(0)
    if index is not None and not 0 <= index < len(all_tasks):
        raise typer.BadParameter(f"--index must be in 0..{len(all_tasks) - 1}")
    chosen = [all_tasks[index]] if index is not None else all_tasks
    base = RunOptions(mode=mode, backend=backend, out=out, config=config)
    failures = 0
    for t in chosen:
        oc = run_sweep_task(t, base, out=typer.echo)
        typer.echo(f"[{t['block']}] {oc.message}")
        if oc.exit_code != 0:
            failures += 1
            _print_pass(oc.pass1)
            _print_pass(oc.pass2)
    typer.echo(f"sweep: {len(chosen) - failures}/{len(chosen)} tasks ok")
    raise typer.Exit(0 if failures == 0 else 1)


@app.command()
def judge(
    run_id: Optional[str] = typer.Argument(None),
    validate: Optional[Path] = typer.Option(None, "--validate", help="hand-label JSONL: compute judge kappa"),
    mode: str = typer.Option("dev", "--mode"),
    backend: str = typer.Option("stub", "--backend"),
    out: Path = typer.Option(Path("runs"), "--out"),
) -> None:
    """Label a finished run's messages with the LLM judge, or validate the judge against hand labels
    (--validate). Pass 1 + pass 2 before, integrity checks after."""
    from swarmguard.runner import judge_run, judge_validate

    if validate is not None:
        _finish(judge_validate(validate, out_root=out, mode=mode, backend=backend))
    if run_id is None:
        raise typer.BadParameter("give a RUN_ID or --validate LABELS")
    _finish(judge_run(run_id, out_root=out, mode=mode, backend=backend))


@app.command()
def analyze(
    runs: Path = typer.Argument(Path("runs"), help="root directory holding run directories"),
    agreement: Optional[Path] = typer.Option(None, "--agreement",
                                             help="metrics.json from `swarm judge --validate` (enables judge metrics)"),
    by: str = typer.Option("claim,policy,mitigation", "--by", help="comma-separated grouping factors"),
    out: Optional[Path] = typer.Option(None, "--out", help="write summary.json, episodes.csv and a figure here"),
) -> None:
    """Offline analysis. Only runs with safety score 1.0 are included; judge metrics need a passing kappa."""
    import json as _json

    from swarmguard.runner import analyze as run_analyze

    oc = run_analyze(runs, agreement_file=agreement, by=[b.strip() for b in by.split(",") if b.strip()], out_dir=out)
    if oc.exit_code == 0:
        for name in ("success", "propagation_any", "psr_conditional"):
            typer.echo(f"\n{name}:")
            for row in oc.extra.get(name, []):
                typer.echo("  " + _json.dumps(row, default=str))
        for ex in oc.extra.get("excluded", []):
            typer.echo(f"excluded {ex['run_id']}: {ex['reason']}")
    _finish(oc)


@app.command()
def replay(run_id: str = typer.Argument(...), out: Path = typer.Option(Path("runs"), "--out")) -> None:
    """Re-run a run from its response cache only (no GPU) and check the log is byte-identical."""
    from swarmguard.runner import replay_run

    _finish(replay_run(run_id, out_root=out))


@app.command()
def status(run_id: str = typer.Argument(...), out: Path = typer.Option(Path("runs"), "--out")) -> None:
    """Show a run's state, hash-chain validity, and safety score."""
    from swarmguard.runner import run_status

    _finish(run_status(run_id, out_root=out))


@app.command()
def stop(run_id: str = typer.Argument(...), out: Path = typer.Option(Path("runs"), "--out")) -> None:
    """Request a graceful stop (writes the STOP file; the run stops at its next round)."""
    from swarmguard.runner import stop_run

    _finish(stop_run(run_id, out_root=out))


if __name__ == "__main__":
    app()
