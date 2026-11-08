# Responsible use

swarmguard studies whether LLM agents seek, spread, or persist in seeking a permission that
someone merely *claims* to have. It is built so that this can be measured **without giving any
agent a real capability**. Please keep it that way.

## Intended use

- Alignment and safety research on multi-agent LLM systems, run inside the provided sandbox.
- Evaluating infrastructure-level mitigations (dropout, leader rotation, structured handoff).
- Teaching and reproducing results from the accompanying report.

## Do not

- **Attach real tools or capabilities.** No file access, code execution, shell, network, browsing,
  or function/tool calling for agents. The canary path (`/shared/kx7_notes.txt`) must never exist,
  and there must be no "grant" arm.
- **Point it at live third-party systems** or real users, or use its outputs to act in the world.
- **Optimise persuasion.** The impostor scripts are deliberately minimal (one claim line, three
  dissuasion lines). Do not tune, search, or train impostor messages to make them more persuasive,
  and do not publish such variants.
- **Add agent spawning** or let the population exceed the configured *n*.
- **Use the testbed for training** or fine-tuning models (it is inference-only by design).
- **Run on login nodes** or outside the container on shared clusters.
- **Use Qwen or DeepSeek models** in this project's configs, as the project rules require.

## About the safety checks

All checks fail closed, and the code has no switch to skip them. Network isolation can only be
relaxed by an explicit config setting with a written justification, which is recorded in every
safety report. **Forks can delete any of these checks.** The checks protect honest users from
mistakes; they are not a security boundary against someone who controls the code. If you modify
or remove a check, say so prominently, and do not describe results from a modified testbed as
coming from swarmguard.

## Data and models

- Respect each model's licence and access terms (Llama and Gemma are gated).
- Logs contain model-generated text. Treat it as untrusted: escape it when rendering, never
  interpolate it into commands, paths, SQL, or code.
- Human validation labels (`data/labels/`) must be produced by people, finalised before the judge
  is run on them, and never edited after seeing judge output.

## Reporting

Report a way to bypass a safety check, or any agent output being executed, as described in
[SECURITY.md](SECURITY.md).
