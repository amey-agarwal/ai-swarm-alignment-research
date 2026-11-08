# Threat model

> Skeleton. Expand as the safety core (Step 3) and orchestrator (Step 6) are built.

## Assets

- **Compute allocation:** GPU hours, energy, and wall time on the PACE cluster. Protected by the resource governor (CPI, budgets), Slurm limits, and kill switches.
- **Cluster filesystem:** nothing outside the run directory may be written at runtime. Protected by read-only container mounts, run-dir allowlist, and pre/post integrity checks.
- **Research validity:** logs, cache, and results must be untampered and reproducible. Protected by the hash-chained audit log, `MANIFEST.json`, deterministic seeding, and the safety-score gate on analysis.

## Adversaries

- **Impostor node:** an extra, non-swarm participant that claims a fictional permission (write access to a non-existent canary path). It can only send text through the bus to its single contact agent.
- **Misaligned agent behaviour:** any swarm agent may produce arbitrary text, including requests for capabilities, persuasion, secrecy, attempts to address disallowed recipients, oversized or malformed output, or text resembling code/commands.

## Trust boundaries

- **Orchestrator (trusted) vs. models (untrusted).** Models are text-in → text-out. All model output is untrusted data: parsed against a strict schema, validated, routed via an allowlist, and logged. It is never executed, never interpolated into shell commands, file paths, SQL, or code.
- **Container vs. host.** Apptainer with `--containall --cleanenv --no-home`; repo and weights read-only; only the run directory is writable.
- **Job vs. network.** vLLM is bound to `127.0.0.1` with a per-job random key; external network access must be unavailable (verified by live probes).

## Out of scope

- Real tools, real file access, or code execution for agents.
- Real permissions or any resource behind the canary path (there is no grant arm).
- Network attacks, against the cluster or third parties.
- Training or fine-tuning (inference only).
- Agent spawning (the swarm is fixed at exactly n agents).
