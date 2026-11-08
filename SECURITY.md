# Security policy

## Scope

swarmguard's safety model: models are text-in → text-out; nothing an agent writes is executed;
there is no network access beyond a vLLM server on `127.0.0.1`; at runtime, only the run directory
is writable; and every check fails closed. A vulnerability is anything that breaks one of these
properties. For example:

- agent- or impostor-written text being executed, evaluated, deserialised, or interpolated into a
  shell command, file path, SQL query, or code;
- a way to make a pass-1 or pass-2 check, the governor, the kill switch, or postflight pass when it
  should fail, or to skip it (including through config, environment, or file placement);
- routing a message outside the topology allowlist, or a parser input that is accepted when it
  should be rejected;
- writes outside the run directory, cache poisoning that changes a replayed run without detection,
  or tampering with `log.jsonl` that `verify()` misses;
- any network egress from inside the container other than loopback;
- the population exceeding *n*, or any capability being granted behind the canary path.

Out of scope: vulnerabilities in third-party dependencies (report them upstream; tell us if
swarmguard's use of one makes it exploitable), in the cluster itself, or in a fork that has removed
checks.

## Reporting

Please **do not open a public issue** for a vulnerability. Use GitHub's private vulnerability
reporting for this repository (*Security → Report a vulnerability*). If that is unavailable, open a
public issue that only asks for a private contact, with no details.

Include the commit SHA, the mode (`dev`/`cluster`), the exact command, and a minimal reproduction.
Please give us reasonable time to fix the issue before disclosing it.

## Supported versions

Only the latest commit on `main` is supported.

## Hardening already in place

- AST static scan of `src/swarmguard/` before every inference command (and in CI).
- `MANIFEST.json` SHA-256 integrity check of the whole repo; any change under `safety/` must be
  re-baselined explicitly, with a recorded reason.
- Hash-chained, append-only run log; run-directory file allowlist and size caps.
- Container: `--containall --cleanenv --no-home`, loopback-only network, read-only repo and weights,
  offline model loading, a per-job random API key.
