#!/usr/bin/env python3
"""PreToolUse hook: block Edit/Write on protected paths (exit code 2 + message).

Duplicates the permissions.deny rules in .claude/settings.json on purpose.
Fails closed: unreadable input or a missing file path blocks the tool call.

Protected:
  - container/**
  - .claude/**
  - src/swarmguard/safety/**   (only once .claude/safety_locked exists; created at end of Step 3)
"""

import json
import os
import sys
from pathlib import Path

ALWAYS_PROTECTED = ("container", ".claude")
SAFETY_DIR = ("src", "swarmguard", "safety")
SAFETY_LOCK_MARKER = (".claude", "safety_locked")


def block(msg: str) -> int:
    print(f"protect_paths: BLOCKED: {msg}", file=sys.stderr)
    return 2


def is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return block("could not parse hook input (fail closed)")

    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict) or not tool_input.get("file_path"):
        return block("no tool_input.file_path in hook input (fail closed)")

    project_dir = Path(
        os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or Path(__file__).resolve().parents[2]
    ).resolve()
    target = Path(str(tool_input["file_path"]))
    if not target.is_absolute():
        target = project_dir / target
    # resolve() normalises '..' and follows symlinks so they cannot be used to dodge the check.
    target = target.resolve()

    protected = [project_dir.joinpath(p) for p in ALWAYS_PROTECTED]
    if project_dir.joinpath(*SAFETY_LOCK_MARKER).exists():
        protected.append(project_dir.joinpath(*SAFETY_DIR))

    for root in protected:
        if is_under(target, root.resolve()):
            rel = root.relative_to(project_dir)
            return block(
                f"{target} is under protected path '{rel}/'. "
                f"Edits require the user to say 'approved: edit <path>' and to make the change manually."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
