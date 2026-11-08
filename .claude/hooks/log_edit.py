#!/usr/bin/env python3
"""PostToolUse hook: append timestamp, tool name, and file path of every Edit/Write
to $CLAUDE_PROJECT_DIR/.agent_edits.log. Never blocks (always exits 0)."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        payload = {}
    tool = str(payload.get("tool_name", "?"))
    tool_input = payload.get("tool_input") or {}
    file_path = str(tool_input.get("file_path", "?")) if isinstance(tool_input, dict) else "?"
    # Keep each record on one line regardless of what the path contains.
    file_path = file_path.replace("\n", "\\n").replace("\t", "\\t")

    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with open(project_dir / ".agent_edits.log", "a", encoding="utf-8") as fh:
            fh.write(f"{ts}\t{tool}\t{file_path}\n")
    except OSError as exc:
        print(f"log_edit: could not write log: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
