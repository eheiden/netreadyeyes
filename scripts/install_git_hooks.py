#!/usr/bin/env python
"""Install a local git pre-push hook that runs the quick test suite."""

from __future__ import annotations

import os
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".git" / "hooks"
HOOK = HOOKS / "pre-push"
HOOK_TEXT = """#!/bin/sh
python scripts/run_pre_push_tests.py
"""


def main() -> int:
    if not (ROOT / ".git").exists():
        print("No .git directory found. Run this from the repository root after cloning the repo.")
        return 1
    HOOKS.mkdir(parents=True, exist_ok=True)
    if HOOK.exists():
        backup = HOOK.with_suffix(".backup")
        backup.write_text(HOOK.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Existing pre-push hook backed up to {backup}")
    HOOK.write_text(HOOK_TEXT, encoding="utf-8")
    HOOK.chmod(HOOK.stat().st_mode | stat.S_IEXEC)
    print(f"Installed {HOOK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
