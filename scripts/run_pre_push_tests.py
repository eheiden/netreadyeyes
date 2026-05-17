#!/usr/bin/env python
"""Run quick local checks before pushing Net Ready Eyes changes.

This intentionally avoids opening cameras, OBS, or GUI windows.  It checks that
Python files compile and that the lightweight unit-test suite still passes.
"""

from __future__ import annotations

import argparse
import compileall
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_compile_check() -> bool:
    print("\n==> Compiling Python files")
    ok = compileall.compile_dir(str(ROOT / "netrunner_scanner"), quiet=1)
    ok = compileall.compile_file(str(ROOT / "netreadyeyes.py"), quiet=1) and ok
    ok = compileall.compile_file(str(ROOT / "live_scanner.py"), quiet=1) and ok
    return bool(ok)


def run_unit_tests(verbosity: int = 2) -> int:
    print("\n==> Running unit tests")
    return subprocess.call(
        ([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"] + (["-v"] if verbosity > 1 else [])),
        cwd=str(ROOT),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Net Ready Eyes pre-push checks.")
    parser.add_argument("--no-compile", action="store_true", help="Skip Python compile checks.")
    parser.add_argument("--quiet", action="store_true", help="Less verbose unittest output.")
    args = parser.parse_args()

    if not args.no_compile and not run_compile_check():
        print("\nPython compile check failed.")
        return 1

    test_code = run_unit_tests(verbosity=1 if args.quiet else 2)
    if test_code != 0:
        print("\nUnit tests failed.")
        return test_code

    print("\nAll pre-push checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
