import importlib.util
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_MODULES = [
    "cv2",
    "numpy",
    "PIL",
    "collector_vision",
]

OPTIONAL_TOOLS = [
    "node",
    "npm",
    "nvm",
]


def check_python():
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print("  ERROR: Python 3.10+ is recommended.")
        return False
    return True


def check_modules():
    ok = True
    print("\nPython modules:")
    for module in REQUIRED_MODULES:
        found = importlib.util.find_spec(module) is not None
        print(f"  {module:18} {'OK' if found else 'MISSING'}")
        ok = ok and found
    return ok


def check_tools():
    print("\nCommand-line tools:")
    for tool in OPTIONAL_TOOLS:
        path = shutil.which(tool)
        print(f"  {tool:18} {path or 'not found'}")


def check_catalog():
    print("\nProject files:")
    catalog = Path("netrunner-catalog.npz")
    print(f"  netrunner-catalog.npz {'OK' if catalog.exists() else 'MISSING - build or copy a catalog before running'}")


def main():
    print("Net Ready Eyes setup check")
    print(f"Platform: {platform.platform()}")
    py_ok = check_python()
    mods_ok = check_modules()
    check_tools()
    check_catalog()

    if py_ok and mods_ok:
        print("\nCore Python setup looks usable.")
    else:
        print("\nSetup needs attention. Install missing dependencies, then rerun this check.")


if __name__ == "__main__":
    main()
