
import sys
import traceback

from .config import CONSOLE_ERROR_RED

RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def print_error(message, exc=None):
    if CONSOLE_ERROR_RED:
        prefix = RED
        suffix = RESET
    else:
        prefix = ""
        suffix = ""

    print(f"{prefix}{message}{suffix}", file=sys.stderr)

    if exc is not None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print(f"{prefix}{tb}{suffix}", file=sys.stderr)


def print_warning(message):
    if CONSOLE_ERROR_RED:
        print(f"{YELLOW}{message}{RESET}", file=sys.stderr)
    else:
        print(message, file=sys.stderr)
