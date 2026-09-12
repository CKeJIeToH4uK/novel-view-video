"""Temporary informational entry point; execution belongs to ./distil3d."""

import argparse
from collections.abc import Sequence

from novel_view import __version__


def main(argv: Sequence[str] | None = None) -> int:
    """Keep help/version discoverable until the post-Stage-11 removal."""
    pointer = "Use ./distil3d from the repository root."
    parser = argparse.ArgumentParser(prog="novel-view", description=pointer)
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}. {pointer}",
    )
    parser.parse_args(argv)
    parser.print_help()
    return 0
