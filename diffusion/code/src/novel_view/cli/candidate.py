"""CLI записи и безопасного чтения candidate."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from novel_view.runtime.candidate import (
    build_candidate,
    read_candidate,
    write_candidate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m novel_view.cli.candidate")
    commands = parser.add_subparsers(dest="command", required=True)

    write = commands.add_parser("write")
    write.add_argument("--source-revision", required=True)
    write.add_argument("--lock-revision", required=True)
    write.add_argument("--core-image-id", required=True)
    write.add_argument("--moge-image-id", required=True)
    write.add_argument("--output", type=Path, required=True)

    read = commands.add_parser("read")
    read.add_argument("--candidate", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "write":
            candidate = build_candidate(
                source_revision=args.source_revision,
                lock_revision=args.lock_revision,
                core_image_id=args.core_image_id,
                moge_image_id=args.moge_image_id,
                lock_inventory=sys.stdin.read(),
            )
            write_candidate(args.output, candidate)
            return 0

        candidate = read_candidate(args.candidate)
        print(f"candidate_id={candidate.candidate_id}")
        print(f"source_revision={candidate.source_revision}")
        print("source_dirty=false")
        print(f"platform={candidate.platform}")
        print(f"lock_revision={candidate.lock_revision}")
        print(f"core_tag={candidate.images.core.tag}")
        print(f"core_image_id={candidate.images.core.image_id}")
        print(f"moge_tag={candidate.images.moge.tag}")
        print(f"moge_image_id={candidate.images.moge.image_id}")
        print(f"transport={candidate.transport}")
        return 0
    except (OSError, ValueError) as error:
        print(f"candidate: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
