"""Strict small-record readers used by the fixed host acceptance command."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="novel_view.cli.acceptance")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("checkpoint", "selection"):
        command = commands.add_parser(name)
        command.add_argument("--version", type=int, choices=(1, 2), required=True)
        command.add_argument("--path", type=Path, required=True)
        command.add_argument("--expected-attempt", required=True)
    candidate = commands.add_parser("waymo-candidate")
    candidate.add_argument("--path", type=Path, required=True)
    candidate.add_argument("--clip-selection", type=Path, required=True)
    gate = commands.add_parser("waymo-depth-selection")
    gate.add_argument("--path", type=Path, required=True)
    gate.add_argument("--reports-selection", type=Path, required=True)
    ddw = commands.add_parser("waymo-ddw")
    ddw.add_argument("--path", type=Path, required=True)
    ddw.add_argument("--clip-selection", type=Path, required=True)
    ddw.add_argument("--expected-depth-selection", required=True)
    ddw.add_argument("--kind", choices=("canary-v1", "canary-v2", "probe", "survey"), required=True)
    return parser


def _checkpoint(path: Path, version: int, expected_attempt: str) -> None:
    if version == 1:
        from novel_view.training.gen3c.lora.record import read_checkpoint_record_v1

        record = read_checkpoint_record_v1(path)
    else:
        from novel_view.training.gen3c.lora.depth_record import (
            read_checkpoint_record_v2,
        )

        record = read_checkpoint_record_v2(path)
    if record.source_attempt != expected_attempt:
        raise ValueError("checkpoint record belongs to another attempt")
    checkpoint_name = PurePosixPath(record.checkpoint)
    if checkpoint_name.is_absolute() or checkpoint_name.name != record.checkpoint:
        raise ValueError("checkpoint record must name one sibling checkpoint")
    checkpoint = PurePosixPath(
        "/runs", _attempt_directory(expected_attempt), "checkpoints", checkpoint_name
    )
    print(f"source_attempt={record.source_attempt}")
    print(f"completed_step={record.completed_step}")
    print(f"checkpoint={checkpoint}")


def _selection(path: Path, version: int, expected_attempt: str) -> None:
    from novel_view.training.gen3c.checkpoint_selection import (
        read_checkpoint_selection_v1,
        read_checkpoint_selection_v2,
    )

    selection = (
        read_checkpoint_selection_v1(path)
        if version == 1
        else read_checkpoint_selection_v2(path)
    )
    prefix = _attempt_directory(expected_attempt) / "checkpoints"
    locators = (*selection.input_checkpoint_records, selection.checkpoint)
    if any(PurePosixPath(locator).parent != prefix for locator in locators):
        raise ValueError("checkpoint selection mixes training attempts")
    print(f"training_attempt={expected_attempt}")
    print(f"completed_step={selection.completed_step}")
    print(f"checkpoint_record={selection.selected_checkpoint_record}")
    print(f"checkpoint={selection.checkpoint}")


def _attempt_directory(reference: str) -> PurePosixPath:
    parts = reference.split("/")
    if len(parts) != 3 or any(part in ("", ".", "..") for part in parts):
        raise ValueError("expected attempt must be job/run/attempt")
    return PurePosixPath(parts[0], parts[1], "attempts", parts[2])


def _waymo_result(
    path: Path,
    format_name: str,
    outcome: str,
    *,
    expected_exit: int = 0,
    selected_backend: str = "none",
) -> None:
    from novel_view.cli.runs import read_attempt_record

    attempt = read_attempt_record(path.with_name("attempt.json"))
    state = "succeeded" if expected_exit == 0 else "failed"
    if attempt["worker_exit_code"] != expected_exit or attempt["recorded_state"] != state:
        raise ValueError("outcome disagrees with the recorded worker exit/state")
    allowed = outcome == "selected" and selected_backend == "moge-v1-lidar-scale"
    print(f"format={format_name}")
    print(f"outcome={outcome}")
    print(f"expected_exit={expected_exit}")
    print(f"selected_backend={selected_backend}")
    print(f"ddw_allowed={str(allowed).lower()}")


def _waymo_candidate(path: Path, clip_selection: Path) -> None:
    from novel_view.preparation.waymo_depth.candidate_record import (
        CANDIDATE_RECORD_VERSION,
        read_candidate_record,
    )
    from novel_view.preparation.waymo_depth.spec import load_depth_clip_selection

    record = read_candidate_record(path)
    if record.clip_key != load_depth_clip_selection(clip_selection):
        raise ValueError("candidate record belongs to another clip")
    _waymo_result(path, CANDIDATE_RECORD_VERSION, "completed")


def _waymo_depth_selection(path: Path, reports_selection: Path) -> None:
    from novel_view.preparation.waymo_depth.selection import DepthGateDecision
    from novel_view.preparation.waymo_depth.selection_record import (
        SELECTION_RECORD_VERSION,
        read_selection_record,
    )
    from novel_view.preparation.waymo_depth.spec import load_depth_report_selection

    record = read_selection_record(path)
    if record.schema_version != SELECTION_RECORD_VERSION or record.workflow_version != 2:
        raise ValueError("acceptance requires the current v2 depth gate result")
    if record.candidate_results != load_depth_report_selection(reports_selection):
        raise ValueError("depth gate did not use the complete ordered report selection")
    selected = isinstance(record.outcome, DepthGateDecision)
    _waymo_result(
        path,
        SELECTION_RECORD_VERSION,
        "selected" if selected else "scientific_stop",
        selected_backend=record.outcome.selected_backend if selected else "none",
    )


def _waymo_ddw(path: Path, kind: str, clip_selection: Path, expected_depth_selection: str) -> None:
    from novel_view.preparation.waymo_ddw.legacy.record import read_legacy_ddw_outcome
    from novel_view.preparation.waymo_depth.spec import load_depth_clip_selection

    record = read_legacy_ddw_outcome(path, kind)
    if record.clip_key != load_depth_clip_selection(clip_selection) or (
        record.depth_selection != expected_depth_selection
    ):
        raise ValueError("DDW result belongs to another clip or depth selection")
    _waymo_result(
        path,
        record.format,
        record.status,
        expected_exit=2 if record.status == "scientific_stop" else 0,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "checkpoint":
        _checkpoint(arguments.path, arguments.version, arguments.expected_attempt)
    elif arguments.command == "selection":
        _selection(arguments.path, arguments.version, arguments.expected_attempt)
    elif arguments.command == "waymo-candidate":
        _waymo_candidate(arguments.path, arguments.clip_selection)
    elif arguments.command == "waymo-depth-selection":
        _waymo_depth_selection(arguments.path, arguments.reports_selection)
    else:
        _waymo_ddw(
            arguments.path,
            arguments.kind,
            arguments.clip_selection,
            arguments.expected_depth_selection,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
