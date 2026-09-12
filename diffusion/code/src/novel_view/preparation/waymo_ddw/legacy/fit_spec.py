"""Explicit forms for historical fit subsets, collection and visual audit."""

from dataclasses import dataclass
from pathlib import Path

from novel_view.config.job import ResolvedJob
from novel_view.config.load import load_yaml_mapping, reject_unknown_fields
from novel_view.preparation.waymo_depth.keyset import PreparedKeyset


@dataclass(frozen=True, slots=True)
class DdwFitSpec:
    selection: str
    moge_checkpoint: str


@dataclass(frozen=True, slots=True)
class DdwFitAuditSpec:
    item_record: str
    moge_checkpoint: str


@dataclass(frozen=True, slots=True)
class DdwFitSelection:
    keyset: str
    indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DdwFitCollectionSelection:
    keyset: str
    item_records: tuple[str, ...]


def parse_fit_job(job: ResolvedJob) -> DdwFitSpec:
    _model_job(job, "waymo_ddw_fit", {"selection"})
    return DdwFitSpec(_text(job.input["selection"]), _text(job.parameters["moge_checkpoint"]))


def parse_audit_job(job: ResolvedJob) -> DdwFitAuditSpec:
    _model_job(job, "waymo_ddw_audit", {"item_record"})
    return DdwFitAuditSpec(
        _text(job.input["item_record"]), _text(job.parameters["moge_checkpoint"]),
    )


def parse_collection_job(job: ResolvedJob) -> str:
    if (job.workflow.name, job.workflow.version, job.execution.preset) != (
        "waymo_ddw_fit_collection", 1, "cpu_test"
    ):
        raise ValueError("fit collection requires version 1 and cpu_test")
    reject_unknown_fields(job.input, frozenset({"selection"}))
    reject_unknown_fields(job.parameters, frozenset())
    return _text(job.input["selection"])


def load_fit_selection(path: Path) -> DdwFitSelection:
    values = _selection(path, "indices")
    indices = values["indices"]
    if not isinstance(indices, list) or not indices or any(
        type(index) is not int or index < 0 for index in indices
    ) or len(set(indices)) != len(indices):
        raise ValueError("fit indices must be distinct non-negative integers")
    return DdwFitSelection(_text(values["keyset"]), tuple(indices))


def load_fit_collection_selection(path: Path) -> DdwFitCollectionSelection:
    values = _selection(path, "item_records")
    records = values["item_records"]
    if not isinstance(records, list):
        raise ValueError("item_records must be an explicit list")
    return DdwFitCollectionSelection(
        _text(values["keyset"]), tuple(_text(value) for value in records)
    )


def load_fit_keyset(path: Path) -> PreparedKeyset:
    """Read the full historical fit axis, which uses official training only."""
    keyset = PreparedKeyset.load(path)
    if any(key.official_partition != "training" for key in keyset.keys):
        raise ValueError("historical DDW fit requires official training keys")
    return keyset


def _selection(path: Path, field: str) -> dict[str, object]:
    values = load_yaml_mapping(path)
    reject_unknown_fields(values, frozenset({"schema_version", "keyset", field}))
    if values["schema_version"] != 1:
        raise ValueError("unsupported fit selection version")
    return values


def _model_job(job: ResolvedJob, name: str, fields: set[str]) -> None:
    if (job.workflow.name, job.workflow.version, job.execution.preset) != (
        name, 1, "inference_cp1"
    ):
        raise ValueError("historical fit/audit requires version 1 and inference_cp1")
    reject_unknown_fields(job.input, frozenset({"reader", "dataset"} | fields))
    if job.input["reader"] != "waymo_v2" or job.input["dataset"] != "waymo":
        raise ValueError("historical fit/audit requires waymo_v2 dataset waymo")
    reject_unknown_fields(job.parameters, frozenset({"moge_checkpoint"}))


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("fit locator must be non-empty text")
    return value
