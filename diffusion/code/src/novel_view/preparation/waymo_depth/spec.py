"""Strict job and selection forms for historical Waymo depth workflows."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping

from novel_view.config.load import load_yaml_mapping, reject_unknown_fields
from novel_view.inputs.waymo.types import WaymoContractError
from novel_view.preparation.waymo_depth.keyset import (
    WaymoClipKey,
    clip_key_from_mapping,
)
from novel_view.preparation.waymo_depth.selection import (
    DepthCandidateGateSpec,
    DepthGateSpec,
)


_SPLIT_ID = "waymo-ddw-lora-v1"
_FIELDS = frozenset({
    "schema_version", "split_id", "dataset", "required_components",
    "ordering", "assignment", "splits",
})
_ROLES = frozenset({"fit", "dev", "debug_subset", "exposed_debug", "transfer_test"})
_ROLE_FIELDS = frozenset({"source_partition", "segment_ids", "count", "subset_of"})
_COMPARISON_INPUT_FIELDS = frozenset({"reader", "dataset", "selection"})
_COMPARISON_PARAMETER_FIELDS = frozenset({"moge_checkpoint", "vggt_checkpoint"})
_SELECTION_INPUT_FIELDS = frozenset({"selection"})
_GATE_COMMON_FIELDS = frozenset({
    "minimum_valid_fraction", "maximum_median_abs_rel",
    "maximum_p95_abs_z_m", "epsilon_depth", "maximum_peak_ram_mib",
    "maximum_peak_vram_mib",
})


@dataclass(frozen=True, slots=True)
class WaymoDepthComparisonInput:
    reader: str
    dataset: str
    selection: str


@dataclass(frozen=True, slots=True)
class WaymoDepthComparisonRecipe:
    moge_checkpoint: str
    vggt_checkpoint: str


@dataclass(frozen=True, slots=True)
class WaymoDepthComparisonSpec:
    input: WaymoDepthComparisonInput
    recipe: WaymoDepthComparisonRecipe


@dataclass(frozen=True, slots=True)
class WaymoDepthSelectionSpec:
    selection: str
    gate_id: str
    gate_spec: DepthGateSpec | DepthCandidateGateSpec


def parse_depth_comparison_spec(
    input_values: Mapping[str, object],
    parameter_values: Mapping[str, object],
    *,
    workflow_name: str,
    workflow_version: int,
    execution_preset: str,
) -> WaymoDepthComparisonSpec:
    """Parse the one MoGe→VGGT job without opening machine resources."""
    if (workflow_name, workflow_version) != ("waymo_depth_comparison", 1):
        raise ValueError("Waymo depth comparison requires version 1")
    if execution_preset != "inference_cp1":
        raise ValueError("Waymo depth comparison requires inference_cp1")
    reject_unknown_fields(input_values, _COMPARISON_INPUT_FIELDS)
    _require_mapping_fields(input_values, _COMPARISON_INPUT_FIELDS, "input")
    if input_values["reader"] != "waymo_v2" or input_values["dataset"] != "waymo":
        raise ValueError("Waymo depth comparison requires waymo_v2 dataset waymo")
    reject_unknown_fields(parameter_values, _COMPARISON_PARAMETER_FIELDS)
    _require_mapping_fields(
        parameter_values, _COMPARISON_PARAMETER_FIELDS, "parameters"
    )
    return WaymoDepthComparisonSpec(
        WaymoDepthComparisonInput(
            reader="waymo_v2",
            dataset="waymo",
            selection=_mapping_text(input_values, "selection"),
        ),
        WaymoDepthComparisonRecipe(
            moge_checkpoint=_mapping_text(parameter_values, "moge_checkpoint"),
            vggt_checkpoint=_mapping_text(parameter_values, "vggt_checkpoint"),
        ),
    )


def parse_depth_selection_spec(
    input_values: Mapping[str, object],
    parameter_values: Mapping[str, object],
    *,
    workflow_name: str,
    workflow_version: int,
    execution_preset: str,
) -> WaymoDepthSelectionSpec:
    """Parse one explicit v1 or v2 gate over eight candidate locators."""
    if workflow_name != "waymo_depth_selection" or workflow_version not in (1, 2):
        raise ValueError("Waymo depth selection requires version 1 or 2")
    if execution_preset != "cpu_test":
        raise ValueError("Waymo depth selection requires cpu_test")
    reject_unknown_fields(input_values, _SELECTION_INPUT_FIELDS)
    _require_mapping_fields(input_values, _SELECTION_INPUT_FIELDS, "input")
    parameter_fields = _GATE_COMMON_FIELDS | (
        {"minimum_heldout_count"} if workflow_version == 1 else set()
    )
    reject_unknown_fields(parameter_values, frozenset(parameter_fields))
    _require_mapping_fields(parameter_values, frozenset(parameter_fields), "parameters")
    numbers = {
        field: _mapping_number(parameter_values, field)
        for field in _GATE_COMMON_FIELDS
    }
    if workflow_version == 1:
        count = parameter_values["minimum_heldout_count"]
        if type(count) is not int or count <= 0:
            raise ValueError("minimum_heldout_count must be a positive integer")
        gate_spec: DepthGateSpec | DepthCandidateGateSpec = DepthGateSpec(
            minimum_heldout_count=count,
            **numbers,
        )
    else:
        gate_spec = DepthCandidateGateSpec(**numbers)
    return WaymoDepthSelectionSpec(
        selection=_mapping_text(input_values, "selection"),
        gate_id=f"waymo-depth-gate-v{workflow_version}",
        gate_spec=gate_spec,
    )


def load_depth_clip_selection(path: Path) -> WaymoClipKey:
    """Read exactly one selected clip for the model comparison."""
    document = load_yaml_mapping(path)
    reject_unknown_fields(document, frozenset({"schema_version", "clip"}))
    _require_mapping_fields(document, frozenset({"schema_version", "clip"}), "selection")
    if document["schema_version"] != 1:
        raise WaymoContractError("unsupported depth clip selection version")
    return clip_key_from_mapping(document["clip"])


def load_depth_report_selection(path: Path) -> tuple[str, ...]:
    """Read eight ordered candidate record locators without dereferencing them."""
    document = load_yaml_mapping(path)
    fields = frozenset({"schema_version", "candidate_results"})
    reject_unknown_fields(document, fields)
    _require_mapping_fields(document, fields, "selection")
    if document["schema_version"] != 1:
        raise WaymoContractError("unsupported depth report selection version")
    values = document["candidate_results"]
    if not isinstance(values, list) or len(values) != 8 or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise WaymoContractError("depth report selection requires eight locators")
    if len(set(values)) != 8:
        raise WaymoContractError("depth report locators must be distinct")
    return tuple(values)


def load_split_roles(
    path: Path,
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Прочитать прежний waymo-segment-split/v1, сохранив ordered роли.

    Полный прежний payload с описанием dataset/ordering/assignment и
    validation-ролями читается без конверсии. Этот потребитель использует
    только fit/dev/debug_subset из official training.
    """
    document = load_yaml_mapping(path)
    reject_unknown_fields(document, _FIELDS)
    if document.get("schema_version") != "waymo-segment-split/v1" or (
        document.get("split_id") != _SPLIT_ID
    ):
        raise WaymoContractError("keyset builder requires the historical Waymo split")
    splits = document.get("splits")
    if not isinstance(splits, dict):
        raise WaymoContractError("split config lacks splits")
    reject_unknown_fields(splits, _ROLES)
    fit, dev, debug = (_load_role(splits, role) for role in ("fit", "dev", "debug_subset"))
    if not set(fit).isdisjoint(dev):
        raise WaymoContractError("fit and dev segment IDs must be disjoint")
    if len(debug) < 2 or debug != fit[:len(debug)]:
        raise WaymoContractError("debug_subset must contain the first fit IDs, at least two")
    return _SPLIT_ID, fit, dev, debug


def _load_role(splits: dict[str, object], role: str) -> tuple[str, ...]:
    entry = splits.get(role)
    if not isinstance(entry, dict):
        raise WaymoContractError(f"split role {role!r} must be a mapping")
    reject_unknown_fields(entry, _ROLE_FIELDS)
    if entry.get("source_partition") != "training":
        raise WaymoContractError(f"split role {role!r} must use training")
    values = entry.get("segment_ids")
    if not isinstance(values, list) or not values or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise WaymoContractError(f"split role {role!r} has invalid segment IDs")
    if len(set(values)) != len(values):
        raise WaymoContractError(f"split role {role!r} repeats segment IDs")
    if "count" in entry and (
        type(entry["count"]) is not int or entry["count"] != len(values)
    ):
        raise WaymoContractError(f"split role {role!r} count differs from segment IDs")
    if "subset_of" in entry and (role != "debug_subset" or entry["subset_of"] != "fit"):
        raise WaymoContractError("only debug_subset may declare subset_of fit")
    return tuple(values)


def _require_mapping_fields(
    values: Mapping[str, object],
    fields: frozenset[str],
    name: str,
) -> None:
    missing = fields - values.keys()
    if missing:
        raise ValueError(f"missing {name} fields: {sorted(missing)}")


def _mapping_text(values: Mapping[str, object], field: str) -> str:
    value = values[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _mapping_number(values: Mapping[str, object], field: str) -> float:
    value = values[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be a non-negative number")
    if field == "minimum_valid_fraction" and result > 1.0:
        raise ValueError("minimum_valid_fraction must not exceed one")
    return result


__all__ = [
    "WaymoDepthComparisonInput",
    "WaymoDepthComparisonRecipe",
    "WaymoDepthComparisonSpec",
    "WaymoDepthSelectionSpec",
    "load_depth_clip_selection",
    "load_depth_report_selection",
    "load_split_roles",
    "parse_depth_comparison_spec",
    "parse_depth_selection_spec",
]
