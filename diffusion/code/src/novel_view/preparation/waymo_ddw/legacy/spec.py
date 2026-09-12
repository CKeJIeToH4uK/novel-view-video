"""External job fields for the four historical DDW measurement orders."""

from dataclasses import dataclass
from typing import Mapping

from novel_view.config.load import reject_unknown_fields
from novel_view.preparation.waymo_ddw.legacy.axis import DdwVariant


@dataclass(frozen=True, slots=True)
class LegacyDdwInput:
    dataset: str
    selection: str
    depth_selection: str


@dataclass(frozen=True, slots=True)
class LegacyDdwRecipe:
    moge_checkpoint: str
    axis: tuple[DdwVariant, ...]


@dataclass(frozen=True, slots=True)
class LegacyDdwSpec:
    input: LegacyDdwInput
    recipe: LegacyDdwRecipe


def parse_legacy_ddw_spec(
    input_values: Mapping[str, object],
    parameter_values: Mapping[str, object],
    *,
    workflow_name: str,
    workflow_version: int,
    execution_preset: str,
) -> LegacyDdwSpec:
    """Parse scientific forms without opening data, records or model files."""
    workflow = (workflow_name, workflow_version)
    if workflow not in (
        ("waymo_ddw_canary", 1),
        ("waymo_ddw_canary", 2),
        ("waymo_ddw_probe", 1),
        ("waymo_ddw_survey", 1),
    ):
        raise ValueError("unsupported historical DDW workflow")
    if execution_preset != "inference_cp1":
        raise ValueError("historical DDW measurement requires inference_cp1")
    reject_unknown_fields(
        input_values,
        frozenset({"reader", "dataset", "selection", "depth_selection"}),
    )
    if input_values["reader"] != "waymo_v2" or input_values["dataset"] != "waymo":
        raise ValueError("historical DDW requires waymo_v2 dataset waymo")
    reject_unknown_fields(parameter_values, frozenset({"moge_checkpoint", "axis"}))
    raw_axis = parameter_values["axis"]
    if not isinstance(raw_axis, list):
        raise ValueError("DDW axis must be an ordered list")
    axis = []
    for item in raw_axis:
        if not isinstance(item, dict):
            raise ValueError("each DDW variant must be a mapping")
        reject_unknown_fields(item, frozenset({"magnitude_m", "sign"}))
        magnitude, sign = item["magnitude_m"], item["sign"]
        if type(magnitude) is not int or type(sign) is not int:
            raise ValueError("legacy DDW magnitude and sign must be integers")
        axis.append(DdwVariant(magnitude, sign))
    first_magnitude = (
        3 if workflow_name == "waymo_ddw_probe"
        else 2 if workflow == ("waymo_ddw_canary", 2)
        else 1
    )
    expected = tuple(
        DdwVariant(magnitude, sign)
        for magnitude in range(first_magnitude, 5)
        for sign in (-1, 1)
    )
    if tuple(axis) != expected:
        raise ValueError("DDW axis disagrees with the historical workflow order")
    return LegacyDdwSpec(
        LegacyDdwInput(
            dataset="waymo",
            selection=_text(input_values["selection"], "selection"),
            depth_selection=_text(input_values["depth_selection"], "depth_selection"),
        ),
        LegacyDdwRecipe(
            moge_checkpoint=_text(parameter_values["moge_checkpoint"], "moge_checkpoint"),
            axis=tuple(axis),
        ),
    )


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


__all__ = [
    "LegacyDdwInput", "LegacyDdwRecipe", "LegacyDdwSpec", "parse_legacy_ddw_spec",
]
