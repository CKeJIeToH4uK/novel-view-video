"""Read the one old rejected DDW fit item still consumed by audit."""

from __future__ import annotations

from typing import Any


LEGACY_FIT_ITEM_SCHEMA = "gen3c-waymo-ddw-fit-clip/v1"


def parse_legacy_fit_item_document(document: object) -> dict[str, Any]:
    """Admit only rejected v1 evidence whose absent payload needs no old base."""
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != LEGACY_FIT_ITEM_SCHEMA
    ):
        raise ValueError("unsupported legacy DDW fit item schema")
    if document.get("status") != "rejected" or document.get("payload") is not None:
        raise ValueError("legacy fit reader supports only rejected audit items")
    return document


__all__ = [
    "LEGACY_FIT_ITEM_SCHEMA",
    "parse_legacy_fit_item_document",
]
