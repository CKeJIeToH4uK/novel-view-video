"""Saved raw arrays become metric EUVS geometry only after source binding."""

import json
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from novel_view.generation.euvs.source_views import (
    EuvsSourceView,
    EuvsSourceViewError,
    _require_source_match,
    bind_source_view,
    build_vggt_source_request,
    resolve_euvs_vggt_geometry,
    unique_source_views,
)
from novel_view.inputs.euvs.spec import EuvsFrameSelection, EuvsPairSelection, EuvsSelection
from novel_view.models.vggt.record import (
    PREDICTION_FILES,
    load_saved_vggt_omega_prediction,
    save_vggt_omega_prediction,
)
from novel_view.models.vggt.request import VggtOmegaRawPrediction
from novel_view.models.vggt.spec import VggtOmegaInputMode, build_vggt_omega_input_plan


def selection(name, tokens):
    return EuvsPairSelection(
        name,
        (),
        "boston",
        "CAM_F0",
        EuvsFrameSelection("source", tokens),
        EuvsFrameSelection("target", (name,)),
    )


def test_first_pair_ordered_source_identity_and_stream():
    first = selection("first", ("a", "b"))
    reverse = selection("reverse", ("b", "a"))
    assert unique_source_views(
        EuvsSelection((first, selection("duplicate", ("a", "b")), reverse))
    ) == (first, reverse)
    with pytest.raises(EuvsSourceViewError):
        unique_source_views(EuvsSelection((first, selection("collision", ("a", "c")))))
    frames = [NS(raster=NS(rgb=np.full((4, 6, 3), value, np.uint8))) for value in [7, 2, 9]]
    view = NS(geometry_input=NS(source=NS(image_size_hw=(4, 6), frames=frames)))
    request = build_vggt_source_request(view, VggtOmegaInputMode.BALANCED_512)
    assert request.frame_count == 3 and request.input_plan.source_size_hw == (4, 6)
    assert [int(rgb[0, 0, 0]) for rgb in request.rgb_frames] == [7, 2, 9]


@pytest.mark.parametrize("legacy", [False, True])
def test_record_values_and_metric_binding_keep_current_or_attested_identity(tmp_path, legacy):
    tokens, size = ("a", "b", "c", "d"), (4, 6)
    plan = build_vggt_omega_input_plan(size, VggtOmegaInputMode.BALANCED_512)
    cameras = np.tile(np.eye(4), (4, 1, 1))
    cameras[:, 0, 3] = -np.arange(4)
    depth = np.broadcast_to(
        np.arange(1, 5, dtype=np.float32)[:, None, None], (4, *plan.model_size_hw)
    ).copy()
    intrinsics = np.tile(np.array([[3.0, 0.25, 1.5], [0, 4, 1], [0, 0, 1]], np.float32), (4, 1, 1))
    intrinsics[:, 0, 0] += np.arange(4)
    raw = VggtOmegaRawPrediction(
        plan,
        depth,
        depth + 10,
        np.arange(36, dtype=np.float32).reshape(4, 9),
        cameras[:, :3].astype(np.float32),
        intrinsics,
    )
    directory = tmp_path / "prediction"
    save_vggt_omega_prediction(raw, tokens, size, directory)
    assert {p.name for p in directory.iterdir()} == {*PREDICTION_FILES.values(), "summary.json"}
    assert PREDICTION_FILES["model_w2c"] == "predicted_w2c.npy"
    assert PREDICTION_FILES["model_intrinsics"] == "predicted_intrinsics.npy"
    if legacy:
        path = directory / "summary.json"
        summary = json.loads(path.read_text())
        del summary["source_image_tokens"]
        path.write_text(json.dumps(summary))
    saved = load_saved_vggt_omega_prediction(directory)
    assert saved.source_tokens == (None if legacy else tokens)
    for name in PREDICTION_FILES:
        np.testing.assert_array_equal(getattr(saved.prediction, name), getattr(raw, name))
    provenance = "legacy-attested" if legacy else "ordered-source-tokens"
    assert saved.provenance == provenance
    valid = np.ones((4, *size), np.bool_)
    valid[1, 2, 3] = False
    source = NS(
        sequence_id=tokens,
        image_size_hw=size,
        frames=[NS(raster=NS(plan=NS(valid_mask=mask))) for mask in valid],
    )
    measured = cameras.copy()
    measured[:, 0, 3] *= 2
    geometry = NS(source=source, reference_to_camera=measured)
    view = EuvsSourceView(selection("pair", tokens), None, geometry)
    if legacy:
        with pytest.raises(EuvsSourceViewError):
            bind_source_view(view, directory)
        resolved = resolve_euvs_vggt_geometry(geometry, directory, expected_provenance=provenance)
    else:
        resolved = bind_source_view(view, directory)
        reordered = NS(source=NS(sequence_id=("a", "c", "b", "d"), image_size_hw=size))
        with pytest.raises(EuvsSourceViewError):
            resolve_euvs_vggt_geometry(reordered, directory)
    assert resolved.provenance == provenance
    assert resolved.alignment.scale_m_per_model_unit == pytest.approx(2.0)
    assert resolved.posed_depth.sequence_id == tokens
    expected = np.broadcast_to(np.arange(2, 9, 2)[:, None, None], (4, *size)).copy()
    expected[~valid] = 0
    np.testing.assert_array_equal(resolved.posed_depth.depth_z_m, expected)
    with pytest.raises(EuvsSourceViewError):
        _require_source_match(replace(saved, source_frame_count=3), tokens, size)
