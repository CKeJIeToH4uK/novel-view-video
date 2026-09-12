"""Real SQLite/CSV/RGB seam: metadata, temporal order and physical cameras."""

import csv
import pickle
import sqlite3
import sys
import types
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest

from novel_view.inputs.euvs.index import FramesIndex, FramesIndexError
from novel_view.inputs.euvs.pair import EuvsPairInputError, load_euvs_pair_input
from novel_view.inputs.euvs.rgb import rasterize_euvs_sequence, rasterize_euvs_source
from novel_view.inputs.nuplan.db import (
    NuPlanDataError, read_camera_calibrations, read_ego_poses,
)


def _official_pickle(name, values):
    names = (
        "nuplan", "nuplan.database", "nuplan.database.common",
        "nuplan.database.common.data_types",
    )
    modules = {name: types.ModuleType(name) for name in names}
    for parent, child in zip(names, names[1:]):
        setattr(modules[parent], child.rsplit(".", 1)[1], modules[child])
    wrapper = type(name, (list,), {"__module__": names[-1]})
    setattr(modules[names[-1]], name, wrapper)
    with patch.dict(sys.modules, modules):
        return pickle.dumps(wrapper(values), protocol=5)


def _write_index(root, rows, columns=None):
    with (root / "frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=columns or tuple(rows[0]), extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def dataset(tmp_path):
    # Deliberately non-lexical token order and independent image/pose clocks.
    rows = [dict(
        image_path=f"{token:016x}.png", db_path=database,
        location="32", logical_locations="32|33", traversal=traversal,
        timestamp_us=image_time, channel="CAM_F0", image_token=f"{token:016x}",
        ego_pose_token=f"{token + 256:016x}", camera_token="00000000000000aa",
        additional_csv_field="allowed",
    ) for token, database, traversal, image_time in (
        (3, "first.db", "2", 300), (1, "first.db", "2", 500),
        (2, "second.db", "2", 700), (4, "target.db", "6", 100), (5, "target.db", "6", 200),
    )]
    for database in ("first.db", "second.db", "target.db"):
        connection = sqlite3.connect(tmp_path / database)
        connection.executescript("""
            CREATE TABLE ego_pose (token BLOB PRIMARY KEY, timestamp INTEGER,
                x REAL, y REAL, z REAL, qw REAL, qx REAL, qy REAL, qz REAL, epsg INTEGER);
            CREATE TABLE camera (token BLOB PRIMARY KEY, channel TEXT, model TEXT,
                translation BLOB, rotation BLOB, intrinsic BLOB, distortion BLOB,
                width INTEGER, height INTEGER);
        """)
        for row, pose_time in reversed(list(zip(rows, (301, 499, 710, 95, 205)))):
            if row["db_path"] == database:
                token = int(row["image_token"], 16)
                connection.execute("INSERT INTO ego_pose VALUES (?,?,?,?,?,?,?,?,?,?)", (
                    bytes.fromhex(row["ego_pose_token"]), pose_time,
                    4_000_000.0 + 10 * token, 500_000.0, 3.0, 1.0, 0.0, 0.0, 0.0, 32611,
                ))
        intrinsic = ([[12., 0., 3.], [0., 10., 2.], [0., 0., 1.]] if database == "second.db"
                     else [[8., 0., 4.], [0., 9., 3.], [0., 0., 1.]])
        connection.execute("INSERT INTO camera VALUES (?,?,?,?,?,?,?,?,?)", (
            bytes.fromhex("00000000000000aa"), "CAM_F0", "pinhole",
            _official_pickle("Translation", [np.float64(1.0), 2.0, 3.0]),
            _official_pickle("Rotation", [1.0, 0.0, 0.0, 0.0]),
            _official_pickle("CameraIntrinsic", intrinsic), pickle.dumps([0.] * 5), 8, 6,
        ))
        connection.commit()
        connection.close()
    # Target PNGs deliberately do not exist yet: source rasterization must not open them.
    for row in rows[:3]:
        token = int(row["image_token"], 16)
        Image.fromarray(np.full((6, 8, 3), (10 * token, token, 200 - token), np.uint8)).save(
            tmp_path / row["image_path"],
        )
    _write_index(tmp_path, rows)
    arguments = dict(
        name="ordered-pair", location="33", channel="CAM_F0",
        source_traversal="2", target_traversal="6",
        source_image_tokens=tuple(row["image_token"] for row in rows[:3]),
        target_image_tokens=tuple(row["image_token"] for row in rows[3:]),
    )
    return rows, arguments


def test_sqlite_csv_rgb_preserves_order_clocks_and_measured_cameras(dataset, tmp_path):
    rows, arguments = dataset
    index = FramesIndex.load(tmp_path)
    pair = load_euvs_pair_input(frames_index=index, **arguments)
    source = rasterize_euvs_source(pair, (3, 4))

    assert pair.source.sequence_id == source.sequence_id == arguments["source_image_tokens"]
    assert pair.target.sequence_id == arguments["target_image_tokens"]
    assert (pair.source.traversal, pair.target.traversal) == ("2", "6")
    assert [frame.ref.timestamp_us for frame in pair.source.frames] == [300, 500, 700]
    assert [frame.raw_ego_pose.timestamp_us for frame in pair.source.frames] == [301, 499, 710]
    assert [frame.raw_ego_pose.timestamp_us for frame in pair.target.frames] == [95, 205]
    np.testing.assert_array_equal(
        pair.source.frames[0].geometry.camera_center_global,
        [4_000_031., 500_002., 6.],
    )
    assert pair.target.frames[0].geometry.epsg == 32611
    np.testing.assert_array_equal(
        [frame.raster.rgb[0, 0] for frame in source.frames],
        [[30, 3, 197], [10, 1, 199], [20, 2, 198]],
    )
    # Same camera token in two DBs must not reuse the first DB's intrinsics.
    np.testing.assert_array_equal([frame.raster.plan.output_intrinsics for frame in source.frames], [
        [[4., 0., 2.], [0., 4.5, 1.5], [0., 0., 1.]],
        [[4., 0., 2.], [0., 4.5, 1.5], [0., 0., 1.]],
        [[6., 0., 1.5], [0., 5., 1.], [0., 0., 1.]],
    ])
    for frame in source.frames:
        assert frame.raster.rgb.shape == (3, 4, 3) and frame.raster.plan.valid_mask.all()
    refs = tuple(frame.ref for frame in pair.source.frames)
    assert [pose.timestamp_us for pose in read_ego_poses(
        (refs[2], refs[0], refs[2]),
    )] == [710, 301, 710]
    assert read_camera_calibrations(refs)[2].intrinsics[0][0] == 12.
    for location in ("32", "33"):
        resolved = index.resolve(
            arguments["source_image_tokens"][::-1], location=location,
            traversal="2", channel="CAM_F0",
        )
        assert tuple(frame.image_token for frame in resolved) == source.sequence_id[::-1]
        assert resolved[0].db_path == tmp_path / "second.db"
    for row in rows[3:]:
        token = int(row["image_token"], 16)
        Image.fromarray(np.full((6, 8, 3), token, np.uint8)).save(tmp_path / row["image_path"])
    target = rasterize_euvs_sequence(pair.target, (3, 4))
    assert target.sequence_id == arguments["target_image_tokens"]
    np.testing.assert_array_equal([frame.raster.rgb[0, 0, 0] for frame in target.frames], [4, 5])


@pytest.mark.parametrize("role,clock", [
    (role, clock) for role in ("source", "target") for clock in ("image", "pose")
])
def test_independent_source_target_clocks_must_each_increase(dataset, tmp_path, role, clock):
    rows, arguments = dataset
    first, second = (0, 1) if role == "source" else (3, 4)
    row = rows[second]
    if clock == "image":
        row["timestamp_us"] = rows[first]["timestamp_us"] - (role == "target")
        _write_index(tmp_path, rows)
    else:
        with sqlite3.connect(tmp_path / row["db_path"]) as connection:
            connection.execute("UPDATE ego_pose SET timestamp=? WHERE token=?", (
                301 if role == "source" else 94, bytes.fromhex(row["ego_pose_token"]),
            ))
    with pytest.raises(EuvsPairInputError):
        load_euvs_pair_input(frames_index=FramesIndex.load(tmp_path), **arguments)


def test_pair_does_not_mix_coordinate_reference_systems(dataset, tmp_path):
    rows, arguments = dataset
    with sqlite3.connect(tmp_path / "target.db") as connection:
        connection.execute("UPDATE ego_pose SET epsg=32612")
    with pytest.raises(EuvsPairInputError):
        load_euvs_pair_input(frames_index=FramesIndex.load(tmp_path), **arguments)


@pytest.mark.parametrize("malformation", ("duplicate-token", "missing-column"))
def test_external_csv_is_unambiguous(dataset, tmp_path, malformation):
    rows, _ = dataset
    columns = tuple(name for name in rows[0] if name != "camera_token")
    _write_index(tmp_path, rows + [rows[0]] if malformation == "duplicate-token" else rows,
                 None if malformation == "duplicate-token" else columns)
    with pytest.raises(FramesIndexError):
        FramesIndex.load(tmp_path)


class _UnexpectedGlobal:
    def __reduce__(self):
        return print, ("pickle executed",)


def test_pickle_rejects_unrelated_global_without_executing_it(dataset, tmp_path, capsys):
    rows, arguments = dataset
    with sqlite3.connect(tmp_path / "first.db") as connection:
        connection.execute(
            "UPDATE camera SET translation=?",
            (pickle.dumps(_UnexpectedGlobal(), protocol=5),),
        )
    with pytest.raises(NuPlanDataError):
        load_euvs_pair_input(frames_index=FramesIndex.load(tmp_path), **arguments)
    assert capsys.readouterr().out == ""
