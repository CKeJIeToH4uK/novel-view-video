"""Real generated-Parquet proof for the bounded physical Waymo v2 reader."""

from pathlib import Path
import subprocess
import sys

import numpy as np
from numpy.testing import assert_array_equal
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from novel_view.inputs.waymo.camera import WAYMO_CAMERA_ORDER
from novel_view.inputs.waymo.index import build_waymo_v2_window_index
from novel_view.inputs.waymo.lidar import WAYMO_LIDAR_ORDER
from novel_view.inputs.waymo.reader import WaymoV2FrameReader
from novel_view.inputs.waymo.types import WaymoContractError
from tests.support.waymo_v2 import SEGMENT, TIMESTAMPS, lidar_pair_values, write_reader_dataset


def test_generated_parquet_window_preserves_payload_and_canonical_order(tmp_path, monkeypatch):
    write_reader_dataset(tmp_path / "dataset")
    monkeypatch.chdir(tmp_path)
    original_batches = pq.ParquetFile.iter_batches

    def key_batches(file, *args, columns, **kwargs):
        assert columns and all(name.startswith("key.") for name in columns)
        return original_batches(file, *args, columns=columns, **kwargs)

    # Observe actual Arrow column reads: constructing the index must not decode payload.
    with monkeypatch.context() as index_reads:
        index_reads.setattr(pq.ParquetFile, "iter_batches", key_batches)
        reader = WaymoV2FrameReader(Path("dataset"), "validation", SEGMENT, 1, 2)
    bundles = list(reader)
    assert tuple(key.frame_timestamp_micros for key in reader.index.timeline) == TIMESTAMPS
    assert tuple(bundle.key.frame_timestamp_micros for bundle in bundles) == TIMESTAMPS[1:3]
    assert list(reader) == []
    context = bundles[0].context
    assert context.segment_id == SEGMENT
    assert WAYMO_CAMERA_ORDER == ("FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT")
    assert WAYMO_LIDAR_ORDER == ("TOP", "FRONT", "SIDE_LEFT", "SIDE_RIGHT", "REAR")
    assert tuple(c.name for c in context.camera_calibrations) == WAYMO_CAMERA_ORDER
    assert tuple(c.name for c in context.lidar_calibrations) == WAYMO_LIDAR_ORDER
    camera = context.camera_calibration("FRONT")
    lidar = context.lidar_calibration("TOP")
    assert_array_equal(camera.intrinsics, [[101, 0, 3], [0, 111, 4], [0, 0, 1]])
    assert_array_equal(camera.distortion_k1_k2_p1_p2_k3, [0.1, -0.02, 0.001, -0.002, 0.003])
    assert (camera.width, camera.height, camera.rolling_shutter_direction) == (8, 8, 4)
    assert_array_equal(lidar.beam_inclination_values_rad, [-0.2, 0.1])
    assert (lidar.beam_inclination_min_rad, lidar.beam_inclination_max_rad) == (-0.2, 0.1)
    for array in (camera.intrinsics, camera.distortion_k1_k2_p1_p2_k3,
                  camera.vehicle_from_waymo_camera, lidar.vehicle_from_lidar,
                  lidar.beam_inclination_values_rad):
        assert array.dtype == np.float64 and not array.flags.writeable
    for bundle in bundles:
        timestamp = bundle.key.frame_timestamp_micros
        assert tuple(c.name for c in bundle.cameras) == WAYMO_CAMERA_ORDER
        assert tuple(c.name for c in bundle.lidars) == WAYMO_LIDAR_ORDER
        assert bundle.camera("FRONT").name == camera.name
        assert bundle.lidar("TOP").name == lidar.name
        assert_array_equal(bundle.world_from_vehicle_for_frame[:3, 3],
                           np.arange(3) + timestamp / 100)
        for sensor_id, frame in enumerate(bundle.cameras, start=1):
            assert_array_equal(frame.global_linear_velocity_mps, [sensor_id, -sensor_id, 0.5 * sensor_id])
            assert_array_equal(frame.vehicle_angular_velocity_radps,
                               [0.01 * sensor_id, -0.02 * sensor_id, 0.03 * sensor_id])
            expected_offset = timestamp / 100 + sensor_id / 10
            assert_array_equal(frame.world_from_vehicle_at_camera_pose_timestamp[:3, 3],
                               np.arange(3) + expected_offset)
            assert frame.rgb.shape == (8, 8, 3) and frame.rgb.dtype == np.uint8
            assert not frame.rgb.flags.writeable
            assert frame.rgb[..., 0].mean() > frame.rgb[..., 2].mean()
            assert (frame.pose_timestamp_seconds, frame.camera_trigger_time_seconds,
                    frame.camera_readout_done_time_seconds, frame.shutter_seconds) == pytest.approx(
                (timestamp / 1e6 + 0.024, timestamp / 1e6, timestamp / 1e6 + 0.044, 0.01))
        for sensor_id, frame in enumerate(bundle.lidars, start=1):
            ranges = lidar_pair_values(timestamp, sensor_id, 4)
            projections = lidar_pair_values(timestamp, sensor_id, 6)
            for actual, expected_range, expected_projection in zip(
                (frame.return1, frame.return2), ranges, projections, strict=True,
            ):
                assert_array_equal(actual.range_image, expected_range)
                assert_array_equal(actual.camera_projection, expected_projection)
                assert actual.spatial_shape == (2, 3)
                for array in (actual.range_image, actual.camera_projection):
                    assert array.dtype == np.float32 and not array.flags.writeable
        top_pose = bundle.lidar("TOP").pixel_pose_vehicle_to_global_rpy_xyz
        assert_array_equal(top_pose, (np.arange(36, dtype=np.float32) + timestamp).reshape(2, 3, 6))
        assert top_pose.dtype == np.float32 and not top_pose.flags.writeable
        assert all(frame.pixel_pose_vehicle_to_global_rpy_xyz is None for frame in bundle.lidars[1:])


@pytest.mark.parametrize("damage", ("duplicate-sensor", "misaligned-timestamp", "missing-key"))
def test_index_rejects_ambiguous_external_grid(tmp_path, damage):
    write_reader_dataset(tmp_path)
    path = tmp_path / "validation/camera_image" / f"{SEGMENT}.parquet"
    table = pq.read_table(path)
    column = "key.frame_timestamp_micros" if damage == "misaligned-timestamp" else "key.camera_name"
    if damage == "missing-key":
        table = table.drop([column])
    else:
        values = table[column].to_pylist()
        values[4] = 999 if damage == "misaligned-timestamp" else 1
        table = table.set_column(table.schema.get_field_index(column), column,
                                 pa.array(values, type=table[column].type))
    pq.write_table(table, path, row_group_size=6)
    with pytest.raises(WaymoContractError):
        build_waymo_v2_window_index(tmp_path, "validation", SEGMENT, 0, 1)


def test_reader_rejects_descending_external_payload_timestamps(tmp_path):
    write_reader_dataset(tmp_path, camera_timestamp_order=(100, 300, 200, 400))
    reader = WaymoV2FrameReader(tmp_path, "validation", SEGMENT, 1, 2)
    with pytest.raises(WaymoContractError):
        next(reader)


def test_reader_import_is_arrow_opencv_and_model_free():
    source = Path(__file__).resolve().parents[2] / "src"
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import novel_view.inputs; "
        "import novel_view.inputs.waymo; import novel_view.inputs.waymo.index; "
        "import novel_view.inputs.waymo.decode; import novel_view.inputs.waymo.lidar_decode; "
        "import novel_view.inputs.waymo.reader; "
        "assert not {'pyarrow','cv2','torch'}.intersection(sys.modules); "
        "assert not any(name.endswith('_worker') for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-I", "-c", code, str(source)], check=True)
