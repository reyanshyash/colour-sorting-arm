"""Image recognition, independent metric planes, tilt and occupancy regressions."""

from pathlib import Path
import numpy as np
import pytest
import cv2
from colour_sorting_arm.perception import (
    EdgeMap,
    camera_observation,
    intersect_plane,
    fit_plane,
    SoftwareCamera,
)
from colour_sorting_arm.simulation import Engine

MODEL = Path(__file__).resolve().parents[1] / "src/colour_sorting_arm/models/scene.xml"


def fixture(objects=None, source=None, tilt=2.0, offset=0.08):
    a = np.deg2rad(tilt)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    C = np.array([0.35 + offset, 0, 1.4])
    f = 480 / (2 * np.tan(np.deg2rad(50) / 2))
    cal = dict(
        position=C.tolist(), rotation=R.tolist(), K=[[f, 0, 319.5], [0, f, 239.5]]
    )
    bounds = [(0.18, 0.58, 0.12, 0.36, 0.20), (0.15, 0.55, -0.4, -0.16, 0.24)]

    def pixel(p):
        q = R.T @ (np.asarray(p) - C)
        return [319.5 + f * q[0] / -q[2], 239.5 - f * q[1] / -q[2]]

    polys = []
    for x0, x1, y0, y1, z in bounds:
        polys.append(
            [pixel(p) for p in [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]]]
        )
    objects = objects or dict(
        red=[0.25, 0.2, 0.25], green=[0.39, 0.2, 0.25], blue=[0.49, 0.3, 0.25]
    )
    packet = dict(
        calibration=cal,
        table_pixels=polys,
        block_pixels={k: pixel(v) for k, v in objects.items()},
    )
    m = EdgeMap(source)
    m.begin(packet)
    for i, (x0, x1, y0, y1, z) in enumerate(bounds):
        for x, y in [
            (x0 + 0.05, y0 + 0.04),
            (x1 - 0.05, y0 + 0.04),
            (x1 - 0.05, y1 - 0.04),
            (x0 + 0.05, y1 - 0.04),
        ]:
            m.add_probe(i, [[x, y, z]] * 5)
    return m, bounds


def test_camera_tilt_offset_and_different_table_heights_are_calibrated():
    m, bounds = fixture()
    objects, slots, tables = m.finish()
    assert m.probe_count == 8 and len(objects) == 3 and len(slots) > 1
    for t, (x0, x1, y0, y1, z) in zip(tables, bounds):
        np.testing.assert_allclose(t["low"], [x0, y0], atol=1e-7)
        np.testing.assert_allclose(t["high"], [x1, y1], atol=1e-7)
        np.testing.assert_allclose(t["polygon"][:, 2], z, atol=1e-7)
        assert t["area_m2"] == pytest.approx((x1 - x0) * (y1 - y0), abs=1e-7)
    assert all(abs(s[2] - 0.265) < 1e-7 for s in slots)
    assert all(
        cv2.pointPolygonTest(
            tables[1]["polygon"][:, :2].astype(np.float32), tuple(s[:2]), True
        )
        > 0.042
        for s in slots
    )


def test_rescan_preserves_source_and_rejects_occupied_drop_cells():
    m, _ = fixture()
    m.finish()
    moved = dict(
        red=[0.25, -0.28, 0.29], green=[0.4, -0.28, 0.29], blue=[0.49, 0.3, 0.25]
    )
    second, _ = fixture(moved, m.source_center)
    objects, slots, _ = second.finish()
    assert set(objects) == {"blue"}
    assert all(np.linalg.norm(s[:2] - np.array([0.25, -0.28])) >= 0.075 for s in slots)


def test_bad_ir_geometry_is_rejected():
    with pytest.raises(ValueError, match="Four"):
        fit_plane([[0, 0, 0.2]] * 3)
    with pytest.raises(ValueError, match="spread"):
        fit_plane([[0, 0, 0.2], [0.1, 0, 0.2], [0.2, 0, 0.2], [0.3, 0, 0.2]])
    with pytest.raises(ValueError):
        fit_plane([[0.2, 0, 0.2], [0.4, 0, 0.2], [0.4, 0.2, 0.25], [0.2, 0.2, 0.2]])


def test_opencv_finds_both_tabletops_in_actual_geometry_render():
    e = Engine(MODEL)
    camera = SoftwareCamera(e.model)
    rgb = camera.render(e.data)
    p = camera_observation(
        rgb,
        e.data.cam_xpos[camera.cam],
        e.data.cam_xmat[camera.cam],
        e.model.cam_fovy[camera.cam],
    )
    assert len(p["table_pixels"]) == 2
    assert set(p["block_pixels"]) == {"red", "green", "blue"}
    assert rgb.shape == (240, 320, 3)


def test_bright_neutral_tabletops_keep_complete_outlines():
    # Native lighting turns warm material nearly white, making hue undefined.
    rgb = np.full((240, 320, 3), 70, np.uint8)
    rgb[25:90, 100:240] = [255, 255, 255]
    rgb[150:215, 100:240] = [245, 245, 240]
    rgb[40:50, 120:130] = [255, 0, 0]
    rgb[55:65, 160:170] = [0, 255, 0]
    rgb[70:80, 200:210] = [0, 0, 255]
    p = camera_observation(rgb, [0.35, 0, 1.4], np.eye(3), 50)
    assert len(p["table_pixels"]) == 2
    assert set(p["block_pixels"]) == {"red", "green", "blue"}
    assert all(
        cv2.contourArea(np.asarray(q, np.float32)) > 8500 for q in p["table_pixels"]
    )
