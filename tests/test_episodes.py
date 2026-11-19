"""A biased scan must remain visible in the independent placement score."""

import json
import numpy as np
import mujoco
from pathlib import Path
from colour_sorting_arm.simulation import Engine, EpisodeLog

MODEL = Path(__file__).resolve().parents[1] / "src/colour_sorting_arm/models/scene.xml"


def test_true_corner_and_signed_error(tmp_path):
    engine = Engine(MODEL, seed=7)
    log = EpisodeLog(engine, 7, tmp_path, lambda _: None)
    table = engine.model.geom("destination_table")
    corner = table.pos[:2] + table.size[:2] * [-1, 1]
    joint = engine.model.body("green_block").jntadr[0]
    engine.data.joint(joint).qpos[:2] = corner + [0.126, -0.076]
    mujoco.mj_forward(engine.model, engine.data)
    log.handle("begin", dict(colour="green", requested_mm=[120, 80]))
    log.release()
    engine.data.joint(joint).qpos[0] += 0.002
    mujoco.mj_forward(engine.model, engine.data)
    log.release()  # A repeated release request must preserve the first measurement.
    log.finish()
    result = json.loads(log.path.read_text().splitlines()[-1])
    np.testing.assert_allclose(result["block_before_open_mm"], [126, 76])
    np.testing.assert_allclose(result["delta_mm"], [8, -4])
    np.testing.assert_allclose(result["distance_mm"], np.hypot(8, 4))
    assert not result[
        "transferred"
    ]  # A final position alone cannot prove settled contact.


def test_random_layouts_replay():
    def layout(seed):
        engine = Engine(MODEL, seed)
        return np.r_[
            engine.model.geom("destination_table").size[:2],
            engine.data.body("green_block").xpos[:2],
        ]

    np.testing.assert_array_equal(layout(7), layout(7))
    assert np.all(layout(7) != layout(8))
