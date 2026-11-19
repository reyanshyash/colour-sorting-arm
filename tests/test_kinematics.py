from pathlib import Path
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from colour_sorting_arm.controller import solve_ik, JOINTS, READY

MODEL = Path(__file__).resolve().parents[1] / "src/colour_sorting_arm/models/scene.xml"


@pytest.mark.parametrize(
    "target",
    [[x, 0.23, z] for x in [0.28, 0.40, 0.48] for z in [0.225, 0.425]]
    + [[0.35, -0.28, 0.425], [0.35, -0.28, 0.225]],
)
def test_ik_reaches_pick_and_place_waypoints(target):
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    result = solve_ik(model, target, READY)
    data = mujoco.MjData(model)
    for name, value in zip(JOINTS, result):
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.site("tool").xpos, target, atol=0.004)
    np.testing.assert_allclose(
        data.site("tool").xmat.reshape(3, 3), np.eye(3), atol=0.04
    )
