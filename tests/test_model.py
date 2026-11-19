"""Requires MuJoCo; checks model validity, settling and actuator dimensions."""

from pathlib import Path
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from colour_sorting_arm.simulation import Engine

MODEL = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "colour_sorting_arm"
    / "models"
    / "scene.xml"
)


def test_model_settles_without_nonfinite_state():
    engine = Engine(MODEL)
    for _ in range(round(1 / engine.model.opt.timestep)):
        mujoco.mj_step(engine.model, engine.data)
    assert np.isfinite(engine.data.qpos).all()
    assert engine.data.time == pytest.approx(1.0)
    assert len(engine.safe_command()) == engine.model.nu
