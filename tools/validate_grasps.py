"""Dynamics-only acceptance test. Ground-truth fixtures are TEST INPUTS.
This deliberately does not claim RGB perception or ROS end-to-end validation.
"""

from pathlib import Path
import json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/colour_sorting_arm"))
from colour_sorting_arm.simulation import Engine
from colour_sorting_arm.controller import solve_ik, READY, JOINTS


def run(seed=7, engine=None, objects=None, slots=None):
    engine = (
        engine
        if engine is not None
        else Engine(ROOT / "src/colour_sorting_arm/models/scene.xml", seed=seed)
    )
    q = np.array([engine.data.joint(n).qpos[0] for n in JOINTS])
    rows = []

    def move(point, opening, seconds=2.5):
        nonlocal q
        goal = solve_ik(engine.model, point, q)
        start = np.array([engine.data.joint(n).qpos[0] for n in JOINTS])
        for k in range(int((seconds + 1.0) / engine.model.opt.timestep)):
            s = min(1, k * engine.model.opt.timestep / seconds)
            blend = s * s * s * (10 - 15 * s + 6 * s * s)
            engine.step(np.r_[start + blend * (goal - start), opening, opening])
        q = goal

    for i, colour in enumerate(("red", "green", "blue")):
        bid = engine.model.body(colour + "_block").id
        target = (
            objects[colour].copy()
            if objects is not None
            else engine.data.xpos[bid].copy()
        )
        destination = (
            slots[i].copy()
            if slots is not None
            else np.array([0.25 + 0.10 * i, -0.28, 0.225])
        )
        try:
            move(target + [0, 0, 0.20], 0.03)
            move(target, 0.03)
            for _ in range(int(2.6 / engine.model.opt.timestep)):
                engine.step(np.r_[q, 0.0, 0.0])
            close = engine.grasp(colour)
            move(target + [0, 0, 0.20], 0.0)
            lift = engine.grasp(colour)
            height = float(engine.data.xpos[bid, 2])
            move(destination + [0, 0, 0.20], 0.0)
            move(destination, 0.0)
            for _ in range(int(2.4 / engine.model.opt.timestep)):
                engine.step(np.r_[q, 0.03, 0.03])
            move(destination + [0, 0, 0.20], 0.03)
            final = engine.data.xpos[bid].copy()
            error = float(np.linalg.norm(final[:2] - destination[:2]))
            rows.append(
                dict(
                    colour=colour,
                    close=close,
                    lift=lift,
                    lift_height=height,
                    final=final.tolist(),
                    xy_error=error,
                    success=bool(
                        close == "grasped"
                        and lift == "grasped"
                        and error < 0.065
                        and 0.21 < final[2] < 0.27
                    ),
                )
            )
        except ValueError as error:
            rows.append(dict(colour=colour, success=False, error=str(error)))
    return rows


if __name__ == "__main__":
    rows = run()
    out = ROOT / "results/physical_grasps.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
    raise SystemExit(0 if all(r["success"] for r in rows) else 1)
