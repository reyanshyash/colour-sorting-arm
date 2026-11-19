"""Run the SAME task controller as ROS against physical dynamics and CPU RGB.
Only transport is replaced with deterministic in-process message delivery.
No body coordinates are fed into perception or the controller.
"""

from pathlib import Path
from types import SimpleNamespace as Message
import sys, json, argparse
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/colour_sorting_arm"))
from colour_sorting_arm.simulation import Engine
from colour_sorting_arm.controller import ArmController, JOINTS
from colour_sorting_arm.perception import SoftwareCamera, camera_observation


class Publisher:
    def __init__(self, callback):
        self.callback = callback

    def publish(self, msg):
        self.callback(msg)


def run(
    seed=7,
    colours=("green",),
    max_time=180.0,
    camera_offset=0.0,
    camera_tilt=0.0,
    mapping_only=False,
    destination_height=None,
):
    engine = Engine(ROOT / "src/colour_sorting_arm/models/scene.xml", seed=seed)
    cid = engine.model.camera("overhead").id
    engine.model.cam_pos[cid, 0] += camera_offset
    angle = np.deg2rad(camera_tilt)
    engine.model.cam_quat[cid] = [np.cos(angle / 2), 0, np.sin(angle / 2), 0]
    if destination_height is not None:
        engine.model.geom("destination_table").pos[2] = destination_height - 0.015
        support = engine.model.geom("destination_table_support")
        support.pos[2] = (destination_height - 0.03) / 2
        support.size[2] = (destination_height - 0.03) / 2
    mujoco.mj_forward(engine.model, engine.data)
    camera = SoftwareCamera(engine.model, height=240, width=320, camera="overhead")
    command = engine.data.ctrl.copy()
    acks = []
    reports = []

    def receive_command(msg):
        nonlocal command
        command = np.array(msg.data)

    def receive_grasp(msg):
        request = json.loads(msg.data)
        result = engine.grasp(request["request"])
        acks.append(json.dumps(dict(request_id=request["request_id"], result=result)))

    def report(msg):
        reports.append(dict(time=float(engine.data.time), message=msg.data))
        print(round(engine.data.time, 2), msg.data, flush=True)

    core = ArmController(
        engine.model,
        Publisher(receive_command),
        Publisher(receive_grasp),
        Publisher(report),
        lambda: float(engine.data.time),
        lambda s: None,
    )
    for colour in colours:
        core.request(Message(data=colour))
    next_image = -1.0
    completed = 0
    while engine.data.time < max_time:
        core.state(
            Message(
                name=JOINTS,
                position=[float(engine.data.joint(n).qpos[0]) for n in JOINTS],
            )
        )
        pending = acks[:]
        acks.clear()
        for ack in pending:
            core.status(Message(data=ack))
        if engine.data.time >= next_image:
            rgb = camera.render(engine.data)
            h, w = rgb.shape[:2]
            core.image(
                Message(
                    encoding="rgb8", data=rgb.tobytes(), height=h, width=w, step=w * 3
                )
            )
            cid = camera.cam
            distance = float(engine.data.sensor("wrist_ir").data[0])
            if distance > 0:
                distance += engine.rng.normal(0, 0.0015)
            site = engine.model.site("ir_origin").id
            hit = (
                engine.data.site_xpos[site]
                + distance * engine.data.site_xmat[site].reshape(3, 3)[:, 2]
                if distance > 0
                else None
            )
            packet = camera_observation(
                rgb,
                engine.data.cam_xpos[cid],
                engine.data.cam_xmat[cid],
                engine.model.cam_fovy[cid],
                hit,
                engine.data.time,
            )
            core.observation(Message(data=json.dumps(packet)))
            next_image = engine.data.time + 0.2
        core.tick()
        for _ in range(round(0.02 / engine.model.opt.timestep)):
            engine.step(command)
        completed = sum(
            r["message"].startswith("placement sequence complete") for r in reports
        )
        if (
            (mapping_only and not core.mapping)
            or (not mapping_only and completed == len(colours))
            or any(
                "aborted:" in r["message"]
                or "Mapping failed:" in r["message"]
                or "block not visible" in r["message"]
                for r in reports
            )
        ):
            break
    placement_errors = (
        {
            c: float(
                np.linalg.norm(
                    engine.data.xpos[engine.model.body(c + "_block").id, :2]
                    - core.slots[i][:2]
                )
            )
            for i, c in enumerate(colours)
        }
        if core.slots
        else {}
    )
    physical_success = (
        (not core.mapping and core.mapper.probe_count == 8)
        if mapping_only
        else completed == len(colours)
        and all(
            placement_errors[c] < 0.04
            and abs(
                engine.data.xpos[engine.model.body(c + "_block").id, 2]
                - core.slots[i][2]
            )
            < 0.02
            for i, c in enumerate(colours)
        )
    )
    scores = []
    for table in core.mapper.tables:
        center = table["polygon"][:, :2].mean(0)
        names = ("source_table", "destination_table")
        name = min(
            names, key=lambda n: np.linalg.norm(engine.model.geom(n).pos[:2] - center)
        )
        g = engine.model.geom(name)
        z = g.pos[2] + g.size[2]
        truth = np.array(
            [
                [g.pos[0] + sx * g.size[0], g.pos[1] + sy * g.size[1], z]
                for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            ]
        )
        error = max(min(np.linalg.norm(p - t) for t in truth) for p in table["polygon"])
        scores.append(
            dict(
                table=name,
                max_corner_error_mm=float(error * 1000),
                area_error_percent=float(
                    100 * (table["area_m2"] / (4 * g.size[0] * g.size[1]) - 1)
                ),
            )
        )
    object_errors = (
        {
            c: float(
                np.linalg.norm(p - engine.data.xpos[engine.model.body(c + "_block").id])
                * 1000
            )
            for c, p in core.objects.items()
        }
        if mapping_only
        else {}
    )
    result = dict(
        mapping_scores=scores,
        object_errors_mm=object_errors,
        camera_offset=camera_offset,
        camera_tilt=camera_tilt,
        surface_map=core.mapper.serialized(),
        physical_success=physical_success,
        placement_errors_m=placement_errors,
        seed=seed,
        colours=list(colours),
        completed=completed,
        sim_time=float(engine.data.time),
        reports=reports,
        mapped_objects={c: v.tolist() for c, v in core.objects.items()},
        finger_positions=[
            float(engine.data.joint(n).qpos[0]) for n in ("finger_left", "finger_right")
        ],
        final_positions={
            c: engine.data.xpos[engine.model.body(c + "_block").id].tolist()
            for c in colours
        },
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--colours", nargs="+", default=["green"])
    p.add_argument("--output", default="controller_validation.json")
    a = p.parse_args()
    result = run(a.seed, a.colours)
    out = ROOT / "results" / a.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["physical_success"] else 1)
