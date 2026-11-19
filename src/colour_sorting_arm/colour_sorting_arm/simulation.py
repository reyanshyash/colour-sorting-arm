"""MuJoCo dynamics, physical grasp sensing and ROS simulator adapter."""

from collections import deque
import numpy as np
import mujoco
from .controller import READY, JOINTS


class Engine:
    def __init__(self, path, seed=7):
        import xml.etree.ElementTree as ET

        rng = np.random.default_rng(seed)
        scene = ET.parse(path)
        # Compile after resizing so collision bounds and rendering use the same boxes.
        for name in ("source_table", "destination_table"):
            geom = scene.find(f".//geom[@name='{name}']")
            half = np.fromstring(geom.get("size"), sep=" ")
            half[:2] *= rng.uniform(0.94, 1.05, 2)
            geom.set("size", " ".join(map(str, half)))
            support = scene.find(f".//geom[@name='{name}_support']")
            support.set("size", " ".join(map(str, [*(half[:2] - 0.02), 0.085])))
        self.model = mujoco.MjModel.from_xml_string(
            ET.tostring(scene.getroot(), encoding="unicode")
        )
        self.data = mujoco.MjData(self.model)
        source = self.model.geom("source_table")
        points = []
        for c in ("red", "green", "blue"):
            for _ in range(1000):
                xy = source.pos[:2] + rng.uniform(-1, 1, 2) * (source.size[:2] - 0.045)
                if all(np.linalg.norm(xy - p) > 0.08 for p in points):
                    break
            else:
                raise RuntimeError("Could not sample separated blocks")
            points.append(xy)
            j = self.model.body(c + "_block").jntadr[0]
            a = self.model.jnt_qposadr[j]
            self.data.qpos[a : a + 3] = [*xy, 0.225]
            angle = rng.uniform(-0.3, 0.3)
            self.data.qpos[a + 3 : a + 7] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
        for n, v in zip(JOINTS, READY):
            self.data.qpos[self.model.joint(n).qposadr[0]] = v
        for n in ("finger_left", "finger_right"):
            self.data.qpos[self.model.joint(n).qposadr[0]] = 0.03
        self.data.ctrl[:] = np.r_[READY, 0.03, 0.03]
        self.drive = self.data.ctrl.copy()
        self.play = self.drive.copy()
        self.rng = rng
        self.contact_history = {
            c: deque(maxlen=round(0.2 / self.model.opt.timestep))
            for c in ("red", "green", "blue")
        }
        mujoco.mj_forward(self.model, self.data)

    def safe_command(self):
        return np.r_[
            [self.data.qpos[self.model.joint(n).qposadr[0]] for n in JOINTS],
            self.data.ctrl[6:],
        ]

    def step(self, target):
        # Play operator approximates 0.15-degree output-side backlash, then 25ms lag.
        gap = np.r_[np.full(6, 0.0026), 0.0, 0.0]
        self.play = np.clip(
            self.play, np.asarray(target) - gap, np.asarray(target) + gap
        )
        delta = (self.play - self.drive) * min(1, self.model.opt.timestep / 0.025)
        delta[6:] = np.clip(
            delta[6:], -0.02 * self.model.opt.timestep, 0.02 * self.model.opt.timestep
        )
        self.drive += delta
        self.data.ctrl[:] = self.drive
        mujoco.mj_step(self.model, self.data)
        for colour in self.contact_history:
            self.contact_history[colour].append(self.contact_mask(colour))

    def contact_mask(self, colour):
        bid = self.model.body(colour + "_block").id
        fingers = [
            int(self.model.joint(n).bodyid[0]) for n in ("finger_left", "finger_right")
        ]
        mask = 0
        for contact in self.data.contact:
            a, b = [
                int(self.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)
            ]
            for i, f in enumerate(fingers):
                if (a == bid and b == f) or (b == bid and a == f):
                    mask |= 1 << i
        return mask

    def grasp(self, request):
        if request == "release":
            return "released"
        if request not in self.contact_history:
            return "invalid_grasp"
        history = self.contact_history[request]
        # Tactile confirmation integrates 200ms; a single contact sample is unreliable.
        touched = all(
            sum(bool(v & (1 << i)) for v in history)
            >= round(0.016 / self.model.opt.timestep)
            for i in range(2)
        )
        return "grasped" if touched else "grasp_rejected_no_bilateral_contact"


import time
import json
from pathlib import Path
from contextlib import nullcontext

PACKAGE = "colour_sorting_arm"


class EpisodeLog:
    """Evaluation only. Scene geometry is never published to the planner."""

    def __init__(self, engine, seed, directory, log):
        import uuid

        self.engine = engine
        self.seed = seed
        self.id = uuid.uuid4().hex
        self.path = Path(directory).expanduser() / "episodes.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.attempt = None
        self.transferred = set()
        self.still_since = None
        self.record("start")
        self.log(f"Episode seed {seed}. Results file {self.path}")

    def record(self, event, **fields):
        with self.path.open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        episode=self.id,
                        seed=self.seed,
                        time_s=self.engine.data.time,
                        event=event,
                        **fields,
                    ),
                    allow_nan=False,
                )
                + "\n"
            )

    def corners(self, role):
        geom = self.engine.model.geom(role + "_table")
        return (
            geom.pos[:2]
            + np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]]) * geom.size[:2]
        )

    def local_mm(self, world):
        return ((world[:2] - self.corners("destination")[0]) * [1, -1] * 1000).tolist()

    def mapped(self, packet):
        for role, t in zip(packet["roles"], packet["tables"]):
            corners = self.corners(role)
            area = float(4 * np.prod(self.engine.model.geom(role + "_table").size[:2]))
            estimate = t["area_m2"]
            from itertools import permutations

            measured = np.array(t["polygon"])[:, :2]
            worst = 1000 * min(
                np.linalg.norm(measured - list(order), axis=1).max()
                for order in permutations(corners)
            )
            percent = 100 * (estimate - area) / area
            self.record(
                "map",
                table=role,
                true_m2=area,
                scanned_m2=estimate,
                error_pct=percent,
                max_corner_mm=worst,
            )
            self.log(
                f"{role}: true {area:.5f} m², scanned {estimate:.5f} m², error {percent:+.2f}%, worst corner {worst:.1f} mm"
            )

    def handle(self, event, request):
        fields = {k: v for k, v in request.items() if k not in ("episode", "event")}
        if event == "finish":
            self.finish()
            if len(self.transferred) == 3:
                self.record("complete")
                return True
            return False
        if event == "begin":
            self.attempt = fields.copy()
            self.still_since = None
        elif event in ("failed", "reset"):
            fields.update(self.attempt or {})
            self.attempt = None
        self.record(event, **fields)
        return event == "reset"

    def sample(self):
        if not self.attempt or "gripper_mm" not in self.attempt:
            return
        d = self.engine.data
        block = d.joint(
            self.engine.model.body(self.attempt["colour"] + "_block").jntadr[0]
        )
        slow = (
            np.linalg.norm(block.qvel[:3]) < 0.005
            and np.linalg.norm(block.qvel[3:]) < 0.1
        )
        if not slow:
            self.still_since = None
        elif self.still_since is None:
            self.still_since = d.time

    def grasp_check(self, request, result):
        if not self.attempt:
            return
        colour = self.attempt["colour"]
        mask = self.engine.contact_mask(colour)
        signature = (request["request_id"], result, mask)
        if signature == getattr(self, "last_grasp_sample", None):
            return
        self.last_grasp_sample = signature
        model, data = self.engine.model, self.engine.data
        fingers = [
            int(model.joint(name).bodyid[0]) for name in ("finger_left", "finger_right")
        ]
        table = model.geom("source_table").id
        touching = set()
        for contact in data.contact:
            pair = (contact.geom1, contact.geom2)
            if table in pair:
                touching.add(
                    model.geom_bodyid[pair[1] if pair[0] == table else pair[0]]
                )
        self.record(
            "grasp_check",
            colour=colour,
            stage=request.get("stage"),
            result=result,
            block_contact=[bool(mask & bit) for bit in (1, 2)],
            table_contact=[finger in touching for finger in fingers],
        )

    def release(self):
        if self.attempt and "gripper_mm" not in self.attempt:
            self.attempt["gripper_mm"] = self.local_mm(
                self.engine.data.site("tool").xpos
            )
            block = self.engine.data.body(self.attempt["colour"] + "_block")
            self.attempt["block_before_open_mm"] = self.local_mm(block.xpos)

    def finish(self):
        attempt = self.attempt
        if attempt is None:
            return
        colour = attempt["colour"]
        model, data = self.engine.model, self.engine.data
        body = model.body(colour + "_block")
        final = self.local_mm(data.xpos[body.id])
        delta = np.array(final) - attempt["requested_mm"]
        block_geom = int(body.geomadr[0])
        table = model.geom("destination_table")
        supported = any(
            {c.geom1, c.geom2} == {block_geom, table.id} for c in data.contact
        )
        settled = self.still_since is not None and data.time - self.still_since >= 0.25
        # Project the complete oriented box footprint onto the true tabletop.
        half_extent = (
            np.abs(data.geom_xmat[block_geom].reshape(3, 3)[:2])
            @ model.geom_size[block_geom]
        )
        inside = bool(
            np.all(
                np.abs(data.geom_xpos[block_geom, :2] - table.pos[:2]) + half_extent
                <= table.size[:2]
            )
        )
        transferred = bool(supported and settled and inside)
        self.record(
            "placement",
            **attempt,
            actual_mm=final,
            delta_mm=delta.tolist(),
            distance_mm=float(np.linalg.norm(delta)),
            settled=bool(settled),
            supported=supported,
            inside=inside,
            transferred=transferred,
        )
        self.log(
            f"{colour} | requested {attempt['requested_mm']} mm | gripper before opening {np.round(attempt.get('gripper_mm', []), 1)} mm | block before opening {np.round(attempt.get('block_before_open_mm', []), 1)} mm | actual {np.round(final, 1)} mm | dx {delta[0]:+.1f}, dy {delta[1]:+.1f}, distance {np.linalg.norm(delta):.1f} mm | transferred={transferred}"
        )
        if transferred:
            self.transferred.add(colour)
        self.attempt = None
        self.still_since = None


def make_simulator():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState, Image
    from std_msgs.msg import Float64MultiArray, String
    from rosgraph_msgs.msg import Clock
    from ament_index_python.packages import get_package_share_directory

    class Simulator(Node):
        def __init__(self):
            super().__init__("simulator", namespace=PACKAGE)
            self.declare_parameter("viewer", False)
            self.declare_parameter("camera_backend", "opengl")
            self.camera_backend = self.get_parameter("camera_backend").value
            if self.camera_backend not in ("opengl", "software"):
                raise ValueError("camera_backend must be opengl or software")
            path = Path(get_package_share_directory(PACKAGE)) / "models" / "scene.xml"
            self.declare_parameter("seed", -1)
            self.declare_parameter("results_dir", str(Path.cwd() / "results"))
            self.path = path
            self.seed = self.get_parameter("seed").value
            if self.seed < 0:
                self.seed = int(np.random.SeedSequence().generate_state(1)[0])
            self.reset_pending = False
            self.engine = Engine(path, seed=self.seed)
            self.session = self.new_log()
            self.model, self.data = self.engine.model, self.engine.data
            self.command = self.data.ctrl.copy()
            self.last_command = time.monotonic()
            self.watchdog_command = None
            self.state_pub = self.create_publisher(JointState, "joint_states", 10)
            self.clock_pub = self.create_publisher(Clock, "/clock", 10)
            self.status_pub = self.create_publisher(String, "status", 10)
            self.create_subscription(
                Float64MultiArray, "command", self.receive_command, 10
            )
            self.create_subscription(String, "grasp", self.receive_grasp, 10)
            self.observation_pub = self.create_publisher(String, "observations", 10)
            self.image_pub = self.create_publisher(
                Image, "camera/rgb", qos_profile_sensor_data
            )
            self.renderer = None
            self.map_polygons = []
            self.create_subscription(
                String, "surface_map", self.receive_surface_map, 10
            )
            self.last_image = -1.0
            self.render_failed = False
            self.joints = [
                (i, mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i))
                for i in range(self.model.njnt)
                if self.model.jnt_type[i]
                in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            ]
            self.create_timer(0.01, self.tick)

        def new_log(self):
            return EpisodeLog(
                self.engine,
                self.seed,
                self.get_parameter("results_dir").value,
                self.get_logger().info,
            )

        def reset(self):
            stamp = self.data.time
            if self.renderer is not None:
                self.renderer.close()
            self.renderer = None
            self.seed += 1
            self.engine = Engine(self.path, self.seed)
            self.model, self.data = self.engine.model, self.engine.data
            self.data.time = stamp
            self.command = self.data.ctrl.copy()
            self.watchdog_command = None
            self.last_command = time.monotonic()
            self.map_polygons = []
            self.last_image = -1.0
            self.render_failed = False
            self.session = self.new_log()
            self.reset_pending = False

        def receive_surface_map(self, msg):
            packet = json.loads(msg.data)
            if packet.get("episode") not in (None, self.session.id):
                return
            self.map_polygons = [
                np.asarray(t["polygon"], float) for t in packet["tables"]
            ]
            if self.map_polygons:
                self.session.mapped(packet)

        def receive_command(self, msg):
            values = np.asarray(msg.data, dtype=float)
            if values.shape != (self.model.nu,) or not np.isfinite(values).all():
                self.get_logger().warning(
                    f"Expected {self.model.nu} finite actuator commands"
                )
                return
            limited = self.model.actuator_ctrllimited.astype(bool)
            values[limited] = np.clip(
                values[limited],
                self.model.actuator_ctrlrange[limited, 0],
                self.model.actuator_ctrlrange[limited, 1],
            )
            self.command = values
            self.watchdog_command = None
            self.last_command = time.monotonic()

        def receive_grasp(self, msg):
            try:
                request = json.loads(msg.data)
            except (ValueError, TypeError):
                request = msg.data
            if isinstance(request, dict):
                if (
                    request.get("episode") not in (None, self.session.id)
                    or self.reset_pending
                ):
                    return
                event = request.get("event")
                if event:
                    self.reset_pending = self.session.handle(event, request)
                    return
                if request.get("request") == "release":
                    self.session.release()
                result = self.engine.grasp(request.get("request", ""))
                if request.get("request") in self.engine.contact_history:
                    self.session.grasp_check(request, result)
                reply = json.dumps(
                    dict(request_id=request.get("request_id"), result=result)
                )
            else:
                reply = self.engine.grasp(request)
            self.status_pub.publish(String(data=reply))

        def tick(self):
            # A timed-out position-controlled arm holds; force/velocity plants stop driving.
            command = self.command.copy()
            if time.monotonic() - self.last_command > 0.5:
                if self.watchdog_command is None:
                    self.watchdog_command = self.engine.safe_command()
                command = self.watchdog_command
            self.data.ctrl[:] = command
            for _ in range(round(0.01 / self.model.opt.timestep)):
                self.engine.step(command)
            self.session.sample()
            sec = int(self.data.time)
            clock = Clock()
            clock.clock.sec, clock.clock.nanosec = (
                sec,
                int((self.data.time - sec) * 1e9),
            )
            self.clock_pub.publish(clock)
            msg = JointState()
            msg.header.stamp = clock.clock
            msg.name = [name for _, name in self.joints]
            msg.position = [
                float(self.data.qpos[self.model.jnt_qposadr[i]]) for i, _ in self.joints
            ]
            msg.velocity = [
                float(self.data.qvel[self.model.jnt_dofadr[i]]) for i, _ in self.joints
            ]
            self.state_pub.publish(msg)
            if (
                PACKAGE == "colour_sorting_arm"
                and self.data.time - self.last_image >= 0.2
            ):
                self.publish_image(clock.clock)
                self.last_image = self.data.time

        def publish_image(self, stamp):
            if self.render_failed:
                return
            try:
                if self.camera_backend == "software":
                    if self.renderer is None:
                        from .perception import SoftwareCamera

                        self.renderer = SoftwareCamera(self.model, camera="overhead")
                        self.get_logger().info(
                            "Using idealized CPU ray-rendered RGB without lighting or textures"
                        )
                    rgb = self.renderer.render(self.data)
                else:
                    if self.renderer is None:
                        self.renderer = mujoco.Renderer(
                            self.model, height=480, width=640
                        )
                    self.renderer.update_scene(self.data, camera="overhead")
                    rgb = self.renderer.render()
                msg = Image()
                msg.header.stamp, msg.header.frame_id = stamp, "overview_camera"
                msg.height, msg.width = rgb.shape[:2]
                msg.encoding = "rgb8"
                msg.step, msg.is_bigendian = msg.width * 3, 0
                msg.data = rgb.tobytes()
                self.image_pub.publish(msg)
                from .perception import camera_observation

                cam = self.model.camera("overhead").id
                distance = float(self.data.sensor("wrist_ir").data[0])
                hit = None
                if distance > 0:
                    distance += self.engine.rng.normal(0, 0.0015)
                    site = self.model.site("ir_origin").id
                    hit = (
                        self.data.site_xpos[site]
                        + distance * self.data.site_xmat[site].reshape(3, 3)[:, 2]
                    )
                packet = camera_observation(
                    rgb,
                    self.data.cam_xpos[cam],
                    self.data.cam_xmat[cam],
                    self.model.cam_fovy[cam],
                    hit,
                    self.data.time,
                )
                packet["episode"] = self.session.id
                self.observation_pub.publish(String(data=json.dumps(packet)))
            except Exception as error:
                self.render_failed = True
                self.get_logger().error(
                    f"Camera rendering failed: {error}. Check OpenGL/display access."
                )

        def destroy_node(self):
            if self.renderer is not None:
                self.renderer.close()
            super().destroy_node()

    return Simulator()


def main(args=None):
    import rclpy

    rclpy.init(args=args)
    node = make_simulator()
    try:
        while rclpy.ok():
            if node.get_parameter("viewer").value:
                import mujoco.viewer

                context = mujoco.viewer.launch_passive(node.model, node.data)
            else:
                context = nullcontext(None)
            with context as viewer:
                while (
                    rclpy.ok()
                    and not node.reset_pending
                    and (viewer is None or viewer.is_running())
                ):
                    rclpy.spin_once(node, timeout_sec=0.005)
                    if viewer is not None:
                        with viewer.lock():
                            viewer.user_scn.ngeom = 0
                            for index, polygon in enumerate(node.map_polygons):
                                for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
                                    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                                    rgba = (
                                        np.array([0.1, 1.0, 0.25, 1.0])
                                        if index == 0
                                        else np.array([1.0, 0.65, 0.1, 1.0])
                                    )
                                    mujoco.mjv_initGeom(
                                        geom,
                                        mujoco.mjtGeom.mjGEOM_CAPSULE,
                                        np.zeros(3),
                                        np.zeros(3),
                                        np.eye(3).ravel(),
                                        rgba,
                                    )
                                    mujoco.mjv_connector(
                                        geom,
                                        mujoco.mjtGeom.mjGEOM_CAPSULE,
                                        0.003,
                                        a + [0, 0, 0.004],
                                        b + [0, 0, 0.004],
                                    )
                                    viewer.user_scn.ngeom += 1
                        viewer.sync()
            if not node.reset_pending:
                break
            node.reset()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
