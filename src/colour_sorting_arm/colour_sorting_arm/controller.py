"""Inverse kinematics, queued task state machine and ROS controller adapter."""

import numpy as np
import mujoco

JOINTS = ["base_yaw", "shoulder", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
READY = np.array([0.0, -0.9, 1.3, 0.0, -0.4, 0.0])


def solve_ik(model, target, seed):
    data = mujoco.MjData(model)
    ids = [model.joint(name).id for name in JOINTS]
    qadr = model.jnt_qposadr[ids]
    dadr = model.jnt_dofadr[ids]
    q = np.array(seed, dtype=float).copy()
    site = model.site("tool").id
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(250):
        data.qpos[qadr] = q
        mujoco.mj_forward(model, data)
        rotation = data.site_xmat[site].reshape(3, 3)
        # Small rotation that aligns each tool axis with its world counterpart.
        orientation = 0.5 * sum(
            np.cross(rotation[:, i], np.eye(3)[:, i]) for i in range(3)
        )
        position = np.asarray(target) - data.site_xpos[site]
        if (
            np.linalg.norm(position) < 0.003
            and np.linalg.norm(rotation - np.eye(3)) < 0.04
        ):
            return q
        mujoco.mj_jacSite(model, data, jp, jr, site)
        jacobian = np.vstack([jp[:, dadr], 0.3 * jr[:, dadr]])
        error = np.r_[position, 0.3 * orientation]
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 0.002 * np.eye(6), error
        )
        q += np.clip(delta, -0.12, 0.12)
        q = np.clip(q, model.jnt_range[ids, 0] + 0.01, model.jnt_range[ids, 1] - 0.01)
    raise ValueError(f"IK did not converge for {target}; change target or seed")


from pathlib import Path
import json
from collections import deque
from types import SimpleNamespace
from .perception import EdgeMap, COLOURS, table_frame
import cv2

# Transport-neutral envelopes; the ROS adapter converts them to ROS messages.
String = SimpleNamespace
Float64MultiArray = SimpleNamespace


class ArmController:
    def __init__(
        self, model, command_pub, grasp_pub, task_pub, clock, log=print, map_pub=None
    ):
        self.episode = None
        self.waiting_reset = False
        self.model = model
        self.command_pub = command_pub
        self.grasp_pub = grasp_pub
        self.task_pub = task_pub
        self._clock = clock
        self._log = log
        self.map_pub = map_pub
        self.ir_samples = []
        self.last_ir_stamp = -float("inf")
        self.q = READY.copy()
        self.command = np.r_[READY, 0.03, 0.03]
        self.rgb = None
        self.image_time = -100.0
        self.pending = None
        self.steps = []
        self.requests = deque()
        self.source_center = None
        self.mapper = EdgeMap()
        self.mapping = True
        self.latest_packet = None
        self.objects = {}
        self.slots = []
        self.slot_index = 0
        self.build_scan()
        self.active = None
        self.ack = None
        self.check_id = 0
        self.last_check_sent = -float("inf")
        self.last_stamp = None
        self.report(
            "ready: publish red, green or blue to /colour_sorting_arm/pick_colour"
        )

    def build_scan(self):
        # No raster tracing: detect outlines, then plan only eight range positions.
        self.mapper = EdgeMap(self.source_center)
        self.mapping = True
        self.slot_index = 0
        self.latest_packet = None
        self.steps = [("probe_move", READY.copy(), 0.03), ("camera_detect", None, 0.03)]
        if self.map_pub is not None:
            self.map_pub.publish(
                String(
                    data=json.dumps(dict(method="camera_edges_sparse_ir", tables=[]))
                )
            )

    def prepare_probes(self, packet):
        self.mapper.begin(packet)
        steps = []
        seed = self.q.copy()
        for table in range(2):
            candidates = []
            for point in self.mapper.candidates(table):
                try:
                    candidates.append((point, solve_ik(self.model, point, seed)))
                except ValueError:
                    continue
            selected = self.mapper.select_spread(candidates)
            while selected:
                index = int(np.argmin([np.max(abs(q - seed)) for _, q in selected]))
                point, q = selected.pop(index)
                steps.extend([("probe_move", q, 0.03), ("ir_sample", table, 0.03)])
                seed = q
        self.steps = steps + [("map_done", None, 0.03)]
        self.report(
            "Tabletop edges detected; taking four spaced IR measurements per table"
        )

    def mapping_failure(self, text):
        self.event("mapping_failed", reason=text)
        self.report("Mapping failed: " + text + "; send rescan")
        self.steps.clear()
        self.requests.clear()
        self.command[:6] = self.q

    def now(self):
        return self._clock()

    def report(self, text):
        self._log(text)
        self.task_pub.publish(String(data=text))

    def state(self, msg):
        values = dict(zip(msg.name, msg.position))
        if all(n in values for n in JOINTS):
            self.q = np.array([values[n] for n in JOINTS])
            self.last_stamp = self.now()

    def image(self, msg):
        if msg.encoding != "rgb8":
            return
        rows = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
        self.rgb = rows[:, : msg.width * 3].reshape(msg.height, msg.width, 3).copy()
        self.image_time = self.now()

    def status(self, msg):
        try:
            reply = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        if not isinstance(reply, dict) or reply.get("request_id") != self.check_id:
            return
        self.ack = reply.get("result")

    def observation(self, msg):
        packet = json.loads(msg.data)
        episode = packet.get("episode")
        if episode is not None and episode != self.episode:
            if self.episode is not None:
                self.active = None
                self.pending = None
                self.requests.clear()
                self.objects.clear()
                self.source_center = None
                self.__dict__.pop("current_colour", None)
                self.command = np.r_[READY, 0.03, 0.03]
                self.build_scan()
            self.episode = episode
            self.waiting_reset = False
        self.latest_packet = packet
        if packet and self.active and self.active[0] == "ir_sample":
            stamp = packet.get("stamp", -1)
            hit = packet.get("ir_hit")
            if stamp > max(self.started, self.last_ir_stamp) and hit is not None:
                self.ir_samples.append(hit)
                self.last_ir_stamp = stamp

    def request(self, msg):
        colour = msg.data.strip().lower()
        if colour == "next":
            self.waiting_reset = True
            self.event("reset")
            return
        if self.waiting_reset:
            self.report("Waiting for next episode")
            return
        if colour == "standard":
            for command in ("green 80 80", "red 185 145", "blue 280 75"):
                self.request(String(data=command))
            return
        if colour in ("scan", "rescan"):
            if self.active or (self.steps and not self.mapping):
                self.report("Wait until current task finishes before rescanning")
            else:
                self.build_scan()
                self.report("Rescanning surfaces and occupancy")
        elif (
            not colour
            or colour.split()[0] not in COLOURS
            or len(colour.split()) not in (1, 3)
        ):
            self.report("invalid colour: choose red, green or blue")
        elif len(self.requests) >= 20:
            self.report("request queue full")
        else:
            self.requests.append(colour)
            self.report("queued " + colour)

    def event(self, kind, **fields):
        if getattr(self, "episode", None) is not None:
            packet = dict(episode=self.episode, event=kind, **fields)
            self.grasp_pub.publish(String(data=json.dumps(packet)))

    def plan(self, command):
        parts = command.split()
        colour = parts[0]

        def reject(reason):
            self.event("rejected", command=command, reason=reason)
            self.report(reason)

        target = self.objects.get(colour)
        if target is None:
            reject(colour + " block not visible on source table")
            return
        table = self.mapper.destination
        origin, axes = table_frame(table["polygon"])
        occupied = self.mapper.occupied
        if len(parts) == 3:
            try:
                requested = np.array(parts[1:], float)
                if not np.isfinite(requested).all():
                    raise ValueError()
            except ValueError:
                reject("Enter finite X Y coordinates in millimetres")
                return
            xy = origin + (requested / 1000) @ axes
            clearance = cv2.pointPolygonTest(
                table["polygon"][:, :2].astype(np.float32), tuple(xy), True
            )
            if clearance < 0.035 or any(
                np.linalg.norm(xy - p) < 0.07 for p in occupied
            ):
                reject(
                    "Target outside scanned safe area or overlaps an occupied position"
                )
                return
            n = table["normal"]
            destination = np.r_[xy, -(n[:2] @ xy + table["offset"]) / n[2] + 0.025]
        else:
            destination = next(
                (
                    p
                    for p in self.slots
                    if all(np.linalg.norm(p[:2] - q) >= 0.07 for q in occupied)
                ),
                None,
            )
            if destination is None:
                reject("No free observed destination slots")
                return
            requested = 1000 * axes @ (destination[:2] - origin)
        self.destination_xy = destination[:2].copy()
        # Keep the lower jaw edges clear of the source while they slide inward.
        grasp_point = target + [0, 0, 0.008]
        points = [
            target + [0, 0, 0.20],
            grasp_point,
            target + [0, 0, 0.20],
            destination + [0, 0, 0.20],
            destination + [0, 0, 0.008],
            destination + [0, 0, 0.20],
        ]
        seed, solutions = self.q.copy(), []
        try:
            for point in points:
                seed = solve_ik(self.model, point, seed)
                solutions.append(seed.copy())
        except ValueError as error:
            reject(str(error))
            return
        self.event("begin", colour=colour, requested_mm=requested.tolist())
        self.grasp_stage = "after_close"
        self.current_colour = colour
        a, b, c, d, e, f = solutions
        self.steps = [
            ("move", a, 0.03),
            ("move", b, 0.03),
            ("close", b, 0.0),
            ("check_grasp", colour, 0.0),
            ("move", c, 0.0),
            ("check_grasp", colour, 0.0),
            ("move", d, 0.0),
            ("move", e, 0.0),
            ("release", None, 0.03),
            ("move", f, 0.03),
        ]
        self.report("executing " + colour + " pick and place")

    def tick(self):
        if getattr(self, "waiting_reset", False):
            return
        now = self.now()
        if (
            not self.mapping
            and not self.pending
            and not self.active
            and not self.steps
            and self.requests
        ):
            self.pending = (self.requests.popleft(), now)
        if self.pending:
            colour, requested = self.pending
            if (
                self.rgb is not None
                and now - self.image_time < 0.8
                and self.last_stamp is not None
                and now - self.last_stamp < 0.5
            ):
                self.pending = None
                self.plan(colour)
            elif now - requested > 5:
                self.pending = None
                self.event("rejected", command=colour, reason="stale sensors")
                self.report("request timed out: no fresh camera or joint state")
        if self.active is None and self.steps:
            self.active = self.steps.pop(0)
            self.started, self.start_q = now, self.q.copy()
            self.ack = None
            if self.active[0] in ("close", "check_grasp", "release"):
                self.report(
                    f"{getattr(self, 'current_colour', 'arm')}: {self.active[0]}"
                )
            if self.active[0] == "ir_sample":
                self.ir_samples = []
                self.last_ir_stamp = now
            if self.active[0] in ("move", "close", "probe_move"):
                delta = float(np.max(abs(self.active[1] - self.start_q)))
                self.motion_duration = max(
                    2.5 if self.active[0] == "move" else 0.7,
                    1.875 * delta / 0.5,
                    np.sqrt(5.774 * delta / 1.2),
                )
            if self.active[0] in ("check_grasp", "release"):
                self.check_id += 1
                self.last_check_sent = -float("inf")

        if self.active:
            kind, value, opening = self.active
            elapsed = now - self.started
            done = False
            if kind in ("move", "close", "probe_move"):
                duration = (
                    max(2.5, self.motion_duration)
                    if kind == "close"
                    else self.motion_duration
                )
                s = np.clip(elapsed / duration, 0.0, 1.0)
                blend = s * s * s * (10 - 15 * s + 6 * s * s)
                self.command[:6] = self.start_q + blend * (value - self.start_q)
                self.command[6:] = opening
                # Hold the final target before beginning finger closure or lifting.
                done = (
                    elapsed >= duration + 0.3 and np.max(np.abs(self.q - value)) < 0.015
                )
            elif kind == "camera_detect":
                if (
                    self.latest_packet
                    and self.latest_packet.get("stamp", -1) > self.started
                    and len(self.latest_packet.get("table_pixels", [])) == 2
                    and self.last_stamp is not None
                ):
                    try:
                        self.prepare_probes(self.latest_packet)
                    except ValueError as error:
                        self.mapping_failure(str(error))
                    done = True
                elif elapsed > 8:
                    count = len((self.latest_packet or {}).get("table_pixels", []))
                    self.mapping_failure(
                        f"Camera sees {count} complete tabletop outlines; expected two. Check visibility, contrast and camera calibration"
                    )
                    done = True
            elif kind == "ir_sample":
                if len(self.ir_samples) >= 5 and elapsed > 0.4:
                    try:
                        self.mapper.add_probe(value, self.ir_samples)
                    except ValueError as error:
                        self.mapping_failure(str(error))
                    done = True
                elif elapsed > 2.5:
                    self.mapping_failure(
                        "No five fresh wrist-IR returns at the probe position"
                    )
                    done = True
            elif kind == "map_done":
                try:
                    self.objects, self.slots, tables = self.mapper.finish()
                    self.source_center = self.mapper.source_center.copy()
                    self.mapping = False
                    if self.map_pub is not None:
                        self.map_pub.publish(
                            String(
                                data=json.dumps(
                                    dict(self.mapper.serialized(), episode=self.episode)
                                )
                            )
                        )
                    polygon = self.mapper.destination["polygon"]
                    origin, axes = table_frame(polygon)
                    dimensions = (
                        np.max((polygon[:, :2] - origin) @ axes.T, axis=0) * 1000
                    )
                    self.report(
                        f"Destination measured size: X {dimensions[0]:.0f} mm, Y {dimensions[1]:.0f} mm; allow 35 mm edge clearance"
                    )
                    areas = [round(t["area_m2"], 4) for t in tables]
                    self.report(
                        f"Camera-edge map ready: two tables, areas {areas} m2, {len(self.objects)} blocks, {len(self.slots)} free slots, {self.mapper.probe_count} IR positions"
                    )
                except ValueError as error:
                    self.mapping_failure(str(error))
                done = True
            else:
                expected = "grasped" if kind == "check_grasp" else "released"
                if now - self.last_check_sent >= 0.1 and not (self.ack == expected):
                    request = value if kind == "check_grasp" else "release"
                    self.grasp_pub.publish(
                        String(
                            data=json.dumps(
                                dict(
                                    request_id=self.check_id,
                                    request=request,
                                    stage=getattr(self, "grasp_stage", "after_close"),
                                    episode=getattr(self, "episode", None),
                                )
                            )
                        )
                    )
                    self.last_check_sent = now
                if kind == "release" and self.ack == expected:
                    self.command[6:] = 0.03
                done = self.ack == expected and elapsed > (
                    2.0 if kind == "release" else 0.3
                )
                if kind == "check_grasp" and done:
                    self.grasp_stage = "after_lift"
                if kind == "check_grasp" and elapsed > 2.0 and not done:
                    self.event("failed", reason=str(self.ack or "no_grasp_ack"))
                    self.report(
                        "aborted: "
                        + str(self.ack or "no_grasp_ack")
                        + " after waiting for physical contact"
                    )
                    self.steps.clear()
                    self.requests.clear()
                    self.pending = None
                    self.command[:6] = self.q
                    if hasattr(self, "current_colour"):
                        del self.current_colour
                    self.active = None
            if elapsed > (
                getattr(self, "motion_duration", 7.0) + 5.0
                if kind in ("move", "close", "probe_move")
                else 12.0
            ):
                self.event("failed", reason=kind + " timeout")
                self.report(f"aborted: {kind} timeout; holding current pose")
                self.steps.clear()
                self.requests.clear()
                self.pending = None
                if hasattr(self, "current_colour"):
                    del self.current_colour
                self.command[:6] = self.q
                self.active = None
            elif done:
                self.active = None
                if not self.steps:
                    if not self.mapping and hasattr(self, "current_colour"):
                        self.mapper.occupied.append(self.destination_xy)
                        self.event("finish")
                        self.slot_index += 1
                        self.objects.pop(self.current_colour, None)
                        del self.current_colour
                        self.report(
                            "placement sequence complete; next queued request can run"
                        )
        self.command_pub.publish(Float64MultiArray(data=self.command.tolist()))


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState, Image
    from std_msgs.msg import String as ROSString, Float64MultiArray as ROSArray
    from ament_index_python.packages import get_package_share_directory

    class Publisher:
        def __init__(self, pub, msgtype):
            self.pub, self.msgtype = pub, msgtype

        def publish(self, msg):
            self.pub.publish(self.msgtype(data=msg.data))

    class Controller(Node):
        def __init__(self):
            super().__init__("controller", namespace="colour_sorting_arm")
            model = mujoco.MjModel.from_xml_path(
                str(
                    Path(get_package_share_directory("colour_sorting_arm"))
                    / "models/scene.xml"
                )
            )
            command = Publisher(
                self.create_publisher(ROSArray, "command", 10), ROSArray
            )
            grasp = Publisher(self.create_publisher(ROSString, "grasp", 10), ROSString)
            status = Publisher(
                self.create_publisher(ROSString, "task_status", 10), ROSString
            )
            surface = Publisher(
                self.create_publisher(ROSString, "surface_map", 10), ROSString
            )
            self.core = ArmController(
                model,
                command,
                grasp,
                status,
                lambda: self.get_clock().now().nanoseconds * 1e-9,
                self.get_logger().info,
                map_pub=surface,
            )
            self.create_subscription(ROSString, "pick_colour", self.core.request, 10)
            self.create_subscription(
                ROSString, "observations", self.core.observation, 10
            )
            self.create_subscription(ROSString, "status", self.core.status, 10)
            self.create_subscription(JointState, "joint_states", self.core.state, 10)
            self.create_subscription(
                Image, "camera/rgb", self.core.image, qos_profile_sensor_data
            )
            self.create_timer(0.02, self.core.tick)

    rclpy.init(args=args)
    node = Controller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def target_console():
    import rclpy
    from std_msgs.msg import String as ROSString

    rclpy.init()
    node = rclpy.create_node("target_console", namespace="colour_sorting_arm")
    publisher = node.create_publisher(ROSString, "pick_colour", 10)
    try:
        print("Waiting for the arm controller...", flush=True)
        while rclpy.ok() and not publisher.get_subscription_count():
            rclpy.spin_once(node, timeout_sec=0.2)
        print(
            "Enter COLOUR X_mm Y_mm, a colour alone, standard, rescan, next, or quit."
        )
        while rclpy.ok():
            line = input("arm> ").strip().lower()
            if line in ("quit", "exit"):
                break
            if line:
                publisher.publish(ROSString(data=line))
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
