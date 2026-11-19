"""Task handshake regressions: transient contacts, stale replies and true failures."""

from collections import deque
from types import SimpleNamespace as Message
import json
import numpy as np
from colour_sorting_arm.controller import ArmController


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg.data)


def check_fixture():
    c = ArmController.__new__(ArmController)
    clock = [0.1]
    c._clock = lambda: clock[0]
    c._log = lambda text: None
    c.command_pub = Publisher()
    c.grasp_pub = Publisher()
    c.task_pub = Publisher()
    c.mapping = False
    c.pending = None
    c.requests = deque(["blue"])
    c.q = np.zeros(6)
    c.command = np.zeros(8)
    c.steps = [("move", np.ones(6), 0.0)]
    c.active = ("check_grasp", "green", 0.0)
    c.started = 0.0
    c.ack = None
    c.check_id = 4
    c.last_check_sent = -float("inf")
    c.current_colour = "green"
    return c, clock


def test_temporary_rejection_is_retried_and_stale_reply_is_ignored():
    c, clock = check_fixture()
    c.tick()
    c.status(
        Message(
            data=json.dumps(
                dict(request_id=4, result="grasp_rejected_no_bilateral_contact")
            )
        )
    )
    clock[0] = 0.5
    c.tick()
    assert c.active[0] == "check_grasp" and len(c.grasp_pub.messages) == 2
    c.status(Message(data=json.dumps(dict(request_id=3, result="grasped"))))
    assert c.ack == "grasp_rejected_no_bilateral_contact"
    c.status(Message(data=json.dumps(dict(request_id=4, result="grasped"))))
    clock[0] = 0.6
    c.tick()
    assert c.active is None and c.steps[0][0] == "move"
    assert not any(str(m).startswith("aborted:") for m in c.task_pub.messages)


def test_missing_physical_grip_still_aborts_instead_of_bypassing_contact():
    c, clock = check_fixture()
    c.ack = "grasp_rejected_no_bilateral_contact"
    clock[0] = 2.1
    c.tick()
    assert c.active is None and not c.steps and not c.requests
    assert any("aborted:" in m for m in c.task_pub.messages)


def test_startup_plans_camera_capture_instead_of_a_raster_sweep():
    from pathlib import Path
    import mujoco

    model = mujoco.MjModel.from_xml_path(
        str(
            Path(__file__).resolve().parents[1]
            / "src/colour_sorting_arm/models/scene.xml"
        )
    )
    c = ArmController(
        model, Publisher(), Publisher(), Publisher(), lambda: 0.0, lambda text: None
    )
    assert [s[0] for s in c.steps] == ["probe_move", "camera_detect"]
