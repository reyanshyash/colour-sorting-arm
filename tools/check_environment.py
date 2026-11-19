"""Run with the Python interpreter used to build this ROS workspace."""

import importlib
import os
import sys

print("Python:", sys.executable)
print("ROS_DISTRO:", os.environ.get("ROS_DISTRO", "(not sourced)"))
failed = False
for module in ["rclpy", "mujoco", "numpy", "ament_index_python", "launch_ros", "cv2"]:
    try:
        mod = importlib.import_module(module)
        print(module, getattr(mod, "__version__", "available"))
    except ImportError as error:
        failed = True
        print(module, "MISSING:", error)
if os.environ.get("ROS_DISTRO") != "humble":
    failed = True
    print("Source your ROS 2 Humble environment before building or running.")
sys.exit(1 if failed else 0)
