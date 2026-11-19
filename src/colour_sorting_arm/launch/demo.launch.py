"""Launch independent ROS controller and fixed-step MuJoCo simulator."""

import sys
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    interpreter = "mjpython" if sys.platform == "darwin" else sys.executable
    return LaunchDescription(
        [
            DeclareLaunchArgument("seed", default_value="-1"),
            DeclareLaunchArgument("viewer", default_value="false"),
            DeclareLaunchArgument("camera_backend", default_value="opengl"),
            ExecuteProcess(
                cmd=[
                    interpreter,
                    "-m",
                    "colour_sorting_arm.simulation",
                    "--ros-args",
                    "-p",
                    ["seed:=", LaunchConfiguration("seed")],
                    "-p",
                    ["viewer:=", LaunchConfiguration("viewer")],
                    "-p",
                    ["camera_backend:=", LaunchConfiguration("camera_backend")],
                ],
                output="screen",
            ),
            Node(
                package="colour_sorting_arm",
                executable="controller",
                output="screen",
                parameters=[{"use_sim_time": True}],
            ),
        ]
    )
