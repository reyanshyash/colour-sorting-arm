from glob import glob
from setuptools import setup

package_name = "colour_sorting_arm"
setup(
    name=package_name,
    version="0.4.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/models", glob("models/*.xml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "mujoco==3.3.7",
        "opencv-python>=4.6,<4.12",
    ],
    zip_safe=False,
    maintainer="Reyansh Bhardwaj",
    maintainer_email="maintainer@example.com",
    description="Six-axis colour sorting arm: MuJoCo and ROS 2 Humble learning project",
    license="MIT",
    entry_points={
        "console_scripts": [
            "targets = " + package_name + ".controller:target_console",
            "simulator = " + package_name + ".simulation:main",
            "controller = " + package_name + ".controller:main",
        ]
    },
)
