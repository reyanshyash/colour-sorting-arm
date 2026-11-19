# Camera-guided pick and place

A six-axis MuJoCo arm maps two tables, picks coloured blocks with parallel jaws,
and places them at operator-specified coordinates. An overhead RGB camera finds
table outlines and block centres. Four wrist infrared probe positions per table
establish the surface planes. The simulator independently measures where each
block settles.

## Run on macOS with ROS 2 Humble

Use a Python environment containing ROS 2 Humble and colcon. Run these commands
from the repository root. The macOS viewer uses MuJoCo’s `mjpython` executable.

~~~zsh
conda activate ros2_env
source "$CONDA_PREFIX/setup.zsh"
python -m pip install -r requirements.txt
python tools/check_environment.py
colcon build --symlink-install
source install/setup.zsh
ros2 launch colour_sorting_arm demo.launch.py viewer:=true
~~~

In a second terminal, activate and source the same environment and workspace, then
run `ros2 run colour_sorting_arm targets`. After the launch terminal reports
`Camera-edge map ready`, enter commands at the prompt.

~~~text
green 80 80
red 185 145
blue 280 75
~~~

Each pair specifies the block centre in millimetres from the destination corner
nearest the base. X runs along the long edge away from the base, Y along the
short edge towards world −Y. The planner checks reach, edge clearance and
occupancy. A colour alone selects a free slot. Use `standard` for the three
commands above, `rescan` while idle to refresh the map, or `next` to abandon
an episode. Three verified transfers start a new random layout. Add `seed:=7`
to the launch command for a repeatable first layout.

## Measurements

The terminal compares estimated and true tabletop areas, then reports requested
XY, gripper and block XY before opening, settled XY and signed placement error.
Episode records are appended to `results/episodes.jsonl`. Ground truth is used
for scoring. The controller assumes known camera calibration, upright blocks of
known height and two visible, contrasting rectangular surfaces.

In three headless layouts the arm transferred 9/9 blocks with a mean radial
placement error of 8.6 mm. A transfer requires settled contact with the
destination table and a footprint inside its edges. See [validation](VALIDATION.md)
for individual trials and limitations.

## Project layout

The ROS package under `src/colour_sorting_arm/` has three implementation modules:
`perception.py` estimates workspace geometry, `controller.py` handles commands,
kinematics and motion, and `simulation.py` runs physics, sensors and scoring.
The physical model is `models/scene.xml`. Tests and diagnostic tools live outside
the runtime package.

## Offline checks

~~~sh
python -m pip install -r requirements.txt pytest
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python tools/validate_mapping.py
python tools/validate_controller.py --colours green red blue
~~~

The offline checks use MuJoCo dynamics and a software camera. To inspect a native
OpenGL frame on macOS, run `mjpython tools/diagnose_camera.py`.
