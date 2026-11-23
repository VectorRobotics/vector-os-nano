# Malo: existing NAV STACK control

These parameters are for the Malo household simulation (0.42 m planning radius,
0.15 m/s translation, 0.2 rad/s yaw). They are simulation estimates, not hardware
calibration. The external ROS Jazzy workspace must already contain FAR, TARE,
localPlanner/pathFollower, sensor_scan_generation and terrain_analysis.

After installing the household assets and the current VECTOR package, source ROS
and the workspace in the Python environment that provides `rclpy` and MuJoCo 4:

```bash
source /opt/ros/jazzy/setup.bash
export VECTOR_NAV_STACK=/absolute/path/to/Z-Navigation-Stack-go2w
source "$VECTOR_NAV_STACK/install/setup.bash"
export VECTOR_NAV_CONTROL=1
export VECTOR_NAV_CONFIG=/absolute/path/to/vector-os-nano/examples/household/navigation/malo
vector-cli
```

In the real CLI, enter:

```text
启动 malo 的 MuJoCo 仿真，使用 artvip_home 世界，打开仿真窗口。
去餐厅
```

`使用 TARE 开始自主探索。` selects TARE; `stop` cancels movement and `quit`
closes the owned stack. Exploration is bounded to 120 simulation seconds and
does not claim coverage completion when stopped or timed out. The GUI uses an
oblique third-person camera, following the robot and orbiting away from walls.

The control chain is:

```text
CLI goal -> FAR /far/way_point --+
                               +-> /way_point -> localPlanner /path
CLI explore -> TARE /tare/way_point                     |
                                                       v
MuJoCo wheel actuators <- VECTOR lease <- pathFollower /navigation_cmd_vel
```

Only the selected global planner's waypoints are forwarded. The skill thread
applies fresh body-frame commands under the base's stop/ownership guard, capped
to the embodiment limits. Missing or stale route/path/velocity streams stop
motion. There is no authored-map planner or Python pursuit fallback in this mode.
Arrival is checked against the original requested coordinate; a planner's goal
adjustment is not silently accepted as arrival at that coordinate.

`/vector/navigation/status`, `/vector/navigation/path` and
`/vector/navigation/applied_cmd` expose the selected planner and applied commands.
The supplied `system.rviz` also displays the three configured RGB-D cameras,
laser/terrain clouds, TF, paths and load feedback. RobotModel requires the local
CAD-derived robot description publisher used by the full demo launcher. CAD,
scene assets, recordings and credentials are not included here.

The local full demo entry remains
`/home/yusenzlabpc/vector-os-household-output/20261121/02_启动/完整系统演示.sh`.
It supplies the sensor monitor, terminal and RViz arrangement. Hardware deployment,
full-house coverage and long-horizon manipulation are separate acceptance gates.

For a closed run folder containing `nav-bag`, verify that nonzero applied commands
match recorded `pathFollower` output (with the Malo limits) using:

```bash
python /absolute/path/to/vector-os-nano/examples/household/navigation/malo/verify_bag.py /absolute/path/to/run
```

Record `/navigation_cmd_vel`, `/vector/navigation/applied_cmd`,
`/vector/navigation/status` and `/odom_base_link` before the first goal. This check
establishes command provenance and observed motion; it does not replace the
independent raw-physics grade or continuous-motion review.
