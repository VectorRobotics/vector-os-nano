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

Validation on 2026-11-23 (MuJoCo 4.0.1, existing ROS Jazzy NAV STACK):

- One cold bare-CLI `去餐厅` goal: `GROUNDED verified=True`; independent physical
  checks 14/14 pass, original-goal error 0.095974 m, final speed below 0.000001 m/s.
- All 1,862 nonzero applied commands match recorded C++ pathFollower output.
  The separate FAR -> stop -> TARE -> stop -> FAR run matched another 613 commands
  (159 from TARE). It demonstrates exploration control and cancellation, not complete
  house coverage.
- 24 live topics and six TF transforms checked; 160 focused tests pass, none skipped.
  The continuous GUI recording includes simulation, CLI, RViz and all three configured
  RGB-D cameras: 738.4 seconds at 1920x1080/15 fps, full decode passes.
- The preceding failed rollout stopped 0.444 m short because localPlanner inherited
  a 0.5 m arrival threshold. `local.yaml` sets it to 0.05 m; runtime parameter readback
  and a regression test confirm this. The fixed physical criterion stays at 0.15 m.

Evidence is local under `vector-os-household-output/20261121/04_验证证据/`:
`full-system-20261123-101548` (final), `094917` (planner switching), `095747`
(preserved threshold failure). The final recording is
`01_演示/NAV_STACK真实控制-第三人称-20261123.mp4`. No held-out reset seed was declared.
Room coordinates and ideal simulated odometry are known inputs. Continuous ergonomic
motion review remains unverified, so the long-horizon deployment gate is closed.
