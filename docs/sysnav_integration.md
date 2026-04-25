# SysNav Integration — Bringup Mode

Vector OS Nano consumes the **SysNav** project as its standard semantic
scene-graph backend. SysNav runs as a sibling ROS2 workspace; we
subscribe to its topic outputs through a thin Apache-2.0 adapter inside
[`vector_os_nano/integrations/sysnav_bridge/`](../vector_os_nano/integrations/sysnav_bridge/).

> **License boundary**. SysNav is licensed
> **PolyForm-Noncommercial-1.0.0**. Vector OS Nano is **Apache-2.0**.
> We do **not** copy any SysNav source files into this repository.
> Users install SysNav themselves; both are kept in their own
> directories, both are loaded from a single ROS2 environment at
> runtime, and only ROS2 messages cross the boundary.

---

## 1. What SysNav gives us

| SysNav publishes | Type | Use in Vector OS Nano |
|---|---|---|
| `/object_nodes_list` | `tare_planner/ObjectNodeList` | Per-object 3D pose + 8-corner bbox + label + status (`new` / `persistent` / `moving` / `disappeared`) → seeds `WorldModel.ObjectState` |
| `/object_type_query` | `tare_planner/ObjectType` | Forwards a VLM-classified object type request — we can react with a skill or ignore |
| `/obj_points` | `sensor_msgs/PointCloud2` | Per-object pointcloud cluster — optional, used by `pick_top_down` if available |
| `/obj_boxes` | `visualization_msgs/MarkerArray` | RViz visualisation, passthrough |
| `/obj_labels` | `visualization_msgs/MarkerArray` | RViz visualisation, passthrough |
| `/annotated_image` | `sensor_msgs/Image` | Debug / GUI visualisation |

| SysNav consumes | Type | What we publish |
|---|---|---|
| `/target_object_instruction` | `tare_planner/TargetObjectInstruction` | Vector OS Nano translates user intent (e.g. "go find the blue trash can in the kitchen") into this message; SysNav's VLM reasoning + planner take over |
| `/registered_scan` | `sensor_msgs/PointCloud2` | Lidar (Livox Mid-360) — provided by the SLAM stack, not by us |
| `/state_estimation` | `nav_msgs/Odometry` | SLAM odom — same |
| `/camera/image` | `sensor_msgs/Image` | RGB stream — provided by the camera driver |

The L1 (VLM Reasoning) node also subscribes to `/instruction`,
`/anchor_object`, `/room_navigation_query`, `/object_type_query`,
etc. — see `src/vlm_node/vlm_node/vlm_reasoning_node.py` in the SysNav
repo for the full surface.

---

## 2. Hardware requirements (SysNav side)

These come from **SysNav**, not Vector OS Nano. Without them, SysNav
will not start cleanly and our bridge will receive nothing.

| Component | Why | Notes |
|---|---|---|
| **Livox Mid-360 lidar** | All branches use `livox_ros_driver2` + `arise_slam_mid360` for SLAM; semantic mapping clusters lidar voxels | No RealSense / monocular fallback exists in any SysNav branch as of 2026-04. |
| **Ricoh Theta Z1 360-degree camera** | `cloud_image_fusion.py` projects panoramic RGB onto each lidar voxel; narrow-FoV cameras under-cover | Other 360-degree cameras may work if they expose an equivalent equirectangular topic. |
| **Compute** | RTX 4090 desktop + Intel NUC onboard (per SysNav README) | Yusen's RTX 5080 covers the desktop side. |
| **Network** | Wired or stable WiFi between robot, NUC, and desktop | SysNav's domain_bridge config splits topics by domain. |

For Go2 specifically, use the `unitree_go2` branch — it adds
`unitree_webrtc_ros` for `cmd_vel` → Go2 sport_cmd over WebRTC.

> **If the hardware is not yet available**, the bringup adapter still
> imports cleanly (no ROS2 import at module load time) and will simply
> log "no scene graph received" until topics start flowing. Tests use
> mocked subscribers — no hardware needed for unit / integration tests.

---

## 3. Bringup order

Three terminals, in this order:

### Terminal 1 — SysNav (its own workspace)

```bash
cd ~/Desktop/SysNav
git checkout unitree_go2     # match Go2 platform
source /opt/ros/jazzy/setup.bash
source install/setup.bash
./system_real_robot_with_exploration_planner_go2.sh
# Wait until you see:
#   [semantic_mapping_node]: object_nodes_list publisher created
#   [vlm_reasoning_node]: ready
#   [tare_planner]: ready
```

Or for desk-side / bagfile testing:

```bash
./system_bagfile_with_exploration_planner.sh ~/path/to/bag.db3
```

### Terminal 2 — Vector OS Nano

```bash
cd ~/Desktop/vector_os_nano
source /opt/ros/jazzy/setup.bash
source ~/Desktop/SysNav/install/setup.bash    # for tare_planner.msg
.venv-nano/bin/activate
vector-cli go2sim                              # or: vector-cli connect-go2
```

`vector-cli` startup probes for `tare_planner.msg.ObjectNodeList`
imports. If absent, the bridge logs a warning and degrades to
`world_model` populated only by built-in MJCF auto-registration —
behaviour identical to today.

### Terminal 3 — RViz (optional)

```bash
ros2 run rviz2 rviz2 -d ~/Desktop/SysNav/src/exploration_planner/tare_planner/rviz/tare_planner_ground.rviz
```

---

## 4. Adapter contract

```
                       SysNav workspace
                              │
                              │  /object_nodes_list  (ObjectNodeList)
                              │  /room_nodes_list    (RoomNodeList)
                              ▼
        Vector OS Nano: integrations/sysnav_bridge/
                              │
        ┌─────────────────────┼─────────────────────┐
        │ ObjectNode adapter  │ RoomNode adapter    │
        │  ↓                  │  ↓                  │
        │ ObjectState         │ SceneGraph rooms    │
        └──────────┬──────────┴─────────┬───────────┘
                   ▼                    ▼
             world_model              SceneGraph
                   │                    │
                   └────────┬───────────┘
                            ▼
              DetectSkill / PickTopDownSkill / NavigateSkill
              MobilePickSkill / ExploreSkill / PlaceTopDownSkill
```

### Field mapping `ObjectNode` → `ObjectState`

| `tare_planner/ObjectNode` | `vector_os_nano.core.world_model.ObjectState` | Notes |
|---|---|---|
| `object_id[0]` | `object_id` | SysNav merges duplicate detections; we keep the *primary* ID |
| `label` | `label` | Lower-cased, en/cn alias-mapped via existing `label_to_en_query` |
| `position.x/y/z` | `x/y/z` | Already in map frame — no further calibration |
| `bbox3d[0..7]` | `properties["bbox3d_corners"]` | 8 corners, oriented; preserved as raw for grasp planning |
| `status` ∈ {new, persistent, moving, disappeared} | `state` ∈ {`on_table`, `grasped`, `placed`, `unknown`} | Mapped: `disappeared` → `unknown`, `moving` → `unknown`, others → `on_table` (or current state if grasped/placed) |
| `cloud` (PointCloud2) | `properties["pointcloud_topic"]` | Optional; if present, top-down pick can use it for centroid refinement |
| `header.stamp` | `last_seen` | Float seconds |
| `is_asked_vlm` | `properties["asked_vlm"]` | bool |
| `viewpoint_id` | `properties["viewpoint_id"]` | int |

### Confidence

`ObjectNode` does not carry a single confidence scalar — SysNav's
`SingleObject.weighted_class_scores` is internal. We default
`ObjectState.confidence = 1.0` and refine via `is_asked_vlm` (0.7 if
unverified, 1.0 after VLM confirmation). Tunable.

### Topic QoS

SysNav publishes `ObjectNodeList` with depth-200 reliable. We match
with depth-50 reliable. No need for `BEST_EFFORT` — these are
per-room updates, not high-frequency telemetry.

---

## 5. Test strategy

### Unit (no SysNav running)

- `tests/integration/test_sysnav_bridge_object_node_mapping.py` —
  build a mock `ObjectNodeList` payload (without importing
  `tare_planner.msg`; we type-shadow the schema in
  `topic_interfaces.py` for test isolation), feed through the adapter,
  assert `ObjectState` fields populate correctly.
- `tests/integration/test_sysnav_bridge_status_transitions.py` —
  status `disappeared` removes the object after a TTL; `moving`
  keeps it but lowers confidence.

### Integration (mock ROS2 publisher)

- `scripts/mock_sysnav_publisher.py` — replay a recorded
  `ObjectNodeList` JSON through `rclpy` so the bridge can be
  exercised without SysNav running. Useful for nightly CI.

### E2E (real SysNav, GPU + Livox required — Yusen smoke)

- Bringup order from §3, then `vector-cli > 抓起蓝色瓶子` should
  resolve `world_model` against SysNav-populated entries within
  2 seconds of the dog seeing the table.

---

## 6. Carry-forward debt

- Yusen to confirm Go2 sensor stack matches SysNav requirements
  (Livox + 360-degree camera). If a different lidar / camera is
  installed, we may need to publish synthetic `/registered_scan`
  and `/camera/image` from RealSense — NOT in v3.0 scope.
- SysNav's `unitree_go2` branch adds `unitree_webrtc_ros` for
  `cmd_vel` — overlaps with our existing `Go2ROS2Proxy` walk path.
  Reconciliation deferred (one or the other should own the topic).
- Scene graph live updates (status=moving) interact with
  `MobilePickSkill._resolve_target` cache — invalidate on each
  `/object_nodes_list` callback, do not cache stale positions.
- VLM reasoning replication: SysNav has its own VLM node; Vector OS
  Nano has its own LLM brain (Anthropic / OpenAI). Decide which
  layer answers natural-language queries — see ADR (TBD) for
  coordination policy.

---

## 7. References

- SysNav repo (sibling lab): https://github.com/zwandering/SysNav
- SysNav paper: arXiv [2603.06914](https://arxiv.org/abs/2603.06914)
- SysNav project page: https://cmu-vln.github.io/
- Vector Navigation Stack (same lab): https://github.com/VectorRobotics/vector_navigation_stack
