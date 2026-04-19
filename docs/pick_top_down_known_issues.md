# pick_top_down — Known Issues (2026-04-19 live REPL)

Tracks concrete bugs observed in Yusen's `vector-cli go2sim with_arm=1`
REPL session on 2026-04-19 (after commits up through `e0a7e33`).
Next session should start here.

---

## Bug 1 — rclpy "Executor is already spinning" (thread error, non-fatal)

### Symptom
```
vector> go2sim with arm
Exception in thread Thread-3 (<lambda>):
  File ".../piper_ros2_proxy.py", line 186, in <lambda>
    target=lambda: rclpy.spin(self._node), daemon=True,
  File "/opt/ros/jazzy/.../rclpy/executors.py", line 222, in _enter_spin
    raise RuntimeError('Executor is already spinning')
Exception in thread Thread-4 (<lambda>):
  File ".../piper_ros2_proxy.py", line 480, in <lambda>
  ...
  RuntimeError: Executor is already spinning
  ▸ sim start go2 ok 23.8s
```

### Root cause
Three proxies in the main process each start their own spin thread:
- `Go2ROS2Proxy` → `rclpy.spin(node)` in thread
- `PiperROS2Proxy` → `rclpy.spin(node)` in thread
- `PiperGripperROS2Proxy` → `rclpy.spin(node)` in thread

`rclpy.spin()` installs a process-global default executor; only ONE can
be active. The first to call wins, the rest raise.

### Why the sim still starts
Spin errors happen in daemon threads; they log but don't abort main.
The Piper Node objects DO get created — so publishers work (outbound).
But subscribers never fire (no spin serving the Node), so:
- `PiperROS2Proxy._last_joint_state` stays at default `[0.0] * 7`
- `PiperGripperROS2Proxy._last_gripper_pos` stays at 0.0
- `PickTopDownSkill` may read stale/zero state

### Fix plan (next session)
Switch to a single `rclpy.executors.MultiThreadedExecutor` owned by a
shared helper (e.g. a new `Ros2Runtime` singleton or pass executor into
each proxy). Each proxy creates its Node and `executor.add_node(n)`.
One thread runs `executor.spin()`.

Files to touch:
- `vector_os_nano/hardware/sim/go2_ros2_proxy.py` — accept/share executor
- `vector_os_nano/hardware/sim/piper_ros2_proxy.py` — both classes
- `vector_os_nano/vcli/tools/sim_tool.py` — construct executor once

---

## Bug 2 — "Cannot locate target object" for pick_top_down

### Symptom
```
vector> 抓前面绿色
  > VGG 抓前面绿色
  >   [1/1] pick_top_down_goal — 抓前面绿色 via pick_top_down_skill
WARNING:GoalExecutor: execution failed for pick_top_down_goal:
  Cannot locate target object
  > [1/1] pick_top_down_goal failed — Cannot locate target object 0.0s
```

### Root-cause candidates (debug next session)

(a) **`_populate_pickables_from_mjcf` didn't register objects.**
    The main process loads the MJCF independently. If the path resolves
    wrong or loading silently fails, world_model stays empty. Check
    for `[sim_tool] registered N pickable objects` INFO line in startup.
    If missing, fix path resolution.

(b) **Label mismatch** (most likely).
    Goal text: `抓前面绿色`. LLM may have resolved it to
    `object_label="前面绿色"` or `"绿色"` or even empty — neither
    matches stored labels (`"blue bottle"`, `"green bottle"`, `"red can"`).
    `WorldModel.get_objects_by_label` uses substring match on normalised
    strings. `"绿色"` vs `"green bottle"` would not match without a
    color-word translation table.

(c) **Stale arm state** from Bug #1 is orthogonal — target resolution
    hits world_model, not the arm. But worth ruling out since both
    failed in the same session.

### Debug plan (next session)

1. Add a DEBUG line in `PickTopDownSkill._resolve_target` that dumps
   `params` + `wm.get_objects()` before returning None.
2. Check startup log for the "registered N pickable objects" message.
3. If (b), add a color-keyword normaliser to the skill's target
   resolver: map `红 / 绿 / 蓝 / 黄 / 红色 / 绿色 …` → english and
   retry `get_objects_by_label` with the normalised term.
4. Alternatively, update the skill's `parameters.source` metadata so
   VGG sends the `object_id` directly (pickable_bottle_green etc.)
   when the goal hints at a specific object.

### Workaround until fixed
Invoke with an explicit id by tool call:
```
> pick_top_down_skill object_id=pickable_bottle_green
```
or with the english label:
```
> 抓 green bottle
```

---

## Bug 3 — No perception backend for detect_* fallback

### Symptom
After Bug 2 failed, VGG tried a `detect_green_object` fallback:
```
WARNING:DetectSkill: No perception backend available
```

### Cause
`with_arm` sim mode doesn't wire `context.perception`. Only
`agent._vlm` (VLM for image queries) is set.

### Fix plan
Two options:
1. Stand up a minimal perception pipeline in `_start_go2` when
   `with_arm=True` — hook `/camera/image` + `/camera/depth` into a
   simple detect/track module.
2. Make VGG respect the skill's parameter source metadata
   (`source=world_model.objects.label`). When a skill declares its
   parameters come from the world model, don't inject a detect step.

Option 2 is smaller scope and matches the pick_top_down contract
("positions known a priori, no perception").

---

## Next-session checklist

1. Add debug logging to `PickTopDownSkill._resolve_target` +
   `_populate_pickables_from_mjcf` (temporary).
2. Run `go2sim` with_arm=1 and capture the startup + skill logs.
3. Diagnose whether Bug 2 is (a) or (b).
4. Implement the MultiThreadedExecutor refactor for Bug 1.
5. Implement the color-normaliser (or source-respect) for Bug 2.
6. Yusen re-verifies `抓前面绿色`, `抓红色罐头`, `抓蓝瓶子`.
