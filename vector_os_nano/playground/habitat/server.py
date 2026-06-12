# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Habitat sim server — runs INSIDE the pinned conda env (python 3.9).

STANDALONE BY DESIGN: imports only ``habitat_sim`` + stdlib — never the
``vector_os_nano`` package (the repo venv is py3.12; this script is executed
by the habitat conda interpreter, ADR-009 process split). The repo talks to
it through ``bridge.py`` over a localhost TCP socket, one JSON object per
line. The server binds an ephemeral port and prints ``PORT <n>`` on stdout
as the handshake.

Coordinate mapping (documented contract, mirrored in bridge tests):
habitat is Y-up with -Z forward at identity. World (BaseProtocol) frame:
``world = (-z_h, -x_h, y_h)`` and ``heading = yaw`` (rotation about +Y), so
at identity the agent faces +world_x and a positive yaw turns left (CCW),
matching REP-103 expectations.

Kinematics: ``walk`` integrates vx/vyaw in fixed dt steps, each step
navmesh-constrained via ``pathfinder.try_step`` (the VLN-CE recipe —
sliding along walls, never leaving the mesh). No dynamics; this world's
fidelity budget is photoreal vision + navigation, not gait physics.

Streaming (N1): a 50 Hz integration thread consumes ``set_velocity``
commands (0.6 s deadman) against a single pose/velocity authority; the
agent object is a render puppet synced on the op thread before any render.
Extra socket connections are served as STREAM channels (restricted op set
that never touches the simulator), so high-rate state polls and cmd_vel
writes never queue behind a pano render on the main channel.
``pathfinder.try_step`` concurrent with renders is spike-verified safe.
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import threading
import time

import habitat_sim
import numpy as np


def _quat_yaw(rotation) -> float:
    """Yaw (rotation about +Y) of a habitat agent rotation quaternion."""
    # habitat_sim returns a quaternion (w + xi + yj + zk, quaternion package).
    w, x, y, z = float(rotation.w), float(rotation.x), float(rotation.y), float(rotation.z)
    # Yaw about Y from quaternion (forward is -Z at identity).
    return math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (x * x + y * y))


def _yaw_quat(yaw: float):
    import quaternion  # vendored with habitat_sim

    return quaternion.from_rotation_vector([0.0, yaw, 0.0])


def _hab_to_world(p) -> "list[float]":
    return [-float(p[2]), -float(p[0]), float(p[1])]


def _world_to_hab(p) -> np.ndarray:
    return np.array([-float(p[1]), float(p[2]), -float(p[0])], dtype=np.float32)


class _Viewer:
    """Live first-person window (the habitat conda build is HEADLESS — no
    native window exists; this displays the offscreen EGL frames instead).

    All HighGUI calls (namedWindow/imshow/waitKey) happen in ONE daemon
    thread; producers hand BGR frames over a small queue (drop-oldest, the
    display never back-pressures the sim). A user closing the window simply
    stops the display — the sim keeps running.
    """

    def __init__(self, title: str) -> None:
        import queue
        import threading

        import cv2  # noqa: F401 — fail here, loudly, before the thread starts

        self._title = title
        self._queue: "queue.Queue" = queue.Queue(maxsize=2)
        self._dead = False
        threading.Thread(target=self._loop, daemon=True, name="habitat-viewer").start()

    def push(self, frame_bgr) -> None:
        if self._dead:
            return
        import queue

        try:
            self._queue.put_nowait(frame_bgr)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(frame_bgr)
            except queue.Empty:
                pass

    def _loop(self) -> None:
        import queue

        import cv2

        cv2.namedWindow(self._title, cv2.WINDOW_AUTOSIZE)
        while True:
            try:
                frame = self._queue.get(timeout=0.1)
                cv2.imshow(self._title, frame)
            except queue.Empty:
                pass
            cv2.waitKey(1)
            try:
                if cv2.getWindowProperty(self._title, cv2.WND_PROP_VISIBLE) < 1:
                    self._dead = True  # user closed the window; sim continues
                    return
            except Exception:  # noqa: BLE001
                self._dead = True
                return


class HabitatServer:
    def __init__(
        self,
        scene: str,
        gui: bool = False,
        dataset_config: str = "",
        navmesh: str = "",
        agent_radius: float = 0.1,
        agent_height: float = 1.5,
        robot_glb: str = "",
        viewer_mode: str = "first",
        viewer_size: int = 800,
    ) -> None:
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_id = scene
        if dataset_config:
            # Composed dataset scene (N0): scene is an instance NAME the
            # dataset config resolves (stage + furniture + lighting).
            cfg.scene_dataset_config_file = dataset_config
        self._viewer_mode = viewer_mode if viewer_mode in ("first", "chase") else "first"
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        # M3: an always-on egocentric RGB camera (256x256 — VLM-sized, cheap).
        rgb = habitat_sim.CameraSensorSpec()
        rgb.uuid = "rgb"
        rgb.sensor_type = habitat_sim.SensorType.COLOR
        rgb.resolution = [256, 256]
        rgb.position = [0.0, 1.2, 0.0]  # eye height on the agent
        # M4: equirectangular color+depth panoramas at the SysNav camera
        # contract scale — full 960x1920 here; the bridge crops ±60° vertical
        # (rows 160:800) to the 1920x640 cloud_image_fusion expects.
        pano_rgb = habitat_sim.EquirectangularSensorSpec()
        pano_rgb.uuid = "pano_rgb"
        pano_rgb.sensor_type = habitat_sim.SensorType.COLOR
        pano_rgb.resolution = [960, 1920]
        pano_rgb.position = [0.0, 1.2, 0.0]
        pano_depth = habitat_sim.EquirectangularSensorSpec()
        pano_depth.uuid = "pano_depth"
        pano_depth.sensor_type = habitat_sim.SensorType.DEPTH
        pano_depth.resolution = [960, 1920]
        pano_depth.position = [0.0, 1.2, 0.0]
        agent_cfg.sensor_specifications = [rgb, pano_rgb, pano_depth]
        if gui or robot_glb:
            # Dedicated viewer camera (gui runs AND headless body checks).
            # Native render at the window size beats upscaling. N3: 'chase'
            # mode mounts it behind/above the agent looking down — the
            # third-person view that shows the robot body; 'first' keeps the
            # eye view. Size is owner-tunable (--viewer-size).
            viewer_size = max(256, min(int(viewer_size), 1600))
            viewer_rgb = habitat_sim.CameraSensorSpec()
            viewer_rgb.uuid = "viewer_rgb"
            viewer_rgb.sensor_type = habitat_sim.SensorType.COLOR
            viewer_rgb.resolution = [viewer_size, viewer_size]
            if self._viewer_mode == "chase":
                viewer_rgb.position = [0.0, 1.7, 1.3]
                viewer_rgb.orientation = [-0.45, 0.0, 0.0]
            else:
                viewer_rgb.position = [0.0, 1.2, 0.0]
            agent_cfg.sensor_specifications.append(viewer_rgb)
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
        self.agent = self.sim.get_agent(0)
        # Navmesh: bare-glb scenes auto-load a sidecar .navmesh; dataset
        # scenes do NOT (bare habitat_sim ignores navmesh_instances), so the
        # repo side hands us the authored one explicitly. Last resort:
        # recompute from the loaded geometry with the agent's footprint.
        if not self.sim.pathfinder.is_loaded and navmesh:
            self.sim.pathfinder.load_nav_mesh(navmesh)
        if not self.sim.pathfinder.is_loaded:
            settings = habitat_sim.NavMeshSettings()
            settings.set_defaults()
            settings.agent_radius = agent_radius
            settings.agent_height = agent_height
            self.sim.recompute_navmesh(self.sim.pathfinder, settings)
        # Start somewhere legal on the navmesh.
        if self.sim.pathfinder.is_loaded:
            state = self.agent.get_state()
            state.position = self.sim.pathfinder.get_random_navigable_point()
            self.agent.set_state(state)
        # N1 — single pose/velocity authority (habitat frame). Op thread,
        # stream handlers and the integration thread all read/write THIS
        # under the lock; the agent object is a render puppet synced via
        # _sync_agent() on the op thread only (habitat_sim is not
        # thread-safe; pathfinder.try_step concurrent with renders is the
        # one exception, spike-verified).
        st0 = self.agent.get_state()
        self._state_lock = threading.Lock()
        self._pos = np.array(st0.position, dtype=np.float32)
        self._yaw = _quat_yaw(st0.rotation)
        self._vx = self._vyaw = 0.0          # applied (reported) velocity
        self._cmd_vx = self._cmd_vyaw = 0.0  # streaming command
        self._cmd_time = 0.0                 # monotonic stamp of last command
        self._lease = 0                      # op-thread motion loops own the pose
        self._closing = False
        threading.Thread(
            target=self._integration_loop, daemon=True, name="habitat-stream"
        ).start()
        # N3 — visible robot body: one rigid GLB (no joint animation, Tier A)
        # kinematically glued to the pose authority. Best-effort: a missing/
        # broken asset never blocks the world.
        self._body = None
        self._body_y_off = 0.0
        self._markers: list[dict] = []
        if robot_glb:
            try:
                import os as _os

                otm = self.sim.get_object_template_manager()
                ids = otm.load_configs(_os.path.dirname(robot_glb))
                rom = self.sim.get_rigid_object_manager()
                self._body = rom.add_object_by_template_id(ids[0])
                self._body.motion_type = habitat_sim.physics.MotionType.KINEMATIC
                # habitat re-centers the render asset to its bb CENTER, so a
                # floor-y translation sinks the body half its height. Lift by
                # the local-frame feet depth (bb is translation-independent).
                bb = self._body.root_scene_node.cumulative_bb
                self._body_y_off = -float(bb.min.y)
                self._place_body()
            except Exception as exc:  # noqa: BLE001
                print(f"robot body disabled: {exc}", file=sys.stderr, flush=True)
                self._body = None
        self._viewer = None
        if gui:
            try:
                title = f"Vector Habitat — {scene.rsplit('/', 1)[-1]}"
                self._viewer = _Viewer(title)
                self.emit_frame()  # first frame before any motion
            except Exception as exc:  # noqa: BLE001 — viewer is best-effort
                print(f"viewer disabled: {exc}", file=sys.stderr, flush=True)
                self._viewer = None

    # -- robot body (N3) --------------------------------------------------
    def _place_body(self) -> None:
        """Glue the body to the pose authority (op thread only). Model faces
        +x in the GLB; habitat forward is -z at yaw 0 — hence the +pi/2."""
        if self._body is None:
            return
        import magnum as mn

        pos, yaw, _, _ = self._read_pose()
        self._body.translation = mn.Vector3(
            float(pos[0]), float(pos[1]) + self._body_y_off, float(pos[2])
        )
        self._body.rotation = mn.Quaternion.rotation(
            mn.Rad(yaw + math.pi / 2.0), mn.Vector3.y_axis()
        )

    def _hide_body(self) -> None:
        """Teleport the body out of view before EGOCENTRIC renders — the eye
        sensors sit inside the head mesh (0.3.3 has no per-sensor masking)."""
        if self._body is None:
            return
        import magnum as mn

        self._body.translation = mn.Vector3(0.0, -100.0, 0.0)

    def set_markers(self, markers: list) -> int:
        """World-frame labelled markers overlaid on the viewer camera frames
        (owner finding 2026-06-12 round 2: SysNav detections were invisible —
        no scene-graph render surface). Replaces the whole set each call;
        bounded and sanitized (labels are drawn into pixels — cap length,
        strip newlines)."""
        clean: list[dict] = []
        for m in list(markers)[:32]:
            try:
                label = " ".join(str(m.get("label", "")).split())[:24]
                pos = _world_to_hab(
                    [float(m["x"]), float(m["y"]), float(m.get("z", 0.0))]
                )
            except (KeyError, TypeError, ValueError):
                continue
            clean.append({"label": label, "pos": pos})
        self._markers = clean
        return len(clean)

    def _overlay_markers(self, rgb):
        """Project + draw the marker set into a viewer frame (best-effort:
        no cv2 / no markers / odd transforms never break a render)."""
        markers = getattr(self, "_markers", None)
        if not markers:
            return rgb
        try:
            import cv2
            import magnum as mn

            node = self.sim._sensors["viewer_rgb"]._sensor_object.object
            inv = node.absolute_transformation().inverted()
            h, w = int(rgb.shape[0]), int(rgb.shape[1])
            f = (w / 2.0) / math.tan(math.radians(90.0) / 2.0)
            frame = np.ascontiguousarray(rgb[..., :3])
            for m in markers:
                p = inv.transform_point(mn.Vector3(*[float(v) for v in m["pos"]]))
                z = -float(p.z)
                if z < 0.3:
                    continue  # behind / on top of the camera
                u = int(w / 2.0 + f * float(p.x) / z)
                v = int(h / 2.0 - f * float(p.y) / z)
                if not (0 <= u < w and 0 <= v < h):
                    continue
                cv2.circle(frame, (u, v), 6, (40, 220, 90), 2)
                if m["label"]:
                    cv2.putText(
                        frame, m["label"], (u + 8, v - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 220, 90), 1,
                        cv2.LINE_AA,
                    )
            return frame
        except Exception:  # noqa: BLE001 — overlay must never kill a frame
            return rgb

    def _render_viewer_rgb(self):
        """Draw the viewer camera only (never the heavy equirect pair),
        body placed or hidden per viewer mode. Returns the RGB array."""
        if self._viewer_mode == "chase":
            self._place_body()
        else:
            self._hide_body()  # eye camera sits inside the head
        sensor = self.sim._sensors["viewer_rgb"]
        sensor.draw_observation()
        return self._overlay_markers(sensor.get_observation())

    def viewer_frame(self, body: bool = True) -> dict:
        """Viewer-camera frame as base64 PNG (headless N3 acceptance + owner
        artifacts). Requires the viewer sensor (gui or robot body runs).
        ``body=False`` renders with the body hidden — the acceptance
        discriminator proving the robot is actually in view."""
        import base64
        import io

        from PIL import Image

        if "viewer_rgb" not in getattr(self.sim, "_sensors", {}):
            return {"error": "no viewer sensor (boot with --gui or --robot-glb)"}
        self._sync_agent()
        if body:
            rgb = self._render_viewer_rgb()
        else:
            self._hide_body()
            sensor = self.sim._sensors["viewer_rgb"]
            sensor.draw_observation()
            rgb = sensor.get_observation()
        buf = io.BytesIO()
        Image.fromarray(rgb[..., :3]).save(buf, format="PNG")
        return {
            "png_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "viewer_mode": self._viewer_mode,
            "body": self._body is not None,
        }

    def emit_frame(self) -> None:
        """Render the viewer camera and hand the frame to the window (gui only).

        Sim access stays on THIS (op) thread — habitat_sim is not thread-safe;
        the viewer thread only displays. Never raises (display must not break
        the protocol loop).
        """
        if self._viewer is None or self._viewer._dead:
            return
        try:
            import cv2

            rgb = self._render_viewer_rgb()
            frame = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR)
            st = self.get_state()
            hud = (
                f"pos ({st['pos'][0]:.1f}, {st['pos'][1]:.1f})  "
                f"heading {math.degrees(st['heading']):.0f} deg"
            )
            cv2.putText(frame, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            self._viewer.push(frame)
        except Exception as exc:  # noqa: BLE001
            print(f"viewer frame failed: {exc}", file=sys.stderr, flush=True)

    # -- streaming pose authority (N1) -----------------------------------
    _STREAM_HZ = 50.0
    _CMD_TIMEOUT_S = 0.6  # deadman: a stale cmd stream must stop the robot

    def _read_pose(self) -> "tuple[np.ndarray, float, float, float]":
        with self._state_lock:
            return self._pos.copy(), self._yaw, self._vx, self._vyaw

    def _write_pose(self, pos, yaw: float) -> None:
        with self._state_lock:
            self._pos = np.array(pos, dtype=np.float32)
            self._yaw = float(yaw)

    def _acquire_lease(self) -> None:
        with self._state_lock:
            self._lease += 1
            self._cmd_vx = self._cmd_vyaw = 0.0  # explicit motion preempts streaming

    def _release_lease(self) -> None:
        with self._state_lock:
            self._lease = max(0, self._lease - 1)

    def _sync_agent(self) -> "tuple[np.ndarray, float]":
        """Apply the authoritative pose to the agent (render puppet).

        Op thread ONLY. Returns the (pos, yaw) snapshot that was applied so
        renders report exactly the pose they were taken at.
        """
        pos, yaw, _, _ = self._read_pose()
        st = self.agent.get_state()
        st.position = pos
        st.rotation = _yaw_quat(yaw)
        self.agent.set_state(st)
        return pos, yaw

    def set_velocity(self, vx: float, vyaw: float) -> dict:
        with self._state_lock:
            self._cmd_vx = float(vx)
            self._cmd_vyaw = float(vyaw)
            self._cmd_time = time.monotonic()
        return self.get_state()

    def _integration_loop(self) -> None:
        period = 1.0 / self._STREAM_HZ
        last = time.monotonic()
        while not self._closing:
            time.sleep(period)
            now = time.monotonic()
            dt = min(now - last, 4.0 * period)  # clamp scheduler hiccups
            last = now
            with self._state_lock:
                if self._lease:
                    self._vx = self._vyaw = 0.0
                    continue
                vx, vyaw = self._cmd_vx, self._cmd_vyaw
                if now - self._cmd_time > self._CMD_TIMEOUT_S:
                    vx = vyaw = 0.0  # publisher stopped/died: stop the robot
                if vx == 0.0 and vyaw == 0.0:
                    self._vx = self._vyaw = 0.0
                    continue
                yaw = self._yaw + vyaw * dt
                fwd = np.array(
                    [-math.sin(yaw), 0.0, -math.cos(yaw)], dtype=np.float32
                )
                target = self._pos + fwd * (vx * dt)
                if self.sim.pathfinder.is_loaded:
                    self._pos = np.array(
                        self.sim.pathfinder.try_step(self._pos, target),
                        dtype=np.float32,
                    )
                else:
                    self._pos = target
                self._yaw = yaw
                self._vx, self._vyaw = vx, vyaw

    # -- state ----------------------------------------------------------
    def get_state(self) -> dict:
        pos, yaw, vx, vyaw = self._read_pose()
        return {
            "pos": _hab_to_world(pos),
            "heading": yaw,
            "hab_pos": [float(v) for v in pos],
            "vel": [vx, 0.0, vyaw],
        }

    # -- motion ---------------------------------------------------------
    def walk(self, vx: float, vyaw: float, duration: float, dt: float = 0.1) -> dict:
        """Timed velocity walk, paced in WALL TIME (owner finding 2026-06-12
        round 2: the unpaced loop finished a 1 m walk in one blink — motion
        was real but invisible). ``duration`` is the honest Tier A contract:
        the op takes that long and the viewer animates through it."""
        self._acquire_lease()
        try:
            pos, yaw, _, _ = self._read_pose()
            steps = max(1, int(round(duration / dt)))
            step_dt = duration / steps
            for _ in range(steps):
                t0 = time.monotonic()
                yaw += vyaw * step_dt
                # forward in habitat frame for current yaw
                fwd = np.array(
                    [-math.sin(yaw), 0.0, -math.cos(yaw)], dtype=np.float32
                )
                target = pos + fwd * (vx * step_dt)
                if self.sim.pathfinder.is_loaded:
                    pos = np.array(
                        self.sim.pathfinder.try_step(pos, target), dtype=np.float32
                    )
                else:
                    pos = target
                self._write_pose(pos, yaw)
                if self._viewer is not None:
                    self._sync_agent()
                    self.emit_frame()
                time.sleep(max(0.0, step_dt - (time.monotonic() - t0)))
            self._write_pose(pos, yaw)
        finally:
            self._release_lease()
        return self.get_state()

    # -- navigation -------------------------------------------------------
    def navigate_to(
        self, x: float, y: float, tol: float = 0.2, speed: float = 1.0
    ) -> dict:
        """Follow the navmesh shortest path to world (x, y). Deterministic,
        bounded: per-waypoint, face the waypoint (kinematic yaw set) and
        advance via try_step; break when progress stalls (stuck/unreachable).
        Paced in wall time at ``speed`` m/s so the viewer animates the drive
        (``speed <= 0`` keeps the legacy instant advance for harnesses).
        """
        t0 = time.monotonic()
        # Lease FIRST (R6, design review #19): reading the pose before owning
        # motion lets a concurrent cmd_vel stream move the robot under us —
        # the plan would start from a stale pose.
        self._acquire_lease()
        try:
            start, yaw, _, _ = self._read_pose()
            goal_world = [float(x), float(y), _hab_to_world(start)[2]]
            goal = self.sim.pathfinder.snap_point(_world_to_hab(goal_world))

            gd0 = self.geodesic_distance(
                _hab_to_world(start), [float(x), float(y), _hab_to_world(start)[2]]
            )
            if gd0 <= tol:
                # Honest distinct outcome — never a zero-motion 'reached'.
                return {
                    "reached": True, "already_there": True, "moved_m": 0.0,
                    "elapsed_s": round(time.monotonic() - t0, 3),
                    "remaining": gd0, **self.get_state(),
                }

            path = habitat_sim.ShortestPath()
            path.requested_start = start
            path.requested_end = np.array(goal, dtype=np.float32)
            if not self.sim.pathfinder.find_path(path) or not list(path.points):
                return {
                    "reached": False, "reason": "no_path", "already_there": False,
                    "moved_m": 0.0, "elapsed_s": round(time.monotonic() - t0, 3),
                    **self.get_state(),
                }

            pos = start
            moved = 0.0
            step_len = 0.15
            # Intermediate waypoints use a FIXED small threshold; only the
            # FINAL waypoint honours the caller's tol (a 1.5 m tol used to
            # let the follower cut every corner of the path).
            waypoints = list(path.points)[1:]
            for wp_i, wp in enumerate(waypoints):
                wp = np.array(wp, dtype=np.float32)
                wp_tol = tol if wp_i == len(waypoints) - 1 else min(tol, 0.2)
                stall = 0
                for _ in range(400):  # hard bound per waypoint
                    delta = wp - pos
                    planar = math.hypot(float(delta[0]), float(delta[2]))
                    if planar <= wp_tol:
                        break
                    # face the waypoint (habitat: forward = -Z rotated by yaw)
                    yaw = math.atan2(-float(delta[0]), -float(delta[2]))
                    fwd = np.array(
                        [-math.sin(yaw), 0.0, -math.cos(yaw)], dtype=np.float32
                    )
                    target = pos + fwd * min(step_len, planar)
                    new_pos = np.array(
                        self.sim.pathfinder.try_step(pos, target), dtype=np.float32
                    )
                    step_moved = float(np.linalg.norm(new_pos - pos))
                    if step_moved < 1e-4:
                        stall += 1
                        if stall >= 3:
                            break  # wall-stuck: stop honestly, verify will judge
                    else:
                        stall = 0
                    moved += step_moved
                    pos = new_pos
                    self._write_pose(pos, yaw)
                    if self._viewer is not None:
                        self._sync_agent()
                        self.emit_frame()
                    if speed > 0.0 and step_moved > 0.0:
                        time.sleep(step_moved / speed)
            self._write_pose(pos, yaw)
        finally:
            self._release_lease()
        out = self.get_state()
        gd = self.geodesic_distance(out["pos"], [float(x), float(y), out["pos"][2]])
        # Strict arrival (R6, design review #2): the caller's tol, floored
        # only by the kinematic step quantum — the old tol*2.0 inflation let
        # a 1.5 m tol grade 3 m away as 'reached'.
        out["reached"] = bool(gd <= max(tol, 0.4))
        out["already_there"] = False
        out["moved_m"] = round(moved, 3)
        out["elapsed_s"] = round(time.monotonic() - t0, 3)
        out["remaining"] = gd
        return out

    def render(self) -> dict:
        """Egocentric RGB as base64 PNG (lazy PIL — conda env ships it)."""
        import base64
        import io

        from PIL import Image  # lazy: only the render op needs it

        self._sync_agent()  # render at the authoritative pose
        self._hide_body()   # egocentric: the eye sits inside the head mesh
        obs = self.sim.get_sensor_observations()
        rgb = obs["rgb"]
        buf = io.BytesIO()
        Image.fromarray(rgb[..., :3]).save(buf, format="PNG")
        return {
            "png_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
        }

    def pano(self) -> dict:
        """Equirectangular color+depth panorama, pose-synced (M4, SysNav input).

        Raw arrays (base64 of contiguous bytes — no PNG, no PIL): the repo
        side rebuilds them with numpy.frombuffer. depth is float32 metres.
        """
        import base64

        pos, yaw = self._sync_agent()  # pose-synced: report the rendered pose
        self._hide_body()  # the pano must never see our own head interior
        obs = self.sim.get_sensor_observations()
        rgb = np.ascontiguousarray(obs["pano_rgb"][..., :3], dtype=np.uint8)
        depth = np.ascontiguousarray(obs["pano_depth"], dtype=np.float32)
        if self._viewer is not None:
            self.emit_frame()  # streaming motion: window refreshes per pano
        return {
            "rgb_b64": base64.b64encode(rgb.tobytes()).decode("ascii"),
            "depth_b64": base64.b64encode(depth.tobytes()).decode("ascii"),
            "height": int(depth.shape[0]),
            "width": int(depth.shape[1]),
            "pos": _hab_to_world(pos),
            "heading": yaw,
        }

    # -- navigation oracle ----------------------------------------------
    def geodesic_distance(self, a: "list[float]", b: "list[float]") -> float:
        path = habitat_sim.ShortestPath()
        path.requested_start = _world_to_hab(a)
        path.requested_end = _world_to_hab(b)
        self.sim.pathfinder.find_path(path)
        return float(path.geodesic_distance)

    def snap_point(self, p: "list[float]") -> "list[float]":
        snapped = self.sim.pathfinder.snap_point(_world_to_hab(p))
        return _hab_to_world(snapped)

    def objects(self) -> "list[dict]":
        out = []
        try:
            for obj in self.sim.semantic_scene.objects or []:
                if obj is None or obj.category is None:
                    continue
                center = obj.aabb.center
                out.append(
                    {
                        "name": f"{obj.category.name()}_{obj.semantic_id}",
                        "category": obj.category.name(),
                        "pos": _hab_to_world(center),
                    }
                )
        except Exception:  # noqa: BLE001 — scenes without semantics
            return []
        return out

    # -- dispatch ---------------------------------------------------------
    def handle(self, req: dict) -> dict:
        op = req.get("op", "")
        if op == "ping":
            return {"ok": True, "pong": True}
        if op == "get_state":
            return {"ok": True, **self.get_state()}
        if op == "walk":
            return {
                "ok": True,
                **self.walk(
                    float(req.get("vx", 0.0)),
                    float(req.get("vyaw", 0.0)),
                    # #15 — cap: an unbounded duration holds the op thread
                    # past every client read budget (走20米 = 67 s > 60 s).
                    min(float(req.get("duration", 1.0)), _MAX_WALK_DURATION),
                ),
            }
        if op == "stop":
            with self._state_lock:  # stop kills the streaming command too
                self._cmd_vx = self._cmd_vyaw = 0.0
            return {"ok": True, **self.get_state()}
        if op == "set_velocity":
            return {
                "ok": True,
                **self.set_velocity(
                    float(req.get("vx", 0.0)), float(req.get("vyaw", 0.0))
                ),
            }
        if op == "navigate_to":
            return {
                "ok": True,
                **self.navigate_to(
                    float(req["x"]),
                    float(req["y"]),
                    float(req.get("tol", 0.2)),
                    float(req.get("speed", 1.0)),
                ),
            }
        if op == "set_markers":
            return {"ok": True, "count": self.set_markers(req.get("markers") or [])}
        if op == "render":
            return {"ok": True, **self.render()}
        if op == "viewer_frame":
            return {"ok": True, **self.viewer_frame(bool(req.get("body", True)))}
        if op == "pano":
            return {"ok": True, **self.pano()}
        if op == "geodesic_distance":
            return {"ok": True, "distance": self.geodesic_distance(req["a"], req["b"])}
        if op == "snap_point":
            return {"ok": True, "point": self.snap_point(req["p"])}
        if op == "objects":
            return {"ok": True, "objects": self.objects()}
        if op == "shutdown":
            return {"ok": True, "bye": True}
        return {"ok": False, "error": f"unknown op {op!r}"}


# Ops a STREAM channel may run: pure pose/cmd state, never the simulator —
# so a handler thread per extra connection is safe and never queues behind
# a render on the main channel.
# set_markers qualifies: it only swaps a python list (read by op-thread
# renders), never touches the simulator — and marker pushes arrive from the
# REPL's ROS callback thread, which must never queue behind a paced walk or
# a pano render on the main channel.
_STREAM_OPS = frozenset({"ping", "get_state", "set_velocity", "set_markers"})

# #15 — server-side cap on a single walk op: the bridge budgets its read
# timeout as duration + margin, so an uncapped duration could still starve
# every queued op behind one command.
_MAX_WALK_DURATION = 120.0


def _serve_main(rfile, wfile, server) -> None:
    """Main-channel serve loop: one JSON object per line, answer every line.

    ``op`` is localized per iteration (the stream handler's shape): a
    malformed FIRST line used to leave ``req`` unbound and the shutdown
    check then died on NameError, killing the whole server with stderr
    swallowed (#20). A bad line now gets its error response and the loop
    keeps serving.
    """
    for line in rfile:
        line = line.strip()
        if not line:
            continue
        op = ""
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            resp = {"ok": False, "error": f"bad json: {exc}"}
        else:
            op = req.get("op", "") if isinstance(req, dict) else ""
            try:
                resp = server.handle(req)
            except Exception as exc:  # noqa: BLE001 — report, never die mid-protocol
                resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            # #15 — echo the request id so the bridge can pair responses
            # with requests and discard stale lines after a read timeout.
            if isinstance(req, dict) and "rid" in req:
                resp["rid"] = req["rid"]
        wfile.write(json.dumps(resp) + "\n")
        wfile.flush()
        if op == "shutdown":
            break


def _serve_stream(conn: socket.socket, server: "HabitatServer") -> None:
    rfile = conn.makefile("r", encoding="utf-8")
    wfile = conn.makefile("w", encoding="utf-8")
    try:
        for line in rfile:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError as exc:
                resp = {"ok": False, "error": f"bad json: {exc}"}
            else:
                op = req.get("op", "")
                if op not in _STREAM_OPS:
                    resp = {
                        "ok": False,
                        "error": f"op {op!r} not allowed on the stream channel "
                                 f"(allowed: {sorted(_STREAM_OPS)})",
                    }
                else:
                    try:
                        resp = server.handle(req)
                    except Exception as exc:  # noqa: BLE001
                        resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            wfile.write(json.dumps(resp) + "\n")
            wfile.flush()
    except Exception:  # noqa: BLE001 — a dead stream client never kills the sim
        pass
    finally:
        try:
            conn.close()
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--gui", action="store_true",
                    help="open a live first-person viewer window")
    ap.add_argument("--dataset-config", default="",
                    help="scene_dataset_config.json for composed dataset scenes")
    ap.add_argument("--navmesh", default="",
                    help="authored navmesh to load explicitly (dataset scenes)")
    ap.add_argument("--agent-radius", type=float, default=0.1,
                    help="agent footprint radius for navmesh recompute fallback")
    ap.add_argument("--agent-height", type=float, default=1.5,
                    help="agent height for navmesh recompute fallback")
    ap.add_argument("--robot-glb", default="",
                    help="rigid robot-body GLB glued to the agent pose (N3)")
    ap.add_argument("--viewer-mode", default="first",
                    choices=("first", "chase"),
                    help="viewer camera: first-person eye or third-person chase")
    ap.add_argument("--viewer-size", type=int, default=800,
                    help="viewer camera/window size in pixels (square)")
    args = ap.parse_args()

    server = HabitatServer(
        args.scene,
        gui=args.gui,
        dataset_config=args.dataset_config,
        navmesh=args.navmesh,
        agent_radius=args.agent_radius,
        agent_height=args.agent_height,
        robot_glb=args.robot_glb,
        viewer_size=args.viewer_size,
        viewer_mode=args.viewer_mode,
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", args.port))
    sock.listen(4)
    print(f"PORT {sock.getsockname()[1]}", flush=True)

    conn, _ = sock.accept()

    def _accept_streams() -> None:
        # Every connection after the first is a STREAM channel (restricted
        # op set, own handler thread). The listener closing ends the loop.
        while True:
            try:
                extra, _ = sock.accept()
            except OSError:
                return
            threading.Thread(
                target=_serve_stream, args=(extra, server),
                daemon=True, name="habitat-stream-conn",
            ).start()

    threading.Thread(
        target=_accept_streams, daemon=True, name="habitat-stream-accept"
    ).start()

    rfile = conn.makefile("r", encoding="utf-8")
    wfile = conn.makefile("w", encoding="utf-8")
    _serve_main(rfile, wfile, server)
    server._closing = True  # stop the integration thread
    conn.close()
    sock.close()  # unblocks the stream acceptor
    server.sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
