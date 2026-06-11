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
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import sys

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


class HabitatServer:
    def __init__(self, scene: str) -> None:
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_id = scene
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        # M3: an always-on egocentric RGB camera (256x256 — VLM-sized, cheap).
        rgb = habitat_sim.CameraSensorSpec()
        rgb.uuid = "rgb"
        rgb.sensor_type = habitat_sim.SensorType.COLOR
        rgb.resolution = [256, 256]
        rgb.position = [0.0, 1.2, 0.0]  # eye height on the agent
        agent_cfg.sensor_specifications = [rgb]
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
        self.agent = self.sim.get_agent(0)
        # Start somewhere legal on the navmesh.
        if self.sim.pathfinder.is_loaded:
            state = self.agent.get_state()
            state.position = self.sim.pathfinder.get_random_navigable_point()
            self.agent.set_state(state)

    # -- state ----------------------------------------------------------
    def get_state(self) -> dict:
        st = self.agent.get_state()
        return {
            "pos": _hab_to_world(st.position),
            "heading": _quat_yaw(st.rotation),
            "hab_pos": [float(v) for v in st.position],
        }

    # -- motion ---------------------------------------------------------
    def walk(self, vx: float, vyaw: float, duration: float, dt: float = 0.1) -> dict:
        st = self.agent.get_state()
        pos = np.array(st.position, dtype=np.float32)
        yaw = _quat_yaw(st.rotation)
        steps = max(1, int(round(duration / dt)))
        step_dt = duration / steps
        for _ in range(steps):
            yaw += vyaw * step_dt
            # forward in habitat frame for current yaw
            fwd = np.array([-math.sin(yaw), 0.0, -math.cos(yaw)], dtype=np.float32)
            target = pos + fwd * (vx * step_dt)
            if self.sim.pathfinder.is_loaded:
                pos = np.array(
                    self.sim.pathfinder.try_step(pos, target), dtype=np.float32
                )
            else:
                pos = target
        st.position = pos
        st.rotation = _yaw_quat(yaw)
        self.agent.set_state(st)
        return self.get_state()

    # -- navigation -------------------------------------------------------
    def navigate_to(self, x: float, y: float, tol: float = 0.2) -> dict:
        """Follow the navmesh shortest path to world (x, y). Deterministic,
        bounded: per-waypoint, face the waypoint (kinematic yaw set) and
        advance via try_step; break when progress stalls (stuck/unreachable).
        """
        st = self.agent.get_state()
        start = np.array(st.position, dtype=np.float32)
        goal_world = [float(x), float(y), _hab_to_world(start)[2]]
        goal = self.sim.pathfinder.snap_point(_world_to_hab(goal_world))

        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = np.array(goal, dtype=np.float32)
        if not self.sim.pathfinder.find_path(path) or not list(path.points):
            return {"reached": False, "reason": "no_path", **self.get_state()}

        pos = start
        yaw = _quat_yaw(st.rotation)
        step_len = 0.15
        for wp in list(path.points)[1:]:
            wp = np.array(wp, dtype=np.float32)
            stall = 0
            for _ in range(400):  # hard bound per waypoint
                delta = wp - pos
                planar = math.hypot(float(delta[0]), float(delta[2]))
                if planar <= tol:
                    break
                # face the waypoint (habitat: forward = -Z rotated by yaw)
                yaw = math.atan2(-float(delta[0]), -float(delta[2]))
                fwd = np.array([-math.sin(yaw), 0.0, -math.cos(yaw)], dtype=np.float32)
                target = pos + fwd * min(step_len, planar)
                new_pos = np.array(
                    self.sim.pathfinder.try_step(pos, target), dtype=np.float32
                )
                if float(np.linalg.norm(new_pos - pos)) < 1e-4:
                    stall += 1
                    if stall >= 3:
                        break  # wall-stuck: stop honestly, verify will judge
                else:
                    stall = 0
                pos = new_pos
        st.position = pos
        st.rotation = _yaw_quat(yaw)
        self.agent.set_state(st)
        out = self.get_state()
        gd = self.geodesic_distance(out["pos"], [float(x), float(y), out["pos"][2]])
        out["reached"] = bool(gd <= max(tol * 2.0, 0.4))
        out["remaining"] = gd
        return out

    def render(self) -> dict:
        """Egocentric RGB as base64 PNG (lazy PIL — conda env ships it)."""
        import base64
        import io

        from PIL import Image  # lazy: only the render op needs it

        obs = self.sim.get_sensor_observations()
        rgb = obs["rgb"]
        buf = io.BytesIO()
        Image.fromarray(rgb[..., :3]).save(buf, format="PNG")
        return {
            "png_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
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
                    float(req.get("duration", 1.0)),
                ),
            }
        if op == "stop":
            return {"ok": True, **self.get_state()}
        if op == "navigate_to":
            return {
                "ok": True,
                **self.navigate_to(
                    float(req["x"]), float(req["y"]), float(req.get("tol", 0.2))
                ),
            }
        if op == "render":
            return {"ok": True, **self.render()}
        if op == "geodesic_distance":
            return {"ok": True, "distance": self.geodesic_distance(req["a"], req["b"])}
        if op == "snap_point":
            return {"ok": True, "point": self.snap_point(req["p"])}
        if op == "objects":
            return {"ok": True, "objects": self.objects()}
        if op == "shutdown":
            return {"ok": True, "bye": True}
        return {"ok": False, "error": f"unknown op {op!r}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args()

    server = HabitatServer(args.scene)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", args.port))
    sock.listen(1)
    print(f"PORT {sock.getsockname()[1]}", flush=True)

    conn, _ = sock.accept()
    rfile = conn.makefile("r", encoding="utf-8")
    wfile = conn.makefile("w", encoding="utf-8")
    for line in rfile:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            resp = {"ok": False, "error": f"bad json: {exc}"}
        else:
            try:
                resp = server.handle(req)
            except Exception as exc:  # noqa: BLE001 — report, never die mid-protocol
                resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        wfile.write(json.dumps(resp) + "\n")
        wfile.flush()
        if req.get("op") == "shutdown":
            break
    conn.close()
    sock.close()
    server.sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
