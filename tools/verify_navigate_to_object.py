# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Step-#2 real-sim verification: navigate_to_object drives to a known object.

ONE headless go2+arm sim:
  1. Seed the scene graph via the REAL look skill (stub VLM naming, real
     depth localization — the verified #1 path) so 'green bottle' has a position.
  2. Drive the dog AWAY from the table (go2.navigate_to to a far point).
  3. Run NavigateToObjectSkill("green bottle").
  4. MEASURE the dog's actual final position vs the green-bottle MuJoCo GT
     (ground truth, not a nav flag): it must end at the standoff vicinity,
     having started far and moved substantially closer.

Prints `RESULT {json}`.  Run:
  MUJOCO_GL=egl PATH=/usr/bin:$PATH .venv/bin/python tools/verify_navigate_to_object.py
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json  # noqa: E402
import math  # noqa: E402

import mujoco as mj  # noqa: E402

from vector_os_nano.core.scene_graph import SceneGraph  # noqa: E402
from vector_os_nano.core.skill import SkillContext  # noqa: E402
from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2  # noqa: E402
from vector_os_nano.hardware.sim.mujoco_piper import MuJoCoPiper  # noqa: E402
from vector_os_nano.hardware.sim.mujoco_piper_gripper import (  # noqa: E402
    MuJoCoPiperGripper,
)
from vector_os_nano.perception.go2_grasp_perception import Go2GraspPerception  # noqa: E402
from vector_os_nano.perception.vlm_go2 import (  # noqa: E402
    DetectedObject, RoomIdentification, SceneDescription,
)
from vector_os_nano.skills.go2.look import LookSkill  # noqa: E402
from vector_os_nano.skills.navigate_to_object import (  # noqa: E402
    NavigateToObjectSkill, _VICINITY_CLEARANCE_M,
)

_TARGET = "green bottle"
_GT_BODY = "pickable_bottle_green"
_FAR_POINT = (8.0, 3.0)  # ~2.9 m back from the table, open floor in front


class _StubVLM:
    """Stubs only the (network-blocked) object naming; localization stays real."""

    def describe_scene(self, frame):
        objs = [
            DetectedObject(name="green bottle", description="", confidence=0.9),
            DetectedObject(name="blue bottle", description="", confidence=0.9),
            DetectedObject(name="red can", description="", confidence=0.9),
        ]
        return SceneDescription(summary="containers on a table", objects=objs,
                                room_type="kitchen", details="")

    def identify_room(self, frame):
        return RoomIdentification(room="kitchen", confidence=0.9, reasoning="stub")


def _gt_xy(go2: MuJoCoGo2, name: str) -> tuple[float, float, float]:
    model, data = go2._mj.model, go2._mj.data
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
    p = data.xpos[bid]
    return float(p[0]), float(p[1]), float(p[2])


def _dist(ax, ay, bx, by) -> float:
    return math.hypot(ax - bx, ay - by)


def main() -> int:
    out: dict = {}
    go2 = MuJoCoGo2(gui=False, room=True, backend="mpc", with_arm=True)
    go2.connect()
    piper = MuJoCoPiper(go2)
    piper.connect()
    gripper = MuJoCoPiperGripper(go2)
    gripper.connect()
    perception = Go2GraspPerception(go2, width=320, height=240)

    gx, gy, gz = _gt_xy(go2, _GT_BODY)
    out["green_gt"] = [round(gx, 3), round(gy, 3), round(gz, 3)]
    print(f"green bottle GT = ({gx:.3f}, {gy:.3f}, {gz:.3f})")

    # Let the dog settle into a stable stance so the head camera is framed on the
    # table, and warm up the detector, before the first look.
    import time  # noqa: PLC0415
    from vector_os_nano.perception.object_localizer import localize_objects_3d  # noqa: PLC0415, E402
    time.sleep(2.5)
    diag = localize_objects_3d(perception, [_TARGET])
    print(f"diag direct localize {_TARGET!r}: {diag}")
    out["diag_direct_localize"] = [[r[0], round(r[1], 3), round(r[2], 3), round(r[3], 3)] for r in diag]

    # --- 1. seed scene graph via the real look skill (stub VLM naming) ---------
    sg = SceneGraph()
    look_ctx = SkillContext(
        bases={"default": go2},
        perception_sources={"default": perception},
        services={"vlm": _StubVLM(), "spatial_memory": sg},
    )
    look_res = LookSkill().execute({}, look_ctx)
    found = sg.find_objects_by_category(_TARGET)
    seeded = [(o.category, round(o.x, 3), round(o.y, 3)) for o in found]
    print(f"look success={look_res.success}; scene-graph {_TARGET!r} = {seeded}")
    out["seed"] = {"look_success": look_res.success, "objects": seeded}
    if not found or all(abs(o.x) < 0.01 and abs(o.y) < 0.01 for o in found):
        print("RESULT " + json.dumps({**out, "overall_pass": False,
                                      "reason": "scene graph not seeded with a real position"}))
        return 1

    # --- 2. drive the dog AWAY from the table ---------------------------------
    print(f"driving dog away to {_FAR_POINT} ...")
    go2.navigate_to(_FAR_POINT[0], _FAR_POINT[1], timeout=60)
    p = go2.get_position()
    start_dist = _dist(p[0], p[1], gx, gy)
    out["after_move"] = {"pos": [round(p[0], 2), round(p[1], 2)],
                         "dist_to_green": round(start_dist, 2)}
    print(f"after move: pos=({p[0]:.2f},{p[1]:.2f}) dist_to_green={start_dist:.2f}m")

    # --- 3. navigate_to_object('green bottle') --------------------------------
    nav_ctx = SkillContext(
        bases={"default": go2},
        perception_sources={"default": perception},
        services={"spatial_memory": sg},
    )
    res = NavigateToObjectSkill().execute({"object": _TARGET}, nav_ctx)
    pf = go2.get_position()
    final_dist = _dist(pf[0], pf[1], gx, gy)
    rd = res.result_data or {}
    print(f"navigate_to_object success={res.success} diag={res.diagnosis_code}")
    print(f"  standoff={rd.get('standoff')} object_world={rd.get('object_world')}")
    print(f"  final pos=({pf[0]:.2f},{pf[1]:.2f}) dist_to_green={final_dist:.2f}m "
          f"(standoff target {_VICINITY_CLEARANCE_M}m)")
    out["navigate"] = {
        "success": res.success, "diag": res.diagnosis_code,
        "standoff": rd.get("standoff"), "object_world": rd.get("object_world"),
        "final_pos": [round(pf[0], 2), round(pf[1], 2)],
        "final_dist_to_green": round(final_dist, 2),
    }

    # --- 4. verdict (measure ground truth, not the nav flag) ------------------
    started_far = start_dist > 1.5
    arrived = final_dist <= _VICINITY_CLEARANCE_M + 0.6      # standoff + nav tol margin
    moved_closer = final_dist < start_dist - 1.0
    overall = bool(res.success and started_far and arrived and moved_closer)
    out["checks"] = {"started_far": started_far, "arrived_vicinity": arrived,
                     "moved_closer": moved_closer}
    out["overall_pass"] = overall

    for dev in (gripper, piper, go2):
        try:
            dev.disconnect()
        except Exception:  # noqa: BLE001
            pass

    print("RESULT " + json.dumps(out))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
