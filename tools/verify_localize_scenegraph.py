# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Step-#1 real-sim verification: accurate object positions into the scene graph.

Runs ONE headless go2+Piper sim and checks three things against MuJoCo ground truth:

  Part 1 (deterministic, HEADLINE): localize_objects_3d() on the 3 table pickables
          -> world (x,y,z) compared to MjData ground-truth body xpos.
  Part 2 (deterministic): feed those coords through SceneGraph.observe_with_viewpoint
          -> save() -> load() -> find_objects_by_category, confirm coords round-trip.
  Part 3 (full path, real GPT-4o VLM): run the actual `look` skill (the path Yusen
          asked about) and inspect the resulting scene-graph object coords vs GT.

Also dumps the robot's perception RGB frame to /tmp so a human (or the orchestrator)
can eyes-on confirm the bottles are actually in view.

Acceptance is the real sim + ground truth, never a unit test. Prints `RESULT {json}`.

Run:  MUJOCO_GL=egl PATH=/usr/bin:$PATH .venv/bin/python tools/verify_localize_scenegraph.py
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402


def _load_dotenv() -> None:
    """Best-effort load of repo-root .env so OPENROUTER_API_KEY is available."""
    p = Path(__file__).resolve().parents[1] / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

import cv2  # noqa: E402
import mujoco as mj  # noqa: E402
import numpy as np  # noqa: E402

from vector_os_nano.core.scene_graph import SceneGraph  # noqa: E402
from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2  # noqa: E402
from vector_os_nano.hardware.sim.mujoco_piper import MuJoCoPiper  # noqa: E402
from vector_os_nano.hardware.sim.mujoco_piper_gripper import (  # noqa: E402
    MuJoCoPiperGripper,
)
from vector_os_nano.perception.go2_grasp_perception import Go2GraspPerception  # noqa: E402
from vector_os_nano.perception.object_localizer import localize_objects_3d  # noqa: E402

# tolerance (m) on the XY distance from a localized point to the nearest GT object.
# The grasp lands at ~7 cm (R3) and nav vicinity is 1.5 m, so for a scene-graph
# position to be "accurate" we want well under 20 cm; report the raw number too.
_XY_TOL_M = 0.20

# Queries to localize.  English class names work best with grounding-dino-tiny.
_QUERIES = ["green bottle", "blue bottle", "red can"]


def read_ground_truth(go2: MuJoCoGo2) -> dict[str, list[float]]:
    """Map free-body (graspable) name -> world [x,y,z] from live MjData.xpos."""
    model, data = go2._mj.model, go2._mj.data
    gt: dict[str, list[float]] = {}
    for i in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        if name is None:
            continue
        ja = model.body_jntadr[i]
        if ja >= 0 and model.jnt_type[ja] == mj.mjtJoint.mjJNT_FREE:
            gt[name] = [float(v) for v in data.xpos[i]]
    return gt


def nearest_gt(
    gt: dict[str, list[float]], x: float, y: float
) -> tuple[str, float]:
    """Return (gt_name, xy_distance) of the GT object nearest to (x,y)."""
    best_name, best_d = "", float("inf")
    for name, (gx, gy, _gz) in gt.items():
        d = ((gx - x) ** 2 + (gy - y) ** 2) ** 0.5
        if d < best_d:
            best_d, best_name = d, name
    return best_name, best_d


def main() -> int:
    out: dict = {"part1": {}, "part2": {}, "part3": {}, "gt": {}}

    go2 = MuJoCoGo2(gui=False, room=True, backend="mpc", with_arm=True)
    go2.connect()
    piper = MuJoCoPiper(go2)
    piper.connect()
    gripper = MuJoCoPiperGripper(go2)
    gripper.connect()
    perception = Go2GraspPerception(go2, width=320, height=240)

    # --- ground truth + camera/robot pose -----------------------------------
    gt = read_ground_truth(go2)
    out["gt"] = {k: [round(c, 3) for c in v] for k, v in gt.items()}
    rpos = go2.get_position()
    print(f"robot pos = ({rpos[0]:.2f}, {rpos[1]:.2f}, {rpos[2]:.2f})  heading={go2.get_heading():.3f}")
    print("GROUND TRUTH (free bodies):")
    for k, v in gt.items():
        print(f"  {k:28s} ({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})")

    # save the perception RGB frame for eyes-on verification
    try:
        rgb = perception.get_color_frame()
        depth = perception.get_depth_frame()
        cv2.imwrite("/tmp/verify_perception_rgb.png", rgb[:, :, ::-1])
        dvalid = depth[(depth > 0) & np.isfinite(depth)]
        out["frame"] = {
            "rgb_shape": list(rgb.shape),
            "depth_valid_frac": round(float(dvalid.size) / float(depth.size), 3),
            "depth_min_m": round(float(dvalid.min()), 3) if dvalid.size else None,
            "depth_max_m": round(float(dvalid.max()), 3) if dvalid.size else None,
        }
        print(f"perception frame: rgb={rgb.shape} depth_valid={out['frame']['depth_valid_frac']} "
              f"range=[{out['frame']['depth_min_m']},{out['frame']['depth_max_m']}]m -> /tmp/verify_perception_rgb.png")
    except Exception as exc:  # noqa: BLE001
        print(f"[frame dump failed] {exc}")

    # ====================================================================
    # PART 1 — localizer accuracy vs ground truth (deterministic, headline)
    # ====================================================================
    print("\n=== PART 1: localize_objects_3d vs GT ===")
    loc = localize_objects_3d(perception, _QUERIES)
    p1_rows = []
    p1_pass = bool(loc)
    for label, x, y, z in loc:
        gname, gd = nearest_gt(gt, x, y)
        gz = gt.get(gname, [0, 0, 0])[2]
        ok = gd <= _XY_TOL_M
        p1_pass = p1_pass and ok
        row = {
            "query_label": label,
            "world": [round(x, 3), round(y, 3), round(z, 3)],
            "nearest_gt": gname,
            "xy_err_m": round(gd, 3),
            "z_err_m": round(abs(z - gz), 3),
            "within_tol": ok,
        }
        p1_rows.append(row)
        print(f"  {label:14s} -> ({x:.3f},{y:.3f},{z:.3f})  nearest={gname} "
              f"xy_err={gd:.3f}m z_err={abs(z-gz):.3f}m  {'OK' if ok else 'FAR'}")
    if not loc:
        print("  localize_objects_3d returned [] — localizer produced NOTHING")
    out["part1"] = {"rows": p1_rows, "pass": p1_pass, "n_localized": len(loc)}

    # ====================================================================
    # PART 2 — scene-graph storage round-trip (deterministic, no VLM)
    # ====================================================================
    print("\n=== PART 2: scene-graph store + persist round-trip ===")
    tmp_path = tempfile.mktemp(suffix="_sg.yaml")
    sg = SceneGraph(persist_path=tmp_path)
    detected = [(label, x, y) for (label, x, y, _z) in loc]
    names = [d[0] for d in detected]
    sg.observe_with_viewpoint(
        "kitchen", float(rpos[0]), float(rpos[1]), float(go2.get_heading()),
        names, "table scene", detected_objects=detected,
    )
    sg.save()
    sg2 = SceneGraph(persist_path=tmp_path)
    sg2.load()
    p2_rows = []
    p2_pass = bool(detected)
    for label, x, y in detected:
        found = sg2.find_objects_by_category(label)
        match = found[0] if found else None
        stored = [round(match.x, 3), round(match.y, 3)] if match else None
        ok = bool(match) and abs(match.x - x) < 1e-6 and abs(match.y - y) < 1e-6
        p2_pass = p2_pass and ok
        p2_rows.append({"label": label, "fed": [round(x, 3), round(y, 3)],
                        "stored_after_reload": stored, "round_trip_ok": ok})
        print(f"  {label:14s} fed=({x:.3f},{y:.3f}) reloaded={stored}  {'OK' if ok else 'MISMATCH'}")
    try:
        os.remove(tmp_path)
    except OSError:
        pass
    out["part2"] = {"rows": p2_rows, "pass": p2_pass}

    # ====================================================================
    # PART 3 — full `look` skill end-to-end (the path Yusen asked about)
    # ====================================================================
    # Run the REAL LookSkill (look -> localize_objects_3d -> observe_with_viewpoint
    # -> scene graph) directly with a hand-built context. This isolates the #1
    # wiring from the planner/executor (which runs an unrelated `home` step with a
    # pre-existing 5-vs-6 joint bug on the Piper arm).
    #
    # Two variants:
    #   3a) real GPT-4o VLM (best-effort; may fail on network/SSL — naming only).
    #   3b) stub VLM: the network-blocked object *naming* is stubbed, but the
    #       localization (the actual #1 fix) stays fully REAL — grounding-dino +
    #       depth + GT comparison. Proves the full look->scene-graph path.
    from vector_os_nano.core.skill import SkillContext
    from vector_os_nano.skills.go2.look import LookSkill
    from vector_os_nano.perception.vlm_go2 import (
        DetectedObject, RoomIdentification, SceneDescription,
    )

    class _StubVLM:
        """VLM whose object naming is hardcoded; everything downstream is real."""

        def describe_scene(self, frame):
            objs = [
                DetectedObject(name="green bottle", description="", confidence=0.9),
                DetectedObject(name="blue bottle", description="", confidence=0.9),
                DetectedObject(name="red can", description="", confidence=0.9),
            ]
            return SceneDescription(
                summary="three containers on a table", objects=objs,
                room_type="kitchen", details="",
            )

        def identify_room(self, frame):
            return RoomIdentification(room="kitchen", confidence=0.9, reasoning="stub")

    def run_look(vlm_obj, tag: str) -> dict:
        print(f"\n=== PART 3{tag}: `look` skill -> scene graph (vlm={type(vlm_obj).__name__}) ===")
        sg_path = tempfile.mktemp(suffix=f"_sg3{tag}.yaml")
        sg_l = SceneGraph(persist_path=sg_path)
        ctx = SkillContext(
            bases={"default": go2},
            perception_sources={"default": perception},
            services={"vlm": vlm_obj, "spatial_memory": sg_l},
        )
        try:
            res = LookSkill().execute({}, ctx)
        except Exception as exc:  # noqa: BLE001
            print(f"  look raised: {exc}")
            return {"ran": True, "look_success": False, "error": str(exc)}
        ok_exec = getattr(res, "success", None)
        rd = getattr(res, "result_data", None) or {}
        vlm_objs = rd.get("objects", [])
        print(f"  look success={ok_exec} room={rd.get('room')} "
              f"diag={getattr(res, 'diagnosis_code', None)} vlm_objects={len(vlm_objs)}")
        for o in vlm_objs:
            wx = o.get("world_x")
            tag2 = (f" world=({wx:.3f},{o['world_y']:.3f},{o['world_z']:.3f})"
                    if wx is not None else " (no world coord)")
            print(f"    - {o.get('name')!r}{tag2}")
        rows, seen = [], set()
        for cat in {o.get("name", "") for o in vlm_objs}:
            for node in sg_l.find_objects_by_category(cat):
                if node.object_id in seen:
                    continue
                seen.add(node.object_id)
                has_coord = abs(node.x) > 1e-3 or abs(node.y) > 1e-3
                gname, gd = nearest_gt(gt, node.x, node.y) if has_coord else ("", None)
                rows.append({"category": node.category,
                             "coord": [round(node.x, 3), round(node.y, 3)],
                             "has_real_coord": has_coord, "nearest_gt": gname,
                             "xy_err_m": round(gd, 3) if gd is not None else None})
                print(f"    graph: {node.category!r} ({node.x:.3f},{node.y:.3f}) "
                      f"real={has_coord} nearest={gname} err={gd}")
        try:
            os.remove(sg_path)
        except OSError:
            pass
        n_real = sum(1 for r in rows if r["has_real_coord"])
        in_tol = all(r["xy_err_m"] is not None and r["xy_err_m"] <= _XY_TOL_M for r in rows)
        return {"ran": True, "look_success": ok_exec, "room": rd.get("room"),
                "n_vlm_objects": len(vlm_objs), "n_graph_objects": len(rows),
                "n_with_real_coord": n_real, "rows": rows,
                "all_accurate": bool(rows) and in_tol}

    # 3a — real VLM (best effort)
    try:
        from vector_os_nano.perception.vlm_go2 import Go2VLMPerception
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        real_vlm = Go2VLMPerception(config={"api_key": api_key}) if api_key else None
        out["part3a_real_vlm"] = (
            run_look(real_vlm, "a") if real_vlm is not None
            else {"ran": False, "error": "no OPENROUTER_API_KEY"}
        )
    except Exception as exc:  # noqa: BLE001
        out["part3a_real_vlm"] = {"ran": False, "error": str(exc)}

    # 3b — stub VLM naming, REAL localization (always runs; the wiring proof)
    out["part3"] = run_look(_StubVLM(), "b")

    # --- teardown -----------------------------------------------------------
    for dev in (gripper, piper, go2):
        try:
            dev.disconnect()
        except Exception:  # noqa: BLE001
            pass

    out["overall_pass"] = bool(
        out["part1"].get("pass")
        and out["part2"].get("pass")
        and out["part3"].get("all_accurate")
    )
    print("\nRESULT " + json.dumps(out))
    return 0 if out["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
