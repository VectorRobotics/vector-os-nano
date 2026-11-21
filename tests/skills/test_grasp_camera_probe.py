# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Opt-in raw sensor evidence for a failed grasp, without actor truth access."""
from vector_os_nano.vcli.worlds.sim_oracles import session_namespace
import json
import os
import random
from pathlib import Path

import pytest
from PIL import Image

from vector_os_nano.sim.session import SimSession, SimSpec


@pytest.mark.sim
@pytest.mark.skipif(not os.environ.get("VECTOR_GRASP_EVIDENCE"), reason="set VECTOR_GRASP_EVIDENCE for the camera probe")
def test_grasp_camera_probe():
    out = Path(os.environ["VECTOR_GRASP_EVIDENCE"])
    out.mkdir(parents=True, exist_ok=True)
    s = SimSession.start(SimSpec("go2_piper", gui=False))
    frames, moves = [], []
    try:
        perception = s.agent._perception
        capture = perception.capture
        def record():
            f = capture()
            name = f"frame-{len(frames):03d}.png"
            Image.fromarray(f.rgb).save(out / name)
            frames.append({"file": name, "time": f.sim_time, "position": f.cam_pos.tolist(),
                           "matrix": f.cam_mat.tolist(), "depth_min": float(f.depth[f.depth > 0].min())})
            return f
        perception.capture = record
        if os.environ.get("VECTOR_GRASP_SEED"):
            rng = random.Random(int(os.environ["VECTOR_GRASP_SEED"]))
            back, side_dir, side = rng.uniform(0., .25), rng.choice(["left", "right"]), rng.uniform(0., .1)
            turn_dir, angle = rng.choice(["left", "right"]), rng.uniform(0., 12.)
            for name, params in [("walk", {"direction": "backward", "distance": round(back, 3)}),
                                 ("walk", {"direction": side_dir, "distance": round(side, 3)}),
                                 ("turn", {"direction": turn_dir, "angle": round(angle, 1)})]:
                assert s.agent.execute_skill(name, params).success
        arm = s.agent._arm
        move = arm.move_joints
        def record_move(q, duration):
            ok = move(q, duration=duration)
            moves.append({"q": q, "duration": duration, "ok": ok,
                          "actual_q": arm.get_joint_positions(), "ee": arm.fk(arm.get_joint_positions())[0],
                          "objects": s.gt.object_positions(), "base": s.gt.base_pose(),
                          "clearance": s.gt.clearance("pickable_bottle_green"),
                          "contacts": sorted(s.gt.contacts("pickable_bottle_green"))})
            return ok
        arm.move_joints = record_move
        result = s.agent.execute_skill("grasp", {"query": "green bottle"})
        evidence = {"frames": frames, "tool_success": result.success, "error": result.error_message,
                    "data": result.result_data, "base": s.gt.base_pose(), "objects": s.gt.object_positions(),
                    "attachments": s.gt.attachments(), "moves": moves}
        (out / "result.json").write_text(json.dumps(evidence, indent=2))
        assert result.success, result.error_message
        assert session_namespace(s)["holding_object"]("pickable_bottle_green")
    finally:
        s.stop()
