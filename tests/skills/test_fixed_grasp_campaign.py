# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Paired reset seeds; old SO-101 truth perception is labelled, never a physical demo."""
import json
import os
import random
from pathlib import Path

import pytest

from vector_os_nano.sim.session import SimSession, SimSpec
from vector_os_nano.vcli.cognitive import abort

N = int(os.environ.get("VECTOR_GRASP_CAMPAIGN", "0"))


@pytest.mark.sim
@pytest.mark.skipif(not N, reason="set VECTOR_GRASP_CAMPAIGN=N")
@pytest.mark.parametrize("body,targets", [("piper", ["red_box"]), ("so101", ["banana", "bottle"])])
def test_fixed_grasp_campaign(body, targets):
    rows = []
    for seed in range(N):
        abort.clear_abort()
        s = SimSession.start(SimSpec(body, gui=False))
        target = targets[seed % len(targets)]
        try:
            rng = random.Random(seed)
            dx, dy = rng.uniform(-.01, .01), rng.uniform(-.01, .01)
            def reset(w):
                m = w.model
                j = m.body_jntadr[w.scene.body(target)]
                adr = m.jnt_qposadr[j]
                q = w.state()[0].copy()
                q[adr] += dx
                q[adr + 1] += dy
                w.reset(q)
            s.runtime.call(reset)
            s.runtime.wait_sim(.5)
            skill = os.environ.get("VECTOR_GRASP_TOOL") or ("pick" if body == "so101" else "perception_grasp")
            params = {"object_label": target} if skill == "pick" else {"query": target}
            result = s.agent.execute_skill(skill, params)
            physical = bool(s.oracles()["holding_object"](target))
            rows.append({"seed": seed, "target": target, "offset": [dx, dy], "skill": skill,
                         "perception": "sim_truth" if skill == "pick" else "camera",
                         "grasp_mode": "contact_gated_constraint_assist", "physical": physical,
                         "attachments": s.gt.attachments(), "objects": s.gt.object_positions(),
                         "contacts": sorted(s.gt.contacts(target)), "clearance": s.gt.clearance(target),
                         "tool_success": result.success, "data": result.result_data,
                         "error": result.error_message})
        except Exception as exc:
            rows.append({"seed": seed, "target": target, "physical": False, "error": str(exc)})
        finally:
            s.stop()
    output = Path(os.environ["VECTOR_CAMPAIGN_OUT"])
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{body}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    passed = sum(r["physical"] for r in rows)
    assert passed == N, f"{body}: {passed}/{N}; full denominator and failures: {output / (body + '.json')}"
