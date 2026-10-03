# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Single-source go2+arm manipulation wiring.

``SimSession`` (the one in-process launcher behind NL ``start_simulation`` and
the ``--sim-go2``/``--embodiment`` flags) registers the Piper manipulation skill
set + perception-grasp here, once, for every go2+arm embodiment (Rule 3 / 11).

The arm/gripper AND the RGB-D perception are attached to the agent BEFORE this is
called: the launcher builds ONE perception at the manifest camera size, and this
helper reuses it, so look/localize and the grasp can never run at different
resolutions (the old helper replaced it with a hardcoded 320x240 one).
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def register_manipulation_skills(agent: Any, *, enable_env: bool = True) -> bool:
    """Register the Piper manipulation skills + perception-grasp on ``agent``.

    Args:
        agent: the constructed Agent (``_arm``/``_gripper``/``_perception`` attached).
        enable_env: honor the ``VECTOR_ENABLE_MANIPULATION=0`` opt-out (default on).

    Raises RuntimeError when the agent has no perception: perception_grasp would
    have nothing to see with, and a silent default size is exactly the drift this
    helper exists to prevent.

    Returns ``True`` when the manipulation stack was wired, ``False`` when skipped
    via the env opt-out. ``perception_grasp`` is registered LAST so it wins the
    shared 抓/grab aliases on the empty-world-model path (it needs no pre-populated
    world model; ``pick_top_down`` does).
    """
    if enable_env and os.environ.get("VECTOR_ENABLE_MANIPULATION", "1") == "0":
        logger.info("[manip] VECTOR_ENABLE_MANIPULATION=0 — manipulation skills skipped")
        return False
    if getattr(agent, "_perception", None) is None:
        raise RuntimeError("perception_grasp needs agent._perception (built once by the launcher)")

    from vector_os_nano.skills.pick_top_down import PickTopDownSkill
    from vector_os_nano.skills.place_top_down import PlaceTopDownSkill
    from vector_os_nano.skills.mobile_pick import MobilePickSkill
    from vector_os_nano.skills.mobile_place import MobilePlaceSkill

    agent._skill_registry.register(PickTopDownSkill())
    agent._skill_registry.register(PlaceTopDownSkill())
    agent._skill_registry.register(MobilePickSkill())
    agent._skill_registry.register(MobilePlaceSkill())

    # Perception-driven grasp (the honest North-Star path): real RGB-D from the
    # go2 head camera -> VLM/colour resolve -> EdgeTAM mask -> rendered-depth
    # pointcloud -> 3D grasp point (NEVER ground truth) -> IK -> weld.
    from vector_os_nano.skills.perception_grasp import PerceptionGraspSkill

    agent._skill_registry.register(PerceptionGraspSkill())
    logger.info("[manip] perception-grasp wired over %s", type(agent._perception).__name__)
    return True
