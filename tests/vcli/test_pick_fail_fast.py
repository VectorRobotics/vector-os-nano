# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Pick fails FAST when the target is genuinely absent (no wasted retries).

Asking to grab an object that isn't in the scene used to exhaust all pick
retries — re-homing and re-detecting the same miss each time (seconds of motion
for a guaranteed failure). The pick skill now breaks out on a target-not-found
diagnosis (object_not_found / no_detections); transient failures still retry.
"""
from __future__ import annotations

import pytest

from tests.harness.sim_session import started

pytestmark = pytest.mark.sim


def _pick(label: str, max_retries: int):
    from vector_os_nano.skills.pick import PickSkill

    with started("so101") as s:
        s.agent._config["skills"]["pick"]["max_retries"] = max_retries
        return PickSkill().execute({"object_label": label, "mode": "hold"}, s.agent._build_context())


def test_absent_target_fails_after_one_attempt_not_max_retries():
    """An object not in the scene must fail FAST (1 attempt), not exhaust max_retries=3."""
    result = _pick("apple", max_retries=3)  # apple absent
    assert result.success is False
    assert result.result_data.get("diagnosis") == "object_not_found"
    # Fail-fast: stopped after the first attempt despite max_retries=3.
    assert result.result_data.get("attempts") == 1, (
        f"absent target should fail fast, not retry: {result.result_data}"
    )


def test_present_target_still_succeeds():
    """Sanity: the fail-fast change does not affect a normal, present target."""
    assert _pick("banana", max_retries=2).success is True
