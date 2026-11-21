# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""StopSkill -- emergency stop for any body: halts the base and the arm it has.

The base gets its ungated stop() (else set_velocity(0, 0, 0)); the arm its stop().
Offered on every body with a base or an arm, so the P0 stop phrase stops them all.
"""
from __future__ import annotations

import logging

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)


@skill(requires=(),
    aliases=["stop", "停", "停下", "halt", "freeze", "别动", "停止"],
)
class StopSkill:
    """Emergency stop -- immediately halt all movement of the base and the arm."""

    name: str = "stop"
    description: str = "Emergency stop -- immediately halt all movement (base and arm)."
    parameters: dict = {}
    preconditions: list[str] = []
    postconditions: list[str] = ["is_stopped"]
    effects: dict = {"is_moving": False}
    failure_modes: list[str] = ["nothing_to_stop", "stop_failed"]
    # Housekeeping: halting changes no task state, so an unproven stop row does not
    # fail a turn another row proved — unless it re-verifies (supersedes) an actuating
    # attempt, which it must then prove (trace_store.turn_reject_reason).
    housekeeping: bool = True

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        """Halt the base and the arm; report which halted and which failed."""
        base, arm = context.base, context.arm
        groups = context.services.get('joint_groups', {})
        if base is None and arm is None and not groups:
            return SkillResult(
                success=False,
                error_message="Nothing to stop: no base and no arm connected",
                diagnosis_code="nothing_to_stop",
            )

        # Stop every other run (navigate, explore) — never the run this stop is a
        # step of; the owner's P0 stop has cancelled every run already.
        try:
            from vector_os_nano.vcli.cognitive.abort import request_abort
            request_abort(spare_caller=True)
        except Exception:  # noqa: BLE001 — halting the motors below is what matters
            pass
        try:
            from vector_os_nano.skills.go2.explore import cancel_exploration, is_exploring
            if is_exploring():
                cancel_exploration()
                logger.info("[STOP] Background exploration cancelled")
        except Exception:  # noqa: BLE001
            pass

        halted: list[str] = []
        failed: dict[str, str] = {}
        if base is not None:
            # stop() is ungated: set_velocity from this thread is IGNORED while
            # another thread's walk/navigate holds the skill token.
            try:
                if callable(getattr(base, "stop", None)):
                    base.stop()
                else:
                    base.set_velocity(0.0, 0.0, 0.0)
                halted.append("base")
            except Exception as exc:  # noqa: BLE001
                failed["base"] = str(exc)
        for name, device in [('arm', arm), *[(f'joint_group:{n}', d) for n, d in groups.items()]]:
            if device is None:
                continue
            try:
                device.stop()
                halted.append(name)
            except Exception as exc:  # noqa: BLE001
                failed[name] = str(exc)
        logger.info("[STOP] halted=%s failed=%s", halted, failed)

        if failed:
            return SkillResult(
                success=False,
                error_message=f"Stop failed for {sorted(failed)}: {failed}",
                diagnosis_code="stop_failed",
                result_data={"stopped": halted, "failed": failed},
            )
        return SkillResult(success=True, result_data={"stopped": halted})
