"""StopSkill -- emergency stop, immediately halts all movement.

Calls set_velocity(0, 0, 0) on the base and publishes a zero-velocity
Twist to /cmd_vel_nav via ROS2 (best-effort, lazy import).
"""
from __future__ import annotations

import logging

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)


@skill(
    aliases=["stop", "停", "停下", "halt", "freeze", "别动", "停止"],
    direct=True,
)
class StopSkill:
    """Emergency stop -- immediately halt all movement."""

    name: str = "stop"
    description: str = "Emergency stop -- immediately halt all movement."
    parameters: dict = {}
    preconditions: list[str] = []
    postconditions: list[str] = ["is_stopped"]
    effects: dict = {"is_moving": False}
    failure_modes: list[str] = ["no_base"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        """Halt all movement.

        Args:
            params: Unused.
            context: SkillContext with base attached.

        Returns:
            SkillResult(success=True) after stop commands are issued.
            SkillResult(success=False, diagnosis_code="no_base") if base missing.
        """
        if context.base is None:
            logger.error("[STOP] No base connected")
            return SkillResult(
                success=False,
                error_message="No base connected",
                diagnosis_code="no_base",
            )

        # Signal global abort — stops VGG, GoalExecutor, navigate, explore
        try:
            from vector_os_nano.vcli.cognitive.abort import request_abort
            request_abort()
            logger.info("[STOP] Global abort signal sent")
        except Exception:
            pass

        # Cancel any background exploration
        try:
            from vector_os_nano.skills.go2.explore import cancel_exploration, is_exploring
            if is_exploring():
                cancel_exploration()
                logger.info("[STOP] Background exploration cancelled")
        except Exception:
            pass

        # Hard stop via base interface
        try:
            context.base.set_velocity(0.0, 0.0, 0.0)
            logger.info("[STOP] set_velocity(0, 0, 0) issued")
        except Exception as exc:
            logger.warning("[STOP] set_velocity failed: %s", exc)

        # Best-effort: publish zero Twist to /cmd_vel_nav via ROS2
        _try_publish_zero_cmdvel()

        return SkillResult(
            success=True,
            result_data={"stopped": True},
        )


def _try_publish_zero_cmdvel() -> None:
    """Publish a zero-velocity Twist to /cmd_vel_nav (ROS2, best-effort).

    All ROS2 imports are lazy so this module works without ROS2.
    Errors are swallowed -- stop must succeed even if ROS2 is absent.
    """
    try:
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import Twist

        if not rclpy.ok():
            return

        # Use a temporary one-shot publisher on an anonymous node
        node = Node("_stop_skill_oneshot")
        try:
            pub = node.create_publisher(Twist, "/cmd_vel_nav", 1)
            msg = Twist()
            # All fields default to 0.0 -- zero velocity
            pub.publish(msg)
            logger.info("[STOP] Published zero Twist to /cmd_vel_nav")
        finally:
            node.destroy_node()

    except Exception as exc:
        logger.debug("[STOP] ROS2 zero-vel publish skipped: %s", exc)
