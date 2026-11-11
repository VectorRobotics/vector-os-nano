# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Vector OS Nano Agent — the main entry point.

This is the ONE class users interact with. It wires together:
- Hardware (arm + gripper)
- Perception pipeline
- Skill registry
- World model
"""
from __future__ import annotations

import logging
from typing import Any

from vector_os_nano.core.config import load_config
from vector_os_nano.core.skill import Skill, SkillContext, SkillRegistry
from vector_os_nano.core.types import SkillResult
from vector_os_nano.core.world_model import WorldModel

logger = logging.getLogger(__name__)


class Agent:
    """Robot arm / mobile base control via structured skill calls.

    Usage::

        from vector_os_nano import Agent, SO101
        arm = SO101(port="/dev/ttyACM0")
        agent = Agent(arm=arm)
        agent.execute_skill("pick", {"object_label": "red cup"})

    All hardware arguments are optional — Agent degrades gracefully when
    components are absent (useful for unit testing and partial deployments).

    Args:
        arm: Object implementing ArmProtocol.  None disables motion.
        gripper: Object implementing GripperProtocol.  If None and arm is an
            SO101Arm (has ``_bus``), a SO101Gripper is created automatically.
        perception: Object implementing PerceptionProtocol.  None disables
            vision-based skills.
        skills: Additional Skill instances to register alongside the built-in
            defaults.
        config: Configuration override — a dict, a path to a YAML file, or
            None to use the built-in defaults.
        auto_perception: When True and perception=None and config specifies
            ``camera.type == "realsense"``, automatically construct a
            RealSenseCamera + VLMDetector + EdgeTAMTracker + PerceptionPipeline.
            Defaults to False because loading the VLM and tracker is slow and
            consumes GPU memory — callers that want auto-setup must opt in
            explicitly.
    """

    def __init__(
        self,
        arm: Any = None,
        gripper: Any = None,
        perception: Any = None,
        skills: list[Skill] | None = None,
        config: dict | str | None = None,
        auto_perception: bool = False,
        base: Any = None,
    ) -> None:
        # ---- Configuration --------------------------------------------------
        if isinstance(config, dict):
            self._config: dict = config
        else:
            self._config = load_config(config)

        # ---- Hardware -------------------------------------------------------
        self._arm = arm
        self._gripper = gripper
        self._base = base

        # Auto-create SO101Gripper when arm shares a serial bus and no gripper
        # was explicitly supplied.
        if gripper is None and arm is not None and hasattr(arm, "_bus") and arm._bus is not None:
            try:
                from vector_os_nano.hardware.so101.gripper import SO101Gripper  # lazy

                self._gripper = SO101Gripper(arm._bus)
            except Exception as exc:
                logger.warning("Could not auto-create SO101Gripper: %s", exc)

        # ---- Perception -----------------------------------------------------
        self._perception = perception

        # Auto-create perception pipeline when caller opts in and no pipeline
        # was supplied explicitly.  Heavy imports (torch, transformers, pyrealsense2)
        # are deferred to here so normal Agent construction stays fast.
        if auto_perception and self._perception is None:
            cam_type = self._config.get("camera", {}).get("type", "")
            if cam_type == "realsense":
                try:
                    from vector_os_nano.perception.realsense import RealSenseCamera
                    from vector_os_nano.perception.vlm import VLMDetector
                    from vector_os_nano.perception.tracker import EdgeTAMTracker
                    from vector_os_nano.perception.pipeline import PerceptionPipeline

                    cam = RealSenseCamera(serial=self._config.get("camera", {}).get("serial") or None)
                    cam.connect()
                    vlm_det = VLMDetector()
                    tracker = EdgeTAMTracker()
                    self._perception = PerceptionPipeline(
                        camera=cam, vlm=vlm_det, tracker=tracker
                    )
                    logger.info("Auto-created perception pipeline (RealSense + VLM + tracker)")
                except Exception as exc:
                    logger.warning("Could not auto-create perception pipeline: %s", exc)

        # ---- World model ----------------------------------------------------
        self._world_model = WorldModel()

        # ---- Skill registry -------------------------------------------------
        self._skill_registry = SkillRegistry()

        # Register built-in skills
        from vector_os_nano.skills import get_default_skills  # lazy-ish (already imported by wave 2)

        for skill in get_default_skills():
            self._skill_registry.register(skill)

        # Register caller-supplied custom skills
        if skills:
            for skill in skills:
                self._skill_registry.register(skill)

        # ---- Lazy-init state ------------------------------------------------
        self._ik_solver: Any = None
        self._calibration: Any = None

        # ---- Sync world model with hardware ----------------------------------
        self._sync_robot_state()

    # -------------------------------------------------------------------------
    # Public API — primary entry point
    # -------------------------------------------------------------------------

    def _sync_robot_state(self) -> None:
        """Sync world model robot state from hardware."""
        if self._arm is not None:
            try:
                joints = self._arm.get_joint_positions()
                ee_pos = (0.0, 0.0, 0.0)
                try:  # the arm's OWN kinematics, never another arm's solver
                    pos, _ = self._arm.fk(joints)
                    ee_pos = tuple(pos)
                except Exception:
                    pass
                self._world_model.update_robot_state(
                    joint_positions=tuple(joints),
                    ee_position=ee_pos,
                )
            except Exception as exc:
                logger.debug("Could not sync robot state: %s", exc)

        if self._base is not None:
            try:
                pos = self._base.get_position()
                heading = self._base.get_heading()
                self._world_model.update_robot_state(
                    position_xy=(pos[0], pos[1]),
                    heading=heading,
                )
            except Exception as exc:
                logger.debug("Could not sync base state: %s", exc)

    # -------------------------------------------------------------------------
    # Convenience methods
    # -------------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to all hardware peripherals."""
        if self._arm is not None:
            self._arm.connect()
        if self._perception is not None and hasattr(self._perception, "connect"):
            self._perception.connect()

    def disconnect(self) -> None:
        """Disconnect from all hardware peripherals."""
        if self._arm is not None:
            self._arm.disconnect()
        if self._perception is not None and hasattr(self._perception, "disconnect"):
            self._perception.disconnect()

    def execute_skill(self, skill_name: str, params: dict | None = None) -> SkillResult:
        """Run ONE registered skill with structured parameters and return its own result.

        No plan, no prerequisite expansion, nothing appended: a skill that needs a
        prerequisite calls it itself, and the caller (the model) sequences the rest.
        """
        skill_obj = self._skill_registry.get(skill_name)
        if skill_obj is None:
            return SkillResult(
                success=False,
                error_message=f"Unknown skill: {skill_name}",
                result_data={"diagnosis": "unknown_skill"},
            )
        result = skill_obj.execute(dict(params or {}), self._build_context())
        self._sync_robot_state()
        return result

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def world(self) -> WorldModel:
        """The current world model."""
        return self._world_model

    @property
    def skills(self) -> list[str]:
        """Names of all currently registered skills."""
        return self._skill_registry.list_skills()

    # -------------------------------------------------------------------------
    # Context manager
    # -------------------------------------------------------------------------

    def __enter__(self) -> "Agent":
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.disconnect()

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _build_context(self) -> SkillContext:
        """Build a SkillContext from current agent state.

        IK solver and calibration are initialised lazily on first call.

        Returns:
            SkillContext bundling all resources skills need during execution.
        """
        # Lazy-init the SO-101 IK solver, ONLY for an arm that takes an external
        # solver (SO101Arm); every other arm carries its own ik/fk. A failure is
        # cached (False) so it is not retried on every call.
        if self._ik_solver is None and hasattr(self._arm, "set_ik_solver"):
            try:
                from vector_os_nano.hardware.so101.ik_solver import IKSolver  # lazy

                self._ik_solver = IKSolver()
                self._arm.set_ik_solver(self._ik_solver)
            except Exception as exc:
                self._ik_solver = False
                logger.debug("IK solver not available: %s", exc)

        # Lazy-init the camera->arm calibration of a fixed (base-less) arm. A mobile
        # base never gets an identity one: its perception carries the camera pose.
        if self._calibration is None and self._base is None:
            try:
                from vector_os_nano.perception.calibration import Calibration  # lazy

                cal_file: str = self._config.get("calibration", {}).get("file", "")
                if cal_file:
                    self._calibration = Calibration.load(cal_file)  # relative to cwd
                else:
                    self._calibration = Calibration()
            except Exception as exc:
                logger.debug("Calibration not available: %s", exc)

        # Build services dict for skills that need injected dependencies
        services: dict = {}
        if hasattr(self, "_vlm") and self._vlm is not None:
            services["vlm"] = self._vlm
        if hasattr(self, "_spatial_memory") and self._spatial_memory is not None:
            services["spatial_memory"] = self._spatial_memory
        if hasattr(self, "_skill_registry"):
            services["skill_registry"] = self._skill_registry

        return SkillContext(
            arms={"default": self._arm} if self._arm else {},
            grippers={"default": self._gripper} if self._gripper else {},
            bases={"default": self._base} if self._base else {},
            perception_sources=(
                {"default": self._perception} if self._perception else {}
            ),
            services=services,
            world_model=self._world_model,
            calibration=self._calibration,
            config=self._config,
        )
