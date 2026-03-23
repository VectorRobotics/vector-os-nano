"""Vector OS Nano Agent — the main entry point.

This is the ONE class users interact with. It wires together:
- Hardware (arm + gripper)
- Perception pipeline
- LLM planner
- Skill registry
- World model
- Task executor
"""
from __future__ import annotations

import logging
from typing import Any

from vector_os_nano.core.config import load_config
from vector_os_nano.core.executor import TaskExecutor
from vector_os_nano.core.skill import Skill, SkillContext, SkillMatch, SkillRegistry
from vector_os_nano.core.types import ExecutionResult
from vector_os_nano.core.world_model import WorldModel

logger = logging.getLogger(__name__)


class Agent:
    """Natural language robot arm control.

    Usage::

        from vector_os_nano import Agent, SO101
        arm = SO101(port="/dev/ttyACM0")
        agent = Agent(arm=arm, llm_api_key="sk-...")
        agent.execute("pick up the red cup")

    All hardware arguments are optional — Agent degrades gracefully when
    components are absent (useful for unit testing and partial deployments).

    Args:
        arm: Object implementing ArmProtocol.  None disables motion.
        gripper: Object implementing GripperProtocol.  If None and arm is an
            SO101Arm (has ``_bus``), a SO101Gripper is created automatically.
        perception: Object implementing PerceptionProtocol.  None disables
            vision-based skills.
        llm: An LLMProvider instance.  If None and llm_api_key is given, a
            ClaudeProvider is created automatically.
        llm_api_key: Convenience shorthand — ignored when llm= is provided.
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
        llm: Any = None,
        llm_api_key: str | None = None,
        skills: list[Skill] | None = None,
        config: dict | str | None = None,
        auto_perception: bool = False,
    ) -> None:
        # ---- Configuration --------------------------------------------------
        if isinstance(config, dict):
            self._config: dict = config
        else:
            self._config = load_config(config)

        # ---- Hardware -------------------------------------------------------
        self._arm = arm
        self._gripper = gripper

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

                    cam = RealSenseCamera()
                    cam.connect()
                    vlm_det = VLMDetector()
                    tracker = EdgeTAMTracker()
                    self._perception = PerceptionPipeline(
                        camera=cam, vlm=vlm_det, tracker=tracker
                    )
                    logger.info("Auto-created perception pipeline (RealSense + VLM + tracker)")
                except Exception as exc:
                    logger.warning("Could not auto-create perception pipeline: %s", exc)

        # ---- LLM provider ---------------------------------------------------
        # llm= takes priority over llm_api_key=
        self._llm = llm
        if llm is None and llm_api_key:
            try:
                from vector_os_nano.llm.claude import ClaudeProvider  # lazy

                llm_cfg = self._config.get("llm", {})
                self._llm = ClaudeProvider(
                    api_key=llm_api_key,
                    model=llm_cfg.get("model", "anthropic/claude-sonnet-4-6"),
                    api_base=llm_cfg.get("api_base", "https://openrouter.ai/api/v1"),
                )
            except Exception as exc:
                logger.warning("Could not create ClaudeProvider: %s", exc)

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

        # ---- Executor -------------------------------------------------------
        self._executor = TaskExecutor()

        # ---- Lazy-init state ------------------------------------------------
        self._ik_solver: Any = None
        self._calibration: Any = None
        self._conversation_history: list[dict] = []

        # ---- Sync world model with hardware ----------------------------------
        self._sync_robot_state()

    # -------------------------------------------------------------------------
    # Public API — primary entry point
    # -------------------------------------------------------------------------

    def _sync_robot_state(self) -> None:
        """Sync world model robot state from hardware."""
        if self._arm is None:
            return
        try:
            joints = self._arm.get_joint_positions()
            ee_pos = (0.0, 0.0, 0.0)
            if self._ik_solver is not None:
                try:
                    pos, _ = self._ik_solver.fk(joints)
                    ee_pos = tuple(pos)
                except Exception:
                    pass
            self._world_model.update_robot_state(
                joint_positions=tuple(joints),
                ee_position=ee_pos,
            )
        except Exception as exc:
            logger.debug("Could not sync robot state: %s", exc)

    def _refresh_objects(self) -> None:
        """Populate world model with current objects from sim or perception.

        Ensures the LLM planner can see what objects are on the table
        even if previous operations cleared the world model.
        """
        if self._arm is None:
            return

        # From MuJoCo sim: get_object_positions() returns ground truth
        if hasattr(self._arm, "get_object_positions"):
            try:
                from vector_os_nano.core.world_model import ObjectState
                objs = self._arm.get_object_positions()
                for name, pos in objs.items():
                    obj = ObjectState(
                        object_id=name,
                        label=name.replace("_", " "),
                        x=float(pos[0]),
                        y=float(pos[1]),
                        z=float(pos[2]),
                        state="on_table",
                        confidence=1.0,
                    )
                    self._world_model.add_object(obj)
            except Exception as exc:
                logger.debug("Could not refresh objects from sim: %s", exc)

    def execute(self, instruction: str, on_message: Any = None, on_step: Any = None, on_step_done: Any = None) -> ExecutionResult:
        """Execute a natural language instruction via multi-stage pipeline.

        Stage 1: CLASSIFY — determine intent (chat/task/direct/query)
        Stage 2: ROUTE — chat→LLM response, direct→immediate skill, task→plan
        Stage 3: PLAN — LLM decomposes into skill sequence + user message
        Stage 4: EXECUTE — run skills step by step
        Stage 5: ADAPT — on failure, retry with context
        Stage 6: SUMMARIZE — LLM generates user-friendly result summary

        Args:
            instruction: Human-readable command string.
            on_message: Optional callback(str) — called with AI message BEFORE execution.
            on_step: Optional callback(step_name, step_idx, total) — called before each step.

        Returns:
            ExecutionResult with trace and AI message.
        """
        self._sync_robot_state()

        # ── Stage 1: MATCH against skill aliases (no LLM needed) ──
        match = self._skill_registry.match(instruction)

        if match is not None:
            if match.direct:
                # Direct skill: execute immediately, zero LLM calls
                return self._execute_matched(match, instruction)

            if match.auto_steps and not self._needs_llm_planning(instruction, match):
                # Auto-steps: expand to skill chain, zero LLM calls
                return self._execute_auto_steps(
                    match, instruction,
                    on_message=on_message, on_step=on_step, on_step_done=on_step_done,
                )

        # ── Stage 2: No alias match or complex instruction → LLM ──
        if self._llm is None:
            if match:
                return self._execute_matched(match, instruction)
            return ExecutionResult(
                success=False, status="failed",
                failure_reason=f"Unknown command and no LLM configured.",
            )

        # ── Stage 3: CLASSIFY intent via LLM ──
        intent = self._llm.classify(instruction)
        logger.info("[Agent] Intent: %s for %r", intent, instruction)

        if intent == "chat":
            return self._handle_chat(instruction)

        if intent == "query":
            return self._handle_query(instruction)

        # intent == "task" or "direct" → LLM planning
        return self._handle_task(
            instruction, on_message=on_message, on_step=on_step, on_step_done=on_step_done,
        )

    def _execute_matched(self, match: SkillMatch, instruction: str) -> ExecutionResult:
        """Execute a directly matched skill (no LLM)."""
        skill = self._skill_registry.get(match.skill_name)
        if skill is None:
            return ExecutionResult(success=False, status="failed",
                                   failure_reason=f"Skill {match.skill_name!r} not found")
        context = self._build_context()
        params: dict = {}
        if match.extracted_arg:
            # Try to pass the extracted arg as object_label or query
            if hasattr(skill, 'parameters'):
                if "object_label" in skill.parameters:
                    params["object_label"] = match.extracted_arg
                elif "query" in skill.parameters:
                    params["query"] = match.extracted_arg
        result = skill.execute(params, context)
        self._sync_robot_state()
        return ExecutionResult(
            success=result.success,
            status="completed" if result.success else "failed",
            steps_completed=1 if result.success else 0,
            steps_total=1,
            failure_reason=result.error_message if not result.success else None,
        )

    def _needs_llm_planning(self, instruction: str, match: SkillMatch) -> bool:
        """Determine if instruction needs LLM planning beyond auto_steps.

        Returns True for complex/multi-object instructions that auto_steps
        can't handle (e.g. "put X on the left", "grab everything").
        """
        text = instruction.lower()
        # If there's a destination mentioned, need LLM to figure out place params
        if any(kw in text for kw in ["放到", "放在", "放去", "put", "to the", "on the",
                                       "前", "后", "左", "右", "中"]):
            return True
        # If it mentions multiple objects or "all"
        if any(kw in text for kw in ["所有", "全部", "都", "all", "every", "each"]):
            return True
        return False

    def _execute_auto_steps(
        self, match: SkillMatch, instruction: str,
        on_message: Any = None, on_step: Any = None, on_step_done: Any = None,
    ) -> ExecutionResult:
        """Execute a skill's auto_steps chain without LLM planning."""
        from vector_os_nano.core.types import TaskPlan, TaskStep

        steps = []
        for i, skill_name in enumerate(match.auto_steps):
            params: dict = {}
            if skill_name == match.skill_name and match.extracted_arg:
                if skill_name == "pick":
                    params["object_label"] = match.extracted_arg
                elif skill_name == "detect":
                    params["query"] = match.extracted_arg
            steps.append(TaskStep(
                step_id=f"s{i+1}",
                skill_name=skill_name,
                parameters=params,
                depends_on=[f"s{i}"] if i > 0 else [],
                preconditions=[],
                postconditions=[],
            ))

        # Add home at end
        steps.append(TaskStep(
            step_id=f"s{len(steps)+1}",
            skill_name="home",
            parameters={},
            depends_on=[f"s{len(steps)}"],
            preconditions=[],
            postconditions=[],
        ))

        plan = TaskPlan(goal=instruction, steps=steps)

        if on_message:
            on_message(f"Executing: {' → '.join(match.auto_steps)} → home")

        context = self._build_context()
        result = self._executor.execute(
            plan, self._skill_registry, context,
            on_step=on_step, on_step_done=on_step_done,
        )
        self._sync_robot_state()
        return result

    def _handle_chat(self, instruction: str) -> ExecutionResult:
        """Handle pure chat — LLM response, no robot action."""
        agent_prompt = self._load_agent_prompt()
        response = self._llm.chat(
            instruction,
            system_prompt=agent_prompt,
            history=self._conversation_history,
        )
        self._conversation_history.append({"role": "user", "content": instruction})
        self._conversation_history.append({"role": "assistant", "content": response})
        if len(self._conversation_history) > 30:
            self._conversation_history = self._conversation_history[-30:]

        return ExecutionResult(
            success=True,
            status="chat",
            message=response,
        )

    def _handle_query(self, instruction: str) -> ExecutionResult:
        """Handle state query — detect objects, then AI describes."""
        # Execute scan + detect
        context = self._build_context()
        scan_skill = self._skill_registry.get("scan")
        detect_skill = self._skill_registry.get("detect")

        if scan_skill:
            scan_skill.execute({}, context)
        if detect_skill:
            detect_skill.execute({"query": "all objects"}, context)

        # Get updated state
        self._sync_robot_state()
        objects_info = ""
        if hasattr(self._arm, "get_object_positions"):
            objs = self._arm.get_object_positions()
            objects_info = ", ".join(
                f"{name} at ({pos[0]:.2f}, {pos[1]:.2f})"
                for name, pos in objs.items()
            )

        # Ask LLM to answer the query with updated info
        agent_prompt = self._load_agent_prompt()
        full_prompt = f"{agent_prompt}\n\nDetected objects: {objects_info}"
        response = self._llm.chat(
            instruction,
            system_prompt=full_prompt,
            history=self._conversation_history,
        )
        self._conversation_history.append({"role": "user", "content": instruction})
        self._conversation_history.append({"role": "assistant", "content": response})

        return ExecutionResult(
            success=True,
            status="query",
            message=response,
        )

    def _handle_task(self, instruction: str, on_message: Any = None, on_step: Any = None, on_step_done: Any = None) -> ExecutionResult:
        """Handle task — plan, execute, summarize."""
        self._conversation_history = [{"role": "user", "content": instruction}]

        # Ensure world model has current objects (sim or perception)
        self._refresh_objects()

        max_retries: int = (
            self._config.get("agent", {}).get("max_planning_retries", 3)
        )

        last_result: ExecutionResult | None = None
        plan_message: str | None = None

        for attempt in range(max_retries):
            world_state = self._world_model.to_dict()
            skill_schemas = self._skill_registry.to_schemas()

            plan = self._llm.plan(
                instruction, world_state, skill_schemas,
                self._conversation_history,
            )

            if plan.message:
                plan_message = plan.message

            if plan.requires_clarification:
                return ExecutionResult(
                    success=False,
                    status="clarification_needed",
                    clarification_question=plan.clarification_question,
                    message=plan.message,
                )

            if not plan.steps:
                return ExecutionResult(
                    success=True,
                    status="chat",
                    message=plan.message or "I'm not sure what to do.",
                )

            # ── Push AI message BEFORE execution ──
            if plan_message and on_message:
                on_message(plan_message)

            # ── Execute with step callbacks ──
            context = self._build_context()
            result = self._executor.execute(
                plan, self._skill_registry, context,
                on_step=on_step, on_step_done=on_step_done,
            )
            self._sync_robot_state()

            if result.success:
                # ── Stage 6: SUMMARIZE ──
                summary = self._summarize(instruction, result)
                return ExecutionResult(
                    success=True,
                    status="completed",
                    steps_completed=result.steps_completed,
                    steps_total=result.steps_total,
                    trace=result.trace,
                    message=plan_message,
                    world_model_diff=result.world_model_diff,
                )

            last_result = result

            # ── Stage 5: ADAPT — inject failure context for re-planning ──
            failed_skill = result.failed_step.skill_name if result.failed_step else "unknown"
            self._conversation_history.append({
                "role": "assistant",
                "content": (
                    f"Execution failed at step '{failed_skill}': {result.failure_reason}. "
                    f"Please adjust the plan or inform the user."
                ),
            })
            logger.warning(
                "[Agent] Attempt %d/%d failed: %s",
                attempt + 1, max_retries, result.failure_reason,
            )

            # If the object simply doesn't exist, no point retrying same plan
            reason = result.failure_reason or ""
            if "Cannot locate" in reason or "not found" in reason.lower():
                # Ask LLM to explain to the user instead of retrying
                agent_prompt = self._load_agent_prompt()
                explain = self._llm.chat(
                    f"The command '{instruction}' failed because: {reason}. "
                    f"Explain this to the user briefly and suggest alternatives.",
                    system_prompt=agent_prompt,
                    history=self._conversation_history,
                )
                return ExecutionResult(
                    success=False,
                    status="failed",
                    steps_completed=result.steps_completed,
                    steps_total=result.steps_total,
                    failed_step=result.failed_step,
                    failure_reason=result.failure_reason,
                    trace=result.trace,
                    message=explain,
                )

        # All attempts exhausted
        if last_result is not None:
            return ExecutionResult(
                success=last_result.success,
                status=last_result.status,
                steps_completed=last_result.steps_completed,
                steps_total=last_result.steps_total,
                failed_step=last_result.failed_step,
                failure_reason=last_result.failure_reason,
                trace=last_result.trace,
                message=plan_message,
            )

        return ExecutionResult(
            success=False,
            status="failed",
            failure_reason="All planning attempts exhausted",
        )

    def _summarize(self, original_request: str, result: ExecutionResult) -> str:
        """Generate a user-friendly summary of execution results."""
        if self._llm is None:
            return ""
        trace_str = "\n".join(
            f"  {s.skill_name}: {s.status} ({s.duration_sec:.1f}s)"
            for s in result.trace
        )
        try:
            return self._llm.summarize(original_request, trace_str)
        except Exception:
            return ""

    def _load_agent_prompt(self) -> str:
        """Load agent.md and fill with current state."""
        from pathlib import Path
        prompt = ""
        for p in [
            Path("config/agent.md"),
            Path(__file__).parent.parent.parent / "config" / "agent.md",
        ]:
            if p.exists():
                prompt = p.read_text()
                break
        if not prompt:
            prompt = "You are V, AI assistant for Vector OS Nano robot arm. {mode} {arm_status} {gripper_status} {objects_info}"

        mode = "MuJoCo simulation" if hasattr(self._arm, "get_object_positions") else "real hardware"
        arm_status = "connected" if self._arm else "disconnected"
        gripper_status = "unknown"
        if self._gripper:
            try:
                pos = self._gripper.get_position()
                gripper_status = "open" if pos > 0.5 else "closed"
            except Exception:
                pass
        objects_info = "unknown"
        if hasattr(self._arm, "get_object_positions"):
            objs = self._arm.get_object_positions()
            objects_info = ", ".join(objs.keys()) if objs else "none"

        try:
            return prompt.format(
                mode=mode, arm_status=arm_status,
                gripper_status=gripper_status, objects_info=objects_info,
            )
        except (KeyError, IndexError):
            return prompt

    # -------------------------------------------------------------------------
    # Convenience methods
    # -------------------------------------------------------------------------

    def home(self) -> bool:
        """Move arm to home position.

        Returns:
            True when the home skill succeeds, False otherwise.
        """
        return self.execute("home").success

    def stop(self) -> None:
        """Emergency stop — halts arm motion immediately."""
        if self._arm is not None:
            self._arm.stop()

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

    def register_skill(self, skill: Skill) -> None:
        """Register a custom skill, immediately available to the planner.

        Overwrites any existing skill with the same name.

        Args:
            skill: Object satisfying the Skill protocol.
        """
        self._skill_registry.register(skill)

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
        # Lazy-init IK solver
        if self._ik_solver is None and self._arm is not None:
            try:
                from vector_os_nano.hardware.so101.ik_solver import IKSolver  # lazy

                self._ik_solver = IKSolver()
                if hasattr(self._arm, "set_ik_solver"):
                    self._arm.set_ik_solver(self._ik_solver)
            except Exception as exc:
                logger.debug("IK solver not available: %s", exc)

        # Lazy-init calibration
        if self._calibration is None:
            try:
                from pathlib import Path

                from vector_os_nano.perception.calibration import Calibration  # lazy

                cal_file: str = self._config.get("calibration", {}).get("file", "")
                if cal_file:
                    # Resolve relative paths: try ~/Desktop/vector_os/<cal_file> then cwd
                    cal_path = Path(cal_file)
                    if not cal_path.is_absolute():
                        candidate = Path.home() / "Desktop" / "vector_os_nano" / cal_path
                        if not candidate.exists():
                            candidate = Path.cwd() / cal_path
                        if candidate.exists():
                            cal_path = candidate
                    self._calibration = Calibration.load(str(cal_path))
                else:
                    self._calibration = Calibration()
            except Exception as exc:
                logger.debug("Calibration not available: %s", exc)

        return SkillContext(
            arm=self._arm,
            gripper=self._gripper,
            perception=self._perception,
            world_model=self._world_model,
            calibration=self._calibration,
            config=self._config,
        )
