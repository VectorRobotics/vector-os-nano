"""Agent Loop — iterative observe-decide-act-verify cycle.

Provides goal-directed execution that can handle iterative tasks
like "clean the table" by running a loop until the LLM judges
the goal is achieved or max_iterations is reached.

Unlike the one-shot planner (Agent.execute -> TaskExecutor), the loop
calls the LLM once per action, observes the result, and decides
the next step dynamically.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from vector_os_nano.core.types import ActionRecord, GoalResult, SkillResult

logger = logging.getLogger(__name__)

_VERIFY_AFTER = frozenset({"pick", "place"})


class AgentLoop:
    """Iterative observe-decide-act-verify engine.

    Args:
        agent_ref: Reference to the Agent instance (provides arm, perception,
            skill_registry, world_model, llm, config).
        config: Agent config dict.
    """

    def __init__(self, agent_ref: Any, config: dict) -> None:
        self._agent = agent_ref
        self._config = config
        loop_cfg = config.get("agent", {}).get("agent_loop", {})
        self._history_max = loop_cfg.get("history_max_actions", 6)
        self._model = loop_cfg.get("model", None)

    def run(
        self,
        goal: str,
        max_iterations: int = 10,
        verify: bool = True,
        on_step: Callable | None = None,
        on_message: Callable | None = None,
    ) -> GoalResult:
        """Execute the observe-decide-act-verify loop.

        Args:
            goal: Natural language goal.
            max_iterations: Safety cap.
            verify: Whether to run perception verification after pick/place.
            on_step: Callback(action_name, iteration, max_iterations) before each action.
            on_message: Callback(str) for LLM messages.

        Returns:
            GoalResult with success, actions trace, and summary.
        """
        start_time = time.monotonic()
        actions: list[ActionRecord] = []
        history: list[dict] = []

        for i in range(max_iterations):
            # OBSERVE
            observation = self._observe()

            # DECIDE
            trimmed_history = self._trim_history(history)
            decision = self._decide(goal, observation, trimmed_history, max_iterations)

            # Check if done
            if decision.get("done"):
                summary = decision.get("summary", "Goal completed.")
                if on_message:
                    on_message(summary)
                return GoalResult(
                    success=True,
                    goal=goal,
                    iterations=i + 1,
                    total_duration_sec=time.monotonic() - start_time,
                    actions=actions,
                    summary=summary,
                    final_world_state=self._agent._world_model.to_dict(),
                )

            # ACT
            action_name = decision.get("action", "scan")
            params = decision.get("params", {})
            reasoning = decision.get("reasoning", "")

            if on_step:
                on_step(action_name, i, max_iterations)

            record = self._act_and_record(
                iteration=i,
                action_name=action_name,
                params=params,
                reasoning=reasoning,
                verify=verify,
            )
            actions.append(record)

            # Update history
            history.append({
                "action": record.action,
                "params": record.params,
                "success": record.skill_success,
                "verified": record.verified,
                "reasoning": record.reasoning,
            })

        # Max iterations reached
        return GoalResult(
            success=False,
            goal=goal,
            iterations=max_iterations,
            total_duration_sec=time.monotonic() - start_time,
            actions=actions,
            summary=f"Max iterations ({max_iterations}) reached without completing goal.",
            final_world_state=self._agent._world_model.to_dict(),
        )

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _observe(self) -> dict:
        """Build observation from current world state."""
        agent = self._agent
        # In sim mode, refresh from ground truth
        is_sim = hasattr(agent._arm, "get_object_positions") if agent._arm else False
        if is_sim:
            agent._refresh_objects()

        return {
            "world_state": agent._world_model.to_dict(),
        }

    def _decide(self, goal: str, observation: dict, history: list, max_iterations: int) -> dict:
        """Ask LLM for next action."""
        agent = self._agent
        if agent._llm is None:
            return {"done": True, "summary": "No LLM configured."}

        skill_schemas = agent._skill_registry.to_schemas()

        try:
            result = agent._llm.decide_next_action(
                goal=goal,
                observation=observation,
                skill_schemas=skill_schemas,
                history=history,
                model_override=self._model,
            )
            return result
        except Exception as exc:
            logger.warning("[AgentLoop] LLM decide failed: %s, defaulting to scan", exc)
            return {"action": "scan", "params": {}, "reasoning": f"LLM error fallback: {exc}"}

    def _act_and_record(
        self,
        iteration: int,
        action_name: str,
        params: dict,
        reasoning: str,
        verify: bool,
    ) -> ActionRecord:
        """Execute a single skill and record the result."""
        agent = self._agent
        start = time.monotonic()

        skill = agent._skill_registry.get(action_name)
        if skill is None:
            logger.warning("[AgentLoop] Skill %r not found", action_name)
            return ActionRecord(
                iteration=iteration,
                action=action_name,
                params=params,
                skill_success=False,
                verified=False,
                reasoning=reasoning,
                duration_sec=time.monotonic() - start,
            )

        context = agent._build_context()
        try:
            result = skill.execute(params, context)
        except Exception as exc:
            logger.error("[AgentLoop] Skill %r raised: %s", action_name, exc)
            return ActionRecord(
                iteration=iteration,
                action=action_name,
                params=params,
                skill_success=False,
                verified=False,
                reasoning=reasoning,
                duration_sec=time.monotonic() - start,
            )

        # Apply world model effects
        agent._world_model.apply_skill_effects(action_name, params, result)

        # Verify if needed
        verified = False
        if verify and self._should_verify(action_name):
            self._verify()
            verified = True

        return ActionRecord(
            iteration=iteration,
            action=action_name,
            params=params,
            skill_success=result.success,
            verified=verified,
            reasoning=reasoning,
            duration_sec=time.monotonic() - start,
        )

    def _should_verify(self, action_name: str) -> bool:
        """Only verify after pick/place (physical world changes)."""
        return action_name in _VERIFY_AFTER

    def _verify(self) -> None:
        """Refresh world model via perception or sim ground truth."""
        agent = self._agent
        is_sim = hasattr(agent._arm, "get_object_positions") if agent._arm else False
        if is_sim:
            agent._refresh_objects()
        else:
            # Hardware: run scan + detect to refresh world model
            scan_skill = agent._skill_registry.get("scan")
            detect_skill = agent._skill_registry.get("detect")
            context = agent._build_context()
            if scan_skill:
                scan_skill.execute({}, context)
            if detect_skill:
                detect_skill.execute({"query": "all objects"}, context)

    def _trim_history(self, history: list) -> list:
        """Keep only the last N entries."""
        return history[-self._history_max:]
