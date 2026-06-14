# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Stage 5 (S5.2) — answer-only GoalTree shape.

The decomposer can emit a 0-action / answer-only plan (one ``SubGoal`` with
``answer_only=True``, the ``answer`` strategy, and ``verify="True"``) for pure
conversation, and the executor/harness runs it: verify trivially true, the answer
text returned as captured output.

The MOAT (CLAUDE.md rule 5) is the focus: the evidence gate must DISTINGUISH a
legitimate answer-only step (no robot evidence BY DESIGN, flagged explicitly) from
an ACTION step that produced no evidence (sentinel verify) — which must STILL fail
the gate. The relaxation is keyed on the explicit ``answer_only`` marker, never on
the verify string.

ADDITIVE: nothing routes to an answer plan yet (no cut-over); these exercise the
new shape directly. Pure kernel logic — no robot, no network, no mujoco.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vector_os_nano.vcli.cognitive.goal_decomposer import GoalDecomposer
from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.strategy_selector import StrategySelector
from vector_os_nano.vcli.cognitive.trace_store import (
    evidence_passed,
    load_trace,
    replay,
    save_trace,
)
from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)
from vector_os_nano.vcli.cognitive.vgg_harness import HarnessConfig, VGGHarness
from vector_os_nano.vcli.worlds.dev import DEV_VOCAB


# ---------------------------------------------------------------------------
# Mock LLM backend (no network)
# ---------------------------------------------------------------------------


class _MockBackend:
    def __init__(self, response: str) -> None:
        self._response = response

    def call(self, messages, tools, system, max_tokens, on_text=None):  # noqa: ANN001
        class _Resp:
            text = self._response

        return _Resp()


def _make_executor(verifier: GoalVerifier) -> GoalExecutor:
    return GoalExecutor(
        strategy_selector=StrategySelector(),
        verifier=verifier,
    )


# ---------------------------------------------------------------------------
# 1. The answer_plan builder produces a verifiable 1-step answer-only tree
# ---------------------------------------------------------------------------


def test_answer_plan_builder_shape() -> None:
    tree = GoalDecomposer.answer_plan("who are you?", "I am Vector OS.")
    assert isinstance(tree, GoalTree)
    assert len(tree.sub_goals) == 1
    sg = tree.sub_goals[0]
    assert sg.answer_only is True
    assert sg.strategy == "answer"
    assert sg.verify == "True"
    assert sg.foreach is None
    assert sg.strategy_params["answer"] == "I am Vector OS."


# ---------------------------------------------------------------------------
# 2. The executor runs an answer step: verify true, answer text captured
# ---------------------------------------------------------------------------


def test_executor_runs_answer_step() -> None:
    tree = GoalDecomposer.answer_plan("hi", "hello there")
    executor = _make_executor(GoalVerifier({}))
    step = executor._execute_sub_goal(tree.sub_goals[0])

    assert step.success is True
    assert step.verify_result is True
    assert step.strategy == "answer"
    # The answer text flows out as captured structured output ({"text": ...}).
    assert step.result_data["output"]["text"] == "hello there"


def test_strategy_selector_routes_answer() -> None:
    sel = StrategySelector()
    result = sel.select(
        SubGoal(name="a", description="d", verify="True", strategy="answer",
                strategy_params={"answer": "x"}, answer_only=True)
    )
    assert result.executor_type == "answer"
    assert result.params["answer"] == "x"


# ---------------------------------------------------------------------------
# 3. End-to-end harness run of an answer plan
# ---------------------------------------------------------------------------


def _decomposer_returning(tree: GoalTree) -> Any:
    class _Dec:
        def decompose(self, task: str, world_context: str) -> GoalTree:
            return tree

    return _Dec()


def test_harness_runs_answer_plan_end_to_end() -> None:
    tree = GoalDecomposer.answer_plan("tell me a joke", "Why did the robot cross the road?")
    executor = _make_executor(GoalVerifier({}))
    harness = VGGHarness(
        decomposer=_decomposer_returning(tree),
        executor=executor,
        config=HarnessConfig(max_step_retries=0, max_pipeline_retries=0),
    )
    trace = harness.run("tell me a joke", "world", goal_tree=tree)

    assert trace.success is True
    assert len(trace.steps) == 1
    assert trace.steps[0].result_data["output"]["text"].startswith("Why did the robot")


# ---------------------------------------------------------------------------
# 4. MOAT — the answer-only step is exempt; an action step with no evidence is NOT
# ---------------------------------------------------------------------------


def _answer_trace() -> ExecutionTrace:
    sg = SubGoal(
        name="answer",
        description="say hi",
        verify="True",
        strategy="answer",
        strategy_params={"answer": "hi"},
        answer_only=True,
    )
    step = StepRecord(
        sub_goal_name="answer",
        strategy="answer",
        success=True,
        verify_result=True,
        duration_sec=0.01,
        result_data={"output": {"text": "hi"}},
    )
    return ExecutionTrace(goal_tree=GoalTree(goal="hi", sub_goals=(sg,)),
                          steps=(step,), success=True, total_duration_sec=0.01)


def _unverified_action_trace() -> ExecutionTrace:
    """An ACTION step (NOT answer_only) with a sentinel verify — no evidence."""
    sg = SubGoal(
        name="do_thing",
        description="change the project",
        verify="True",  # sentinel — carries no deterministic evidence
        strategy="tool_call",
        strategy_params={"tool": "file_write", "args": {}},
        answer_only=False,
    )
    step = StepRecord(
        sub_goal_name="do_thing",
        strategy="tool_call",
        success=True,
        verify_result=True,
        duration_sec=0.01,
    )
    return ExecutionTrace(goal_tree=GoalTree(goal="do", sub_goals=(sg,)),
                          steps=(step,), success=True, total_duration_sec=0.01)


def test_answer_only_step_passes_evidence_gate() -> None:
    # A legitimate answer-only step is evidence-backed-by-design in the dev world.
    assert evidence_passed(_answer_trace(), is_robot=False) is True


def test_unverified_action_step_still_fails_gate_moat() -> None:
    # THE MOAT: an action step with a sentinel verify is NOT counted as verified,
    # even though it has the exact same verify string ("True") as the answer step.
    # The gate distinguishes them ONLY by the explicit answer_only marker.
    assert evidence_passed(_unverified_action_trace(), is_robot=False) is False


def test_answer_and_action_mixed_trace_fails_on_unverified_action() -> None:
    # An answer-only step does not "launder" a sibling unverified action step.
    answer_sg = _answer_trace().goal_tree.sub_goals[0]
    action_sg = _unverified_action_trace().goal_tree.sub_goals[0]
    steps = (
        _answer_trace().steps[0],
        _unverified_action_trace().steps[0],
    )
    trace = ExecutionTrace(
        goal_tree=GoalTree(goal="mixed", sub_goals=(answer_sg, action_sg)),
        steps=steps,
        success=True,
        total_duration_sec=0.02,
    )
    assert evidence_passed(trace, is_robot=False) is False


def test_replay_skips_answer_only_no_false_evidence() -> None:
    # replay() reports "no deterministic predicate checked" for a pure-answer
    # trace (it manufactures no evidence) — and never raises.
    assert replay(_answer_trace(), GoalVerifier({})) is False


def test_answer_only_round_trips_through_trace_store(tmp_path: Path) -> None:
    tr = _answer_trace()
    path = save_trace(tr, tmp_path / "answer.json")
    reloaded = load_trace(path)
    assert reloaded.goal_tree.sub_goals[0].answer_only is True
    # The gate decision survives the round-trip.
    assert evidence_passed(reloaded, is_robot=False) is True


# ---------------------------------------------------------------------------
# 5. The decomposer parses an LLM-emitted answer_only plan and preserves the flag
# ---------------------------------------------------------------------------


def test_decompose_preserves_answer_only_flag() -> None:
    payload = json.dumps({
        "goal": "what can you do?",
        "sub_goals": [
            {
                "name": "answer",
                "description": "explain capabilities",
                "verify": "True",
                "strategy": "answer",
                "timeout_sec": 30,
                "depends_on": [],
                "strategy_params": {"answer": "I can plan and verify tasks."},
                "answer_only": True,
            }
        ],
        "context_snapshot": "",
    })
    dec = GoalDecomposer(_MockBackend(payload), **DEV_VOCAB.as_kwargs())
    tree = dec.decompose("what can you do?", "")

    assert len(tree.sub_goals) == 1
    sg = tree.sub_goals[0]
    # The "answer" strategy survives validation in the dev world (not cleared),
    # and the explicit flag is preserved.
    assert sg.strategy == "answer"
    assert sg.answer_only is True

    # And it verifies true + yields the answer text through the executor.
    executor = _make_executor(GoalVerifier({}))
    step = executor._execute_sub_goal(sg)
    assert step.success is True
    assert step.result_data["output"]["text"] == "I can plan and verify tasks."


def test_action_step_sentinel_verify_is_not_answer_only() -> None:
    # A non-answer step the LLM emits with verify="True" must NOT silently become
    # answer_only (which would let it skip the moat). Only the explicit flag (or
    # the dedicated answer strategy) sets it.
    payload = json.dumps({
        "goal": "create a file",
        "sub_goals": [
            {
                "name": "write_it",
                "description": "write the file",
                "verify": "True",
                "strategy": "tool_call",
                "timeout_sec": 30,
                "depends_on": [],
                "strategy_params": {"tool": "file_write", "args": {}},
            }
        ],
    })
    dec = GoalDecomposer(_MockBackend(payload), **DEV_VOCAB.as_kwargs())
    tree = dec.decompose("create a file", "")
    sg = tree.sub_goals[0]
    assert sg.answer_only is False
    assert sg.strategy == "tool_call"


# ---------------------------------------------------------------------------
# 6. MOAT (adversarial) — an LLM that SETS answer_only:true on an ACTION step
#    must NOT launder that side-effecting step past the evidence gate.
# ---------------------------------------------------------------------------


def _laundered_action_trace() -> ExecutionTrace:
    """An action step the LLM mislabels answer_only=True but routes to a
    side-effecting executor (tool_call). The gate must NOT exempt it: the
    exemption is tied to the side-effect-free 'answer' strategy, not the flag."""
    sg = SubGoal(
        name="do_thing",
        description="change the project",
        verify="True",          # sentinel — no deterministic evidence
        strategy="tool_call",   # SIDE-EFFECTING route — runs a real executor
        strategy_params={"tool": "file_write", "args": {}},
        answer_only=True,       # LLM-set bit (untrusted) — must NOT waive the gate
    )
    step = StepRecord(
        sub_goal_name="do_thing",
        strategy="tool_call",
        success=True,
        verify_result=True,
        duration_sec=0.01,
    )
    return ExecutionTrace(goal_tree=GoalTree(goal="do", sub_goals=(sg,)),
                          steps=(step,), success=True, total_duration_sec=0.01)


def test_llm_answer_only_on_action_step_still_fails_gate_moat() -> None:
    # THE MOAT (rule 5): answer_only is fully LLM-controlled. A step that runs a
    # side-effecting executor (strategy != "answer") is NOT exempted just because
    # the LLM set answer_only=True — the exemption is tied to the zero-I/O
    # "answer" route. So a laundered action step still fails the evidence gate.
    assert evidence_passed(_laundered_action_trace(), is_robot=False) is False


def test_llm_answer_only_on_action_step_not_skipped_by_replay() -> None:
    # replay() must also NOT skip a mislabeled action step as if it were an
    # answer leaf: its sentinel verify is the only predicate, so the trace has
    # nothing deterministic to replay and returns False (no false evidence).
    assert replay(_laundered_action_trace(), GoalVerifier({})) is False


def test_decomposer_refuses_answer_only_on_action_strategy() -> None:
    # The decomposer (belt-and-suspenders with the gate) REFUSES answer_only on a
    # non-"answer" strategy: an adversarial / prompt-injected LLM that emits
    # {strategy:"tool_call", verify:"True", answer_only:true} gets the flag
    # cleared with a fail-loud validation note, so the step stays a real action.
    payload = json.dumps({
        "goal": "create a file",
        "sub_goals": [
            {
                "name": "write_it",
                "description": "write the file",
                "verify": "True",
                "strategy": "tool_call",
                "timeout_sec": 30,
                "depends_on": [],
                "strategy_params": {"tool": "file_write", "args": {}},
                "answer_only": True,  # adversarial: laundering an action
            }
        ],
    })
    dec = GoalDecomposer(_MockBackend(payload), **DEV_VOCAB.as_kwargs())
    tree = dec.decompose("create a file", "")
    sg = tree.sub_goals[0]
    assert sg.strategy == "tool_call"
    assert sg.answer_only is False  # refused
    # And a fail-loud note records the refusal.
    assert any("answer_only refused" in n for n in tree.validation_notes)


def test_decomposer_keeps_answer_only_on_answer_strategy() -> None:
    # The legitimate case is unaffected: answer_only on the "answer" strategy is
    # preserved (no refusal note).
    payload = json.dumps({
        "goal": "who are you?",
        "sub_goals": [
            {
                "name": "answer",
                "description": "introduce",
                "verify": "True",
                "strategy": "answer",
                "timeout_sec": 30,
                "depends_on": [],
                "strategy_params": {"answer": "I am Vector OS."},
                "answer_only": True,
            }
        ],
    })
    dec = GoalDecomposer(_MockBackend(payload), **DEV_VOCAB.as_kwargs())
    tree = dec.decompose("who are you?", "")
    sg = tree.sub_goals[0]
    assert sg.strategy == "answer"
    assert sg.answer_only is True
    assert not any("answer_only refused" in n for n in tree.validation_notes)
