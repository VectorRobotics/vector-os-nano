# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Playground (INC4) — LIVE wiring of a playground scenario into the engine.

Covers the scenario selector added to the CLI:
- ``--scenario tabletop`` resolves the playground world (via the lazy registry
  hook) and, wired through the engine exactly as ``main()`` does, makes the
  engine's verifier namespace carry the playground predicates
  (the GT oracles: holding_object / arm_at_home / object_on / released ...).
- The assertions go THROUGH the engine (parse_args -> _resolve_active_world ->
  init_vgg/_build_verifier_namespace), never by importing the playground into
  the kernel under test.
- An unknown scenario id fails LOUD with the valid set.
- The default (no --scenario) path is byte-unchanged: agent -> robot, none -> dev.

No MuJoCo / network — the arm is a deterministic fake.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tests.harness.fake_gt import bind_fake_sim

from vector_os_nano.vcli.cli import _resolve_active_world, parse_args


# ---------------------------------------------------------------------------
# Deterministic fake arm (no MuJoCo) — same oracle surface the predicates read.
# ---------------------------------------------------------------------------


class FakeArm:
    def __init__(
        self,
        objects: dict[str, list[float]],
        joints: list[float],
        ee: list[float],
        connected: bool = True,
    ) -> None:
        self._objects = objects
        self._joints = list(joints)
        self._ee = list(ee)
        self._connected = connected

    def get_object_positions(self) -> dict[str, list[float]]:
        return {k: list(v) for k, v in self._objects.items()}

    def get_joint_positions(self) -> list[float]:
        return list(self._joints)

    def fk(self, joint_positions: list[float]):
        return list(self._ee), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _fake_arm_agent() -> SimpleNamespace:
    from tests.harness.fake_gt import bind_fake_sim
    from vector_os_nano.embodiments import registry

    arm = FakeArm(
        objects={"mug": [0.22, 0.05, 0.06], "banana": [0.12, 0.12, 0.06]},
        joints=list(registry.load("so101").stance.values()),
        ee=[0.2, 0.0, 0.2],
    )
    agent = SimpleNamespace(_arm=arm, _gripper=None)
    bind_fake_sim(agent, "so101")  # the live sim session's GT oracles
    return agent


def _make_engine():
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.intent_router import IntentRouter
    from vector_os_nano.vcli.tools.base import CategorizedToolRegistry

    class _MockBackend:
        def call(self, messages, tools, system, max_tokens, on_text=None):
            class _R:
                text = "{}"

            return _R()

    return VectorEngine(
        backend=_MockBackend(),
        registry=CategorizedToolRegistry(),
        system_prompt=[],
        intent_router=IntentRouter(),
    )


_PLAYGROUND_PREDICATES = {"holding_object", "arm_at_home", "object_on", "released", "lifted"}


# ---------------------------------------------------------------------------
# --scenario selects the playground world (asserted through the engine).
# ---------------------------------------------------------------------------


class TestScenarioSelectsPlayground:
    def test_resolved_world_is_playground(self) -> None:
        args = parse_args(["--scenario", "tabletop"])
        world = _resolve_active_world(args, agent=None)
        # Assert via the class name / contract, not by importing PlaygroundWorld
        # as a type the kernel would have to know — the world is a duck-typed
        # contract object resolved through the registry.
        assert type(world).__name__ == "PlaygroundWorld"
        assert world.name == "tabletop"
        assert world.is_robot() is True

    def test_engine_namespace_carries_playground_predicates(self) -> None:
        """The selected world, wired into the engine like main() does, makes the
        BUILT verifier namespace contain the playground predicates."""
        args = parse_args(["--scenario", "tabletop"])
        agent = _fake_arm_agent()
        world = _resolve_active_world(args, agent)

        eng = _make_engine()
        # init_vgg(world=...) is the production path that sets engine._world
        # BEFORE the verifier namespace is built. Use it directly so the test
        # exercises the same wiring main() relies on.
        eng.init_vgg(agent=agent, world=world)
        assert type(eng._world).__name__ == "PlaygroundWorld"

        ns = eng._build_verifier_namespace(agent)
        assert _PLAYGROUND_PREDICATES <= set(ns)

    def test_playground_predicates_evaluate_through_goal_verifier(self) -> None:
        """End-to-end: predicates from the selected scenario evaluate through the
        real GoalVerifier off the engine's merged namespace."""
        from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier

        args = parse_args(["--scenario", "tabletop"])
        agent = _fake_arm_agent()
        world = _resolve_active_world(args, agent)
        eng = _make_engine()
        eng.init_vgg(agent=agent, world=world)
        ns = eng._build_verifier_namespace(agent)

        gv = GoalVerifier(ns)
        assert gv.verify("object_on('mug', 'table')") is True
        assert gv.verify("arm_at_home()") is True
        assert gv.verify("holding_object()") is False  # no gripper holding


# ---------------------------------------------------------------------------
# W1.4 — the LIVE executor runs the WORLD's producing-step primitive.
#
# build_step_primitives(agent) returns the per-step PRODUCER the tabletop/foreach
# tests exercise (keyed by DETECT_STRATEGY). Before W1.4 init_vgg never injected it,
# so the real StrategySelector routed ``detect_objects_skill`` to ``invalid``
# (its bare name ``detect_objects`` is not a registered skill) and the producer
# never ran live. These tests prove the WIRED producer is the one that runs through
# the engine's real selector → executor path, and that its output drives a real
# foreach end-to-end (leaf count == produced item count, verify on real oracle).
# ---------------------------------------------------------------------------


class _MutableStubArm:
    """A deterministic stand-in for the SO-101 arm whose ground truth advances.

    Shares the exact oracle surface the playground predicates read. pick lifts the
    held object onto the EE; place drops it (resting) into the tray region, so
    holding_object()/object_on(item, 'tray') flip False->True as the chain runs.
    """

    _TABLE_Z = 0.04  # object centres resting on the table top (z 0.02)
    _LIFT_Z = 0.20
    _TRAY_CENTRE = (0.35, 0.12)  # inside TABLETOP_TRAY's (0.20, 0.0, 0.50, 0.25)

    def __init__(self, object_names: tuple[str, ...]) -> None:
        self._objects: dict[str, list[float]] = {
            name: [0.10 + 0.05 * i, -0.10, self._TABLE_Z]
            for i, name in enumerate(object_names)
        }
        self._ee: list[float] = [0.0, 0.30, 0.30]
        self._connected = True

    def get_object_positions(self) -> dict[str, list[float]]:
        return {k: list(v) for k, v in self._objects.items()}

    def get_joint_positions(self) -> list[float]:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    def fk(self, _joints: Any):
        return list(self._ee), None

    def lift(self, name: str) -> None:
        pos = self._objects[name]
        pos[2] = self._LIFT_Z
        self._ee = [pos[0], pos[1], pos[2]]

    def drop_in_tray(self, name: str) -> None:
        self._objects[name] = [self._TRAY_CENTRE[0], self._TRAY_CENTRE[1], self._TABLE_Z]
        self._ee = [0.0, 0.30, 0.30]


class _StubGripper:
    def __init__(self) -> None:
        self._holding = False

    def is_holding(self) -> bool:
        return self._holding


def _arm_registry():
    """A real SkillRegistry carrying the arm skills — so ``detect_objects`` is NOT
    a registered skill and the selector would route DETECT_STRATEGY to ``invalid``
    (exactly the live case the producer override has to win over)."""
    from vector_os_nano.core.skill import SkillRegistry
    from vector_os_nano.skills.detect import DetectSkill
    from vector_os_nano.skills.home import HomeSkill
    from vector_os_nano.skills.pick import PickSkill

    reg = SkillRegistry()
    for skill in (DetectSkill(), HomeSkill(), PickSkill()):
        reg.register(skill)
    return reg


class TestLiveExecutorRunsWorldPrimitive:
    def test_init_vgg_injects_world_step_primitives(self) -> None:
        """init_vgg wires the world's build_step_primitives into the executor."""
        from vector_os_nano.playground.world import PlaygroundWorld

        agent = _fake_arm_agent()
        world = PlaygroundWorld()  # default tabletop (arm)
        eng = _make_engine()
        eng.init_vgg(agent=agent, world=world, skill_registry=_arm_registry())

        prims = eng._goal_executor._primitives
        assert isinstance(prims, dict)
        # The producer is keyed by the strategy name a plan emits for the producing
        # step — and it is callable (the real detect producer bound to the agent).
        assert PlaygroundWorld.DETECT_STRATEGY in prims
        assert callable(prims[PlaygroundWorld.DETECT_STRATEGY])

    def test_live_selector_routes_detect_strategy_to_world_producer(self) -> None:
        """Through the REAL StrategySelector, a producing step emitting
        DETECT_STRATEGY runs the WORLD producer (not the importlib fallback, which
        would yield nothing / fail invalid). Proof: the produced objects equal the
        scenario's sim-oracle objects."""
        from vector_os_nano.playground.world import PlaygroundWorld
        from vector_os_nano.vcli.cognitive.blackboard import Blackboard
        from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal

        object_names = PlaygroundWorld().scenario.object_names
        arm = _MutableStubArm(object_names)
        agent: Any = SimpleNamespace(_arm=arm, _gripper=_StubGripper())
        bind_fake_sim(agent, "so101")

        world = PlaygroundWorld()
        eng = _make_engine()
        eng.init_vgg(agent=agent, world=world, skill_registry=_arm_registry())
        executor = eng._goal_executor
        executor.blackboard = Blackboard()  # the harness attaches this per run

        # A single producing step whose strategy is the playground's DETECT_STRATEGY.
        tree = GoalTree(
            goal="detect every object",
            sub_goals=(
                SubGoal(
                    name="detect_all",
                    description="detect every object on the table",
                    verify="True",  # a listing proves nothing about the world
                    strategy=PlaygroundWorld.DETECT_STRATEGY,
                ),
            ),
        )
        trace = executor.execute(tree)

        assert trace.success is True
        detect_step = next(s for s in trace.steps if s.sub_goal_name == "detect_all")
        # The WORLD producer ran: its structured output carries the real sim-oracle
        # objects. The importlib fallback has no ``detect_objects_skill`` primitive,
        # so without the wiring this step would fail invalid and yield no objects.
        produced = detect_step.result_data["output"]
        assert produced["count"] == len(object_names)
        assert {o["name"] for o in produced["objects"]} == set(object_names)

    def test_wired_producer_drives_real_foreach_end_to_end(self) -> None:
        """The wired producer's output drives a REAL foreach: leaf count equals the
        produced item count and each per-step verify flips on real oracle state."""
        import json

        from vector_os_nano.playground.world import PlaygroundWorld
        from vector_os_nano.vcli.cognitive.blackboard import Blackboard

        # A scenario WITH a tray drop-zone so object_on(item, 'tray') verifies per item.
        from vector_os_nano.playground.catalog import TABLETOP_TRAY

        object_names = TABLETOP_TRAY.object_names
        n = len(object_names)
        arm = _MutableStubArm(object_names)
        gripper = _StubGripper()
        agent: Any = SimpleNamespace(_arm=arm, _gripper=gripper)
        bind_fake_sim(agent, "so101")

        # The foreach body's pick/place are producer-only ``*_skill`` names too, so
        # the SAME world-primitive override routes them live (the real selector would
        # otherwise mark them invalid). They advance the shared oracle so verify flips.
        def pick(object_label: str | None = None, **_: Any) -> dict[str, Any]:
            if object_label in arm.get_object_positions():
                arm.lift(object_label)
                gripper._holding = True
            return {"picked": object_label}

        def place(object_label: str | None = None, **_: Any) -> dict[str, Any]:
            if object_label in arm.get_object_positions():
                arm.drop_in_tray(object_label)
                gripper._holding = False
            return {"placed": object_label}

        class _TrayWorld(PlaygroundWorld):
            """Augments build_step_primitives with the body pick/place producers so
            the whole chain routes through the wired-primitive path (no DirectSelector)."""

            def build_step_primitives(self, a: Any) -> dict[str, Any]:
                prims = dict(super().build_step_primitives(a))
                prims["pick_skill"] = pick
                prims["place_skill"] = place
                return prims

        world = _TrayWorld(TABLETOP_TRAY)
        eng = _make_engine()
        eng.init_vgg(agent=agent, world=world, skill_registry=_arm_registry())
        executor = eng._goal_executor
        executor.blackboard = Blackboard()

        # Pre-run: nothing placed, nothing held (the oracle starts clean).
        ns = world.build_verify_namespace(agent)
        assert not any(ns["object_on"](n, "tray") for n in object_names)
        assert ns["holding_object"]() is False

        tree = _grab_everything_tree()
        trace = executor.execute(tree)

        assert trace.success is True
        # The detect step's produced list is what the foreach iterates.
        detect_step = next(s for s in trace.steps if s.sub_goal_name == "detect_all")
        detected = detect_step.result_data["output"]["objects"]
        assert {o["name"] for o in detected} == set(object_names)

        names = [s.sub_goal_name for s in trace.steps]
        pick_steps = [x for x in names if x.endswith(".pick_obj")]
        place_steps = [x for x in names if x.endswith(".place_obj")]
        # Leaf count EQUALS the produced item count — driven by the wired producer.
        assert len(pick_steps) == len(detected) == n
        assert len(place_steps) == len(detected) == n

        # Every per-step verify passed DETERMINISTICALLY on the real oracle.
        for s in trace.steps:
            assert s.success is True, f"step {s.sub_goal_name} failed: {s.error}"
            assert s.verify_result is True
            assert s.visual_override is False

        # The oracle advanced: all n objects ended resting in the tray.
        assert all(ns["object_on"](n, "tray") for n in object_names)
        assert ns["holding_object"]() is False

        # Sanity: the captured producer output is JSON-safe structured data.
        json.dumps(detect_step.result_data["output"])


def _grab_everything_tree():
    """scan -> detect_all(DETECT_STRATEGY) -> foreach(obj): pick -> place."""
    from vector_os_nano.playground.world import PlaygroundWorld
    from vector_os_nano.vcli.cognitive.types import ForEachSpec, GoalTree, SubGoal

    body = (
        SubGoal(
            name="pick_obj",
            description="pick up the current object",
            verify="holding_object()",
            strategy="pick_skill",
            strategy_params={"object_label": "${obj.name}"},
        ),
        SubGoal(
            name="place_obj",
            description="place the current object in the tray",
            verify="object_on('${obj.name}', 'tray')",
            strategy="place_skill",
            depends_on=("pick_obj",),
            strategy_params={"object_label": "${obj.name}"},
        ),
    )
    return GoalTree(
        goal="grab everything",
        sub_goals=(
            SubGoal(
                name="detect_all",
                description="detect every object on the table",
                verify="True",
                strategy=PlaygroundWorld.DETECT_STRATEGY,
            ),
            SubGoal(
                name="grab_each",
                description="pick and place each detected object, one by one",
                verify="True",
                strategy="",
                depends_on=("detect_all",),
                foreach=ForEachSpec(
                    source_step="detect_all",
                    source_path="objects",
                    var="obj",
                    body=body,
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Unknown scenario id fails loud (never a silent fallback).
# ---------------------------------------------------------------------------


class TestUnknownScenarioFailsLoud:
    def test_unknown_id_raises_keyerror_with_valid_set(self) -> None:
        args = parse_args(["--scenario", "does-not-exist"])
        with pytest.raises(KeyError) as exc_info:
            _resolve_active_world(args, agent=None)
        msg = str(exc_info.value)
        assert "does-not-exist" in msg
        # The valid set is surfaced so the failure is actionable.
        assert "tabletop" in msg


# ---------------------------------------------------------------------------
# Default (no --scenario) path is unchanged: agent -> robot, none -> dev.
# ---------------------------------------------------------------------------


class TestDefaultUnchanged:
    def test_no_scenario_no_agent_resolves_dev(self) -> None:
        args = parse_args([])
        world = _resolve_active_world(args, agent=None)
        assert type(world).__name__ == "DevWorld"
        assert world.is_robot() is False

    def test_no_scenario_with_agent_resolves_robot(self) -> None:
        args = parse_args([])
        agent: Any = SimpleNamespace(_arm=None, _base=None)
        world = _resolve_active_world(args, agent)
        assert type(world).__name__ == "RobotWorld"
        assert world.is_robot() is True

    def test_default_matches_resolve_world(self) -> None:
        """The no-scenario branch is exactly resolve_world(agent)."""
        from vector_os_nano.vcli.worlds import resolve_world

        args = parse_args([])
        agent: Any = SimpleNamespace(_arm=None, _base=None)
        assert (
            type(_resolve_active_world(args, agent)).__name__
            == type(resolve_world(agent)).__name__
        )
        assert (
            type(_resolve_active_world(args, None)).__name__
            == type(resolve_world(None)).__name__
        )
