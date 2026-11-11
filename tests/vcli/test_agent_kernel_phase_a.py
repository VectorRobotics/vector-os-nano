# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Phase A — verified agent kernel / world decoupling.

Covers: dev verify predicates (read-only), world resolution + persona, the
engine's verify namespace (a world owns its predicates; dev predicates never
enter a robot namespace).

No LLM, no network, no robot deps.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vector_os_nano.vcli.worlds import (
    DevWorld,
    RobotWorld,
    resolve_world,
    dev_verify_namespace,
)


# ---------------------------------------------------------------------------
# Dev verify predicates
# ---------------------------------------------------------------------------


class TestDevPredicates:
    def test_read_only_predicates(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "greet.py").write_text("def greet():\n    return 'hi'\n")
        ns = dev_verify_namespace()
        assert set(ns) == {"file_exists", "grep_count", "path_contains"}
        assert ns["file_exists"]("greet.py") is True
        assert ns["file_exists"]("missing.py") is False
        assert ns["grep_count"]("def greet", "greet.py") == 1
        assert ns["path_contains"]("greet.py", "return 'hi'") is True

    def test_path_traversal_blocked(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        ns = dev_verify_namespace()
        # Escaping cwd must be rejected (observational predicates only). Refused
        # paths raise (GoalVerifier -> False), so a negated predicate cannot
        # verify True vacuously either.
        with pytest.raises(PermissionError):
            ns["file_exists"]("../../../../etc/hosts")
        with pytest.raises(PermissionError):
            ns["grep_count"]("root", "..")
        with pytest.raises(PermissionError):
            ns["path_contains"]("../../etc/hosts", "localhost")
        from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier

        assert GoalVerifier(ns).verify("not path_contains('../../etc/hosts', 'localhost')") is False
        assert GoalVerifier(ns).verify("not file_exists('../../no/such/file')") is False

# ---------------------------------------------------------------------------
# World resolution + persona
# ---------------------------------------------------------------------------


class TestWorldResolution:
    def test_resolve(self) -> None:
        assert resolve_world(None).name == "dev"
        assert resolve_world(None).is_robot() is False
        assert resolve_world(SimpleNamespace()).name == "robot"
        assert resolve_world(SimpleNamespace()).is_robot() is True

    def test_personas_distinct(self) -> None:
        dev_role, dev_tools = DevWorld().persona_blocks()
        rob_role, rob_tools = RobotWorld().persona_blocks()
        assert "verified coding and automation agent" in dev_role
        assert "AI core of a real robot" in rob_role
        assert dev_role != rob_role


def _make_engine():
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.tools.base import CategorizedToolRegistry

    return VectorEngine(backend=None, registry=CategorizedToolRegistry())


class TestDevNamespace:
    def test_dev_verify_namespace_evaluates(self, tmp_path, monkeypatch) -> None:
        """The dev predicates evaluate via GoalVerifier over the engine's namespace."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "greet.py").write_text("def greet():\n    pass\n")
        from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier

        eng = _make_engine()
        ns = eng._build_verifier_namespace(None)
        gv = GoalVerifier(ns)
        assert gv.verify("file_exists('greet.py')") is True
        assert gv.verify("grep_count('def greet', 'greet.py') > 0") is True


# ---------------------------------------------------------------------------
# Robot-path regression (T-ROBOT)
# ---------------------------------------------------------------------------


class TestRobotRegression:
    def test_robot_namespace_intact_plus_dev(self) -> None:
        eng = _make_engine()
        base = SimpleNamespace(
            get_position=lambda: (1.0, 2.0, 0.0), get_heading=lambda: 0.5
        )
        agent = SimpleNamespace(_base=base, _spatial_memory=None)
        ns = eng._build_verifier_namespace(agent)
        # Robot primitives present...
        assert "get_position" in ns and "get_heading" in ns
        # WP02 (dev-oracles-in-robot-world fix): dev file predicates are NEVER merged
        # into a robot namespace, so they cannot ground a robot verify.
        assert "file_exists" not in ns and "grep_count" not in ns


# ---------------------------------------------------------------------------
# Shared prelude (1): a world OWNS its verify namespace (INC2)
# ---------------------------------------------------------------------------


class _FakeVerifyWorld:
    """Minimal world that contributes a verify predicate (and a stub override)."""

    name = "fake"

    def is_robot(self) -> bool:
        return False

    def build_verify_namespace(self, agent: Any) -> dict[str, Any]:
        return {
            "near_object": lambda name: name == "mug",
            # Override the engine's empty perception stub with a real predicate.
            "detect_objects": lambda query="": ["mug", "banana"],
        }


class TestWorldOwnsVerifyNamespace:
    def test_engine_merges_world_predicate(self) -> None:
        """A world wired into the engine contributes predicates into the ns."""
        eng = _make_engine()
        eng._world = _FakeVerifyWorld()
        ns = eng._build_verifier_namespace(None)
        # World-provided predicate is present and callable.
        assert "near_object" in ns
        assert ns["near_object"]("mug") is True
        assert ns["near_object"]("bottle") is False

    def test_world_predicate_overrides_perception_stub(self) -> None:
        """A world predicate takes precedence over the engine's empty stub."""
        eng = _make_engine()
        eng._world = _FakeVerifyWorld()
        agent = SimpleNamespace(_base=None, _spatial_memory=None)
        ns = eng._build_verifier_namespace(agent)
        # WP02 removed the empty stub; the world is now the ONLY provider of
        # detect_objects (no fake constant to fall back on).
        assert ns["detect_objects"]() == ["mug", "banana"]

    def test_world_predicate_evaluates_via_goal_verifier(self) -> None:
        """The merged world predicate is usable from a real GoalVerifier."""
        from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier

        eng = _make_engine()
        eng._world = _FakeVerifyWorld()
        gv = GoalVerifier(eng._build_verifier_namespace(None))
        assert gv.verify("near_object('mug')") is True
        assert gv.verify("near_object('lego')") is False

    def test_empty_world_contribution_is_byte_identical(self) -> None:
        """RobotWorld/DevWorld contribute {} -> the stub default is preserved."""
        agent = SimpleNamespace(_base=None, _spatial_memory=None)
        eng = _make_engine()
        eng._world = RobotWorld()  # build_verify_namespace -> {}
        ns_world = eng._build_verifier_namespace(agent)
        eng_no_world = _make_engine()
        eng_no_world._world = None  # falls back to registry -> robot world ({} too)
        ns_none = eng_no_world._build_verifier_namespace(agent)
        # WP02: the empty perception stubs are gone — a plain robot world with a
        # baseless/armless agent contributes NO detect_objects (an unbound oracle
        # fails closed rather than reading a fake []).
        assert "detect_objects" not in ns_world
        assert "detect_objects" not in ns_none
        # Both resolution paths still agree on the (now empty) key set.
        assert set(ns_world) == set(ns_none)


# Moved from tests/unit/vcli/test_prompt.py (build_system_prompt was deleted with run_turn).
def test_robot_persona_launches_by_tool_and_states_no_skill_count() -> None:
    """cli-entry:robot-persona-wrong-launch-instructions: start_simulation is the launch
    path, bash is ruled out, and no hardcoded skill count or stale pick advice remains."""
    from vector_os_nano.vcli.prompt import ROBOT_TOOL_INSTRUCTIONS as text

    assert "start_simulation(embodiment=" in text and "Never launch a sim with bash" in text
    assert "22 skills" not in text and "launch_explore.sh" not in text
    assert "Always detect/scan before attempting pick" not in text
