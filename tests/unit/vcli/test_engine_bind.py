# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""engine.bind: the agent + world + capability registry + verify namespace every
native turn runs on, committed only as a whole and failing loud.

test_registry_basics moved from tests/vcli/test_level69_capability_seam.py (the rest
of that file drove the deleted GoalExecutor / StrategySelector capability route).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from vector_os_nano.vcli.cognitive.capabilities.registry import CapabilityRegistry
from vector_os_nano.vcli.cognitive.capabilities.types import CapabilityResult
from vector_os_nano.vcli.engine import VectorEngine
from vector_os_nano.vcli.worlds import DevWorld, RobotWorld


class _Cap:
    """Minimal fake capability."""

    def __init__(self, name: str, *, side: bool = False, ok: bool = True,
                 required: list[str] | None = None) -> None:
        self.name = name
        self.kind = "test"
        self.side_effecting = side
        self.input_schema: dict[str, Any] = {"required": required or []}
        self.output_schema: dict[str, Any] = {}
        self.invoked = False
        self._ok = ok

    def estimate(self, payload: dict) -> tuple[float, float]:
        return (0.0, 0.0)

    def invoke(self, payload: dict, context: Any) -> CapabilityResult:
        self.invoked = True
        return CapabilityResult(success=self._ok, output={"echo": payload},
                                error="" if self._ok else "cap failed")


def test_registry_basics() -> None:
    reg = CapabilityRegistry()
    cap = _Cap("detect")
    reg.register(cap)
    assert reg.get("detect") is cap
    assert "detect" in reg
    assert reg.names() == frozenset({"detect"})
    assert reg.get("missing") is None
    assert len(reg) == 1


def test_bind_dev_world() -> None:
    eng = VectorEngine(backend=MagicMock())
    eng.bind(None, DevWorld())
    assert eng._vgg_agent is None and isinstance(eng._world, DevWorld)
    assert eng._capability_registry is not None and len(eng._capability_registry) == 0
    assert set(eng._build_verifier_namespace(None)) == {"file_exists", "grep_count", "path_contains"}


def test_bind_robot_world_without_a_camera_registers_no_capability() -> None:
    eng = VectorEngine(backend=MagicMock())
    eng.bind(None, RobotWorld())
    assert len(eng._capability_registry) == 0


class _World:
    name = "probe"

    def __init__(self, *, caps: Exception | None = None, ns: Exception | None = None) -> None:
        self._caps, self._ns = caps, ns

    def is_robot(self) -> bool:
        return True

    def persona_blocks(self, agent: Any = None) -> tuple[str, str]:
        return "", ""

    def register_capabilities(self, registry: Any, agent: Any, backend: Any) -> None:
        registry.register(_Cap("probe_cap"))
        if self._caps is not None:
            raise self._caps

    def build_verify_namespace(self, agent: Any) -> dict[str, Any]:
        if self._ns is not None:
            raise self._ns
        return {}


@pytest.mark.parametrize("world", [
    _World(caps=RuntimeError("detector weights missing")),
    _World(ns=RuntimeError("no ground truth")),
])
def test_a_failed_bind_raises_and_keeps_the_previous_binding(world: _World) -> None:
    """legacy-cognitive:B19 / producers:init-vgg-partial-state: nothing is half-replaced."""
    eng = VectorEngine(backend=MagicMock())
    eng.bind(None, DevWorld())
    before = (eng._vgg_agent, eng._world, eng._capability_registry)
    agent = object()
    with pytest.raises(RuntimeError):
        eng.bind(agent, world)
    assert (eng._vgg_agent, eng._world, eng._capability_registry) == before


def test_bind_names_the_members_a_world_lacks() -> None:
    """cli-entry:init-vgg-failloud-swallowed: a non-world fails loud with the valid set."""
    eng = VectorEngine(backend=MagicMock())
    with pytest.raises(TypeError) as exc:
        eng.bind(None, object())
    msg = str(exc.value)
    assert "build_verify_namespace" in msg and "register_capabilities" in msg and "persona_blocks" in msg
    assert eng._world is None


def test_a_good_bind_commits_all_three() -> None:
    eng = VectorEngine(backend=MagicMock())
    agent = object()
    eng.bind(agent, _World())
    assert eng._vgg_agent is agent and isinstance(eng._world, _World)
    assert eng._capability_registry.names() == frozenset({"probe_cap"})


def test_the_repl_fails_loud_when_the_bind_fails(monkeypatch) -> None:
    """cli-entry:init-vgg-failloud-swallowed: a setup / bind error is shown and the
    REPL exits 2; it never starts on a half-bound engine."""
    import os

    from vector_os_nano.vcli import cli

    def boom(args: Any) -> Any:
        raise TypeError("cannot bind <world>: it lacks ['persona_blocks']")

    monkeypatch.setattr(cli, "_build_turn_context", boom)
    monkeypatch.setattr(cli, "_maybe_reexec_under_mjpython", lambda args: None)
    monkeypatch.setattr(cli, "console", cli.Console(file=open(os.devnull, "w")))
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2
