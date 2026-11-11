# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""The ONE runtime builder (vcli/runtime.py): every face builds the same runtime.

The REPL and ``-p`` differ only in how they read input and print: the tool list, the
permissions, the verify namespace and the prompt blocks a turn sees are equal.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import vector_os_nano
from vector_os_nano.vcli.runtime import Runtime, build_runtime

PKG = Path(vector_os_nano.__file__).resolve().parent


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    script = tmp_path / "script.json"
    script.write_text('{"turns": []}')
    monkeypatch.setenv("VECTOR_FAKE_LLM_TOOLS", str(script))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    return tmp_path


def _robot() -> Any:
    from vector_os_nano.core.agent import Agent

    base = SimpleNamespace(name="go2", get_position=lambda: [0.0, 0.0, 0.3], get_heading=lambda: 0.0)
    return Agent(base=base, config={})


def _seen_by_a_turn(rt: Runtime) -> dict[str, Any]:
    from vector_os_nano.vcli.cognitive.trace_store import verify_oracles
    from vector_os_nano.vcli.native_loop import _build_motor_tools, _native_system_prompt

    engine, agent = rt.engine, rt.agent
    oracles = frozenset(verify_oracles(agent, engine))
    return {
        "tools": sorted(_build_motor_tools(agent, engine)),
        "categories": rt.registry.list_categories(),
        "no_permission": engine._permissions.no_permission,
        "verify namespace": sorted(engine._build_verifier_namespace(agent)),
        "oracles": sorted(oracles),
        "prompt": [b["text"] for b in _native_system_prompt(engine, oracles, agent)],
        "world": engine._world.name,
        "app_state": sorted(rt.app_state),
    }


@pytest.mark.parametrize("body", ["dev", "robot"])
def test_the_repl_and_print_faces_build_the_same_runtime(home: Path, body: str) -> None:
    agent = _robot() if body == "robot" else None
    repl = build_runtime(agent=agent, face="repl", notice=lambda msg: None)
    printed = build_runtime(agent=agent, face="print", notice=lambda msg: None)
    seen = _seen_by_a_turn(repl)
    assert seen == _seen_by_a_turn(printed)
    assert seen["world"] == body
    assert "start_simulation" in seen["tools"]
    assert repl.session.metadata["face"] == "repl" and printed.session.metadata["face"] == "print"


def test_one_builder_constructs_the_engine_the_registry_and_the_prompt() -> None:
    """grep-level: no second construction of the engine, the tool registry or the
    native system prompt anywhere in the package."""
    sites: dict[str, list[str]] = {}
    pattern = re.compile(r"\b(VectorEngine|CategorizedToolRegistry|_native_system_prompt)\(")
    for path in PKG.rglob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            m = pattern.search(line)
            if m and not line.lstrip().startswith(("def ", "class ", "#")):
                sites.setdefault(m.group(1), []).append(path.relative_to(PKG).as_posix())
    assert sites == {
        "VectorEngine": ["vcli/runtime.py"],
        "CategorizedToolRegistry": ["vcli/runtime.py"],
        "_native_system_prompt": ["vcli/native_loop.py"],
    }
