# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The VECTOR_FAKE_LLM_TOOLS backend-injection seam (deterministic test LLM).

The seam at the SINGLE backend site (``runtime.make_backend``)
is gated ONLY on the env var ``VECTOR_FAKE_LLM_TOOLS=<json-path>``. These tests pin:

(1) ABSENT env -> the real ``create_backend`` is called unchanged; the deleted
    decompose-plan seam (``VECTOR_FAKE_LLM``) is not read any more.
(2) PRESENT env -> a ``FakeToolScriptBackend`` that satisfies ``LLMBackend``.
(3) RED — a scripted ``verify('True')`` after a real write STILL classifies RAN
    through the REAL native producer + evidence gate: the seam injects only the
    model's tool calls, never a verdict.

No network, no MuJoCo — pure seam + gate logic.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from vector_os_nano.vcli.backends import LLMBackend
from vector_os_nano.vcli.config import Credentials

_CREDS = Credentials(provider="anthropic", model="m", base_url=None, key_env="ANTHROPIC_API_KEY", api_key="k")


@pytest.mark.cli_main
@pytest.mark.parametrize("legacy_env", [False, True])
def test_absent_env_uses_real_create_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy_env: bool
) -> None:
    monkeypatch.delenv("VECTOR_FAKE_LLM_TOOLS", raising=False)
    if legacy_env:  # the deleted decompose-plan seam must not route anything any more
        (tmp_path / "plan.json").write_text('{"goal": "g", "sub_goals": []}')
        monkeypatch.setenv("VECTOR_FAKE_LLM", str(tmp_path / "plan.json"))
    from vector_os_nano.vcli import backends, runtime

    sentinel = object()
    with patch.object(backends, "create_backend", return_value=sentinel) as real:
        out = runtime.make_backend(_CREDS)
    real.assert_called_once_with(provider="anthropic", api_key="k", model="m", base_url=None)
    assert out is sentinel


@pytest.mark.cli_main
def test_present_env_uses_the_tool_script_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script_file = tmp_path / "script.json"
    script_file.write_text(json.dumps({"turns": [{"tool_calls": [{"name": "finish", "input": {}}]}]}))
    monkeypatch.setenv("VECTOR_FAKE_LLM_TOOLS", str(script_file))
    from vector_os_nano.vcli import backends, runtime

    with patch.object(backends, "create_backend") as real:
        backend = runtime.make_backend(_CREDS)
    real.assert_not_called()
    from tests.harness.fake_backend import FakeToolScriptBackend

    assert isinstance(backend, FakeToolScriptBackend)
    assert isinstance(backend, LLMBackend)  # structural Protocol check
    assert [c.name for c in backend.call(messages=[], tools=[], system=[], max_tokens=10).tool_calls] == ["finish"]


@pytest.mark.cli_main
def test_scripted_true_verify_classifies_ran_not_grounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.harness.fake_backend import FakeToolScriptBackend, tool_turn
    from vector_os_nano.vcli.cognitive.trace_store import verify_oracle_names
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.permissions import PermissionContext
    from vector_os_nano.vcli.runtime import build_registry
    from vector_os_nano.vcli.verdict import VerdictReport
    from vector_os_nano.vcli.worlds import DevWorld

    monkeypatch.chdir(tmp_path)
    registry = build_registry()
    backend = FakeToolScriptBackend.from_tool_script([
        tool_turn(("file_write", {"file_path": "x.txt", "content": "y"})),
        tool_turn(("verify", {"expr": "True"})),  # a tautology, no oracle
        tool_turn(("finish", {}), end=True),
    ])
    eng = VectorEngine(backend=backend, registry=registry, permissions=PermissionContext(no_permission=True))
    eng.bind(None, DevWorld())
    trace = eng.run_turn_native("do the thing")
    assert (tmp_path / "x.txt").read_text() == "y"  # the write really ran
    report = VerdictReport.from_trace(trace, verify_oracle_names(None, eng))
    assert report.verified is False and report.evidence == "RAN"
