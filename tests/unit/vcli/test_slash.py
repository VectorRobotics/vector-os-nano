# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""The slash commands (vcli/slash.py): one table, every command run, none can kill the REPL."""
from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vector_os_nano.vcli import render, slash
from vector_os_nano.vcli.runtime import Runtime, build_runtime


@pytest.fixture
def out(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    buf = io.StringIO()
    monkeypatch.setattr(render.console, "file", buf)
    return buf


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No key anywhere: no .env on the way, a throwaway HOME and config path."""
    import dotenv

    from vector_os_nano.vcli import config

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(config, "_CONFIG_PATH", tmp_path / ".vector" / "config.yaml")
    from vector_os_nano.vcli.tools import nav_tools

    monkeypatch.setattr(nav_tools, "_TERRAIN_PATH", str(tmp_path / "terrain_map.npz"))
    monkeypatch.delenv("VECTOR_FAKE_LLM_TOOLS", raising=False)
    for var in ("DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _rt(**app: Any) -> Runtime:
    rt = build_runtime(notice=lambda msg: None)
    rt.app_state.update(app)
    return rt


class _SceneGraph:
    def __init__(self, path: Path) -> None:
        self._persist_path = str(path)
        self.cleared = 0

    def clear(self) -> None:
        self.cleared += 1


def test_the_table_is_the_menu() -> None:
    """tests-inventory:slash-commands-untested: every command has a handler and help,
    and the REPL's completer lists exactly the table."""
    names = [c.name for c in slash.COMMANDS]
    assert len(names) == len(set(names)) and all(c.help and callable(c.handler) for c in slash.COMMANDS)
    from prompt_toolkit.document import Document

    shown = [c.text for c in slash.VectorCompleter().get_completions(Document("/"), None)]
    assert shown == [f"/{n}" for n in names]


@pytest.mark.parametrize("name", [c.name for c in slash.COMMANDS if c.name != "quit"])
def test_every_command_runs_without_a_key_and_the_repl_lives(home, out, name) -> None:
    rt = _rt()
    rt.session.append_user("hello")
    assert slash.dispatch(rt, f"/{name}") is True
    assert "failed" not in out.getvalue() and "Traceback" not in out.getvalue(), out.getvalue()


@pytest.mark.parametrize("line", ["/quit", "/exit", "/q"])
def test_quit_leaves_the_repl(home, out, line) -> None:
    assert slash.dispatch(_rt(), line) is False


def test_a_failing_command_prints_its_error_and_the_repl_lives(home, out, monkeypatch) -> None:
    """cli-entry:model-cmd-crash-no-engine: the dispatch is wrapped."""
    def boom(rt: Any, args: list[str]) -> None:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(slash, "_BY_NAME", {**slash._BY_NAME, "usage": slash.Command("usage", "u", boom)})
    assert slash.dispatch(_rt(), "/usage") is True
    assert "/usage failed: RuntimeError: disk on fire" in out.getvalue()
    assert slash.dispatch(_rt(), "/nope") is True and "Unknown: /nope" in out.getvalue()


@pytest.mark.parametrize("line", ["/", "/   "])
def test_a_bare_slash_is_unknown_and_the_repl_lives(home, out, line) -> None:
    """R5V-D1: '/' alone (Enter on the command menu) used to raise ValueError out of
    the REPL."""
    assert slash.dispatch(_rt(), line) is True
    assert "Unknown: /" in out.getvalue()


def test_model_without_a_key_names_the_variable_and_changes_nothing(home, out) -> None:
    """cli-entry:model-cmd-crash-no-engine: /model with no backend used to raise
    AttributeError on None and kill the REPL."""
    rt = _rt()
    before = rt.app_state["credentials"]
    assert slash.dispatch(rt, "/model openrouter") is True
    assert "No OPENROUTER_API_KEY set" in out.getvalue()
    assert rt.engine._backend is None and rt.app_state["credentials"] is before


def test_model_switches_provider_and_key_together_and_rebinds_the_engine(home, out, monkeypatch) -> None:
    """cli-entry:model-switch-key-not-updated / cli-entry:model-switch-split-brain: the
    provider's own key comes with the switch, the next /model starts from it, and the
    engine is re-bound so every holder of the backend has the new one."""
    from vector_os_nano.vcli import backends

    made: list[tuple[str, str, str]] = []
    monkeypatch.setattr(backends, "create_backend", lambda provider, api_key, model, base_url: made.append(
        (provider, api_key, model)) or SimpleNamespace(provider=provider))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
    rt = _rt()
    binds: list[Any] = []
    real_bind = rt.engine.bind
    monkeypatch.setattr(rt.engine, "bind", lambda agent, world: binds.append(rt.engine._backend) or real_bind(agent, world))

    slash.dispatch(rt, "/model openrouter openai/gpt-4.1")
    assert made[-1] == ("openrouter", "or-key", "openai/gpt-4.1") and binds == [rt.engine._backend]
    slash.dispatch(rt, "/model google/gemini-2.5-pro")  # same provider, its own key kept
    assert made[-1] == ("openrouter", "or-key", "google/gemini-2.5-pro")
    slash.dispatch(rt, "/model anthropic")
    assert made[-1] == ("anthropic", "ant-key", "claude-sonnet-4-6")
    assert rt.app_state["credentials"].provider == "anthropic" and len(binds) == 3
    assert "ant-key" not in out.getvalue() and "or-key" not in out.getvalue()


def test_login_rereads_the_environment_without_a_restart(home, out, monkeypatch) -> None:
    """cli-entry:first-run-no-hot-reload-key-echo: no key is typed (nothing to echo);
    a key put in the environment after start is picked up by /login."""
    from vector_os_nano.vcli import backends

    monkeypatch.setattr(backends, "create_backend", lambda **k: SimpleNamespace(**k))
    rt = _rt()
    slash.dispatch(rt, "/login")
    assert rt.engine._backend is None and "No DEEPSEEK_API_KEY set" in out.getvalue()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    slash.dispatch(rt, "/login")
    assert rt.engine._backend.api_key == "ds-key" and "DEEPSEEK_API_KEY found" in out.getvalue()
    assert "ds-key" not in out.getvalue()


def test_config_shows_the_variable_never_the_key(home, out, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-secret-key")
    slash.dispatch(_rt(), "/config")
    assert "DEEPSEEK_API_KEY" in out.getvalue() and "set" in out.getvalue()
    assert "ds-secret-key" not in out.getvalue()


def test_clear_memory_clears_the_one_scene_graph_in_place(home, out, tmp_path) -> None:
    """core:clear-memory-split-brain: every holder keeps the same, now empty, graph."""
    persist = tmp_path / "scene_graph.yaml"
    persist.write_text("rooms: {}")
    sg = _SceneGraph(persist)
    rt = _rt(scene_graph=sg)
    slash.dispatch(rt, "/clear_memory")
    assert sg.cleared == 1 and rt.app_state["scene_graph"] is sg and not persist.exists()


def test_reset_stands_the_body_up(home, out) -> None:
    """The old /reset wrote a flag file nothing read; it calls the base now."""
    calls: list[int] = []
    rt = _rt(agent=SimpleNamespace(_base=SimpleNamespace(reset_pose=lambda: calls.append(1))))
    slash.dispatch(rt, "/reset")
    assert calls == [1] and "pose reset" in out.getvalue()


def test_copy_export_clear(home, out, monkeypatch) -> None:
    copied: list[bytes] = []
    monkeypatch.setattr(slash.subprocess, "run", lambda cmd, input, check, timeout: copied.append(input))
    rt = _rt(last_response="the answer")
    rt.session.append_user("q")
    rt.session.append_assistant("the answer")
    slash.dispatch(rt, "/copy")
    assert copied == [b"the answer"]
    slash.dispatch(rt, "/export")
    exported = home / ".vector" / "exports" / f"{rt.session.session_id}.md"
    assert "**V:** the answer" in exported.read_text()
    slash.dispatch(rt, "/clear")
    assert rt.session._entries == []


def test_a_bad_resume_id_is_a_clean_setup_error(home, out, monkeypatch) -> None:
    """cli-entry:session-litter-and-resume: --resume <unknown id> used to traceback."""
    from vector_os_nano.vcli import cli

    monkeypatch.setattr(cli, "_maybe_reexec_under_mjpython", lambda args: None)
    monkeypatch.setattr(cli, "_setup_logging", lambda verbose: None)
    # The guard's adversarial case: a REAL session file outside ~/.vector/sessions,
    # reachable by a relative id. It is neither loaded nor rewritten.
    user = home / "user"
    monkeypatch.setenv("HOME", str(user))
    (user / ".vector" / "sessions").mkdir(parents=True)
    planted = home / "evil.jsonl"
    body = '{"type": "user", "content": "planted"}\n{"type": "meta", "metadata": {}}\n'
    planted.write_text(body)
    for bad in ("no-such-session", "../../etc/passwd", "../../../evil"):
        with pytest.raises(SystemExit) as ei:
            cli.main(["--resume", bad])
        assert ei.value.code == 2
    assert out.getvalue().count("no saved session") == 3 and "Traceback" not in out.getvalue()
    assert "Resumed" not in out.getvalue() and planted.read_text() == body
