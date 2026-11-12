# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""C1a — the state directory (charter section 5): schemas/paths.py owns every path in it.

The layout is the charter's list; VECTOR_HOME set after import moves everything; no
other module spells Path.home() or expanduser; memory an older release kept under
~/.vector_os_nano is moved in once, never deleted on failure.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import vector_os_nano
from vector_os_nano.schemas import paths

PKG = Path(vector_os_nano.__file__).resolve().parent
_REPO = PKG.parent
# Set, but empty: the child's load_dotenv never fills them from a repo-root .env.
_NO_KEYS = {k: "" for k in ("DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                            "MOONDREAM_API_KEY")}


def test_the_layout_is_the_charters_list() -> None:
    assert list(paths.LAYOUT) == [
        "config.yaml", "history", "VECTOR.md", "personality.md",
        "sessions/<id>.jsonl", "exports/<id>.md",
        "memory/<world>/",
        "embodiments/<id>/", "worlds/<id>/", "assets/<name>/",
        "cache/scenes/", "runs/<stamp>/",
    ]
    assert (paths.SCENE_GRAPH, paths.TERRAIN_MAP) == ("scene_graph.yaml", "terrain_map.npz")
    assert {e for e, own in paths.LAYOUT.items() if own} == {
        "VECTOR.md", "personality.md", "embodiments/<id>/", "worlds/<id>/"}


def test_vector_home_set_after_import_moves_every_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VECTOR_HOME", str(tmp_path / "vh"))
    got = [paths.config_file(), paths.history_file(), paths.user_context(), paths.personality(),
           paths.sessions(), paths.export_file("s1"), paths.memory("apartment"), paths.assets("g1"),
           paths.scene_cache()]
    assert [p.relative_to(tmp_path / "vh").as_posix() for p in got] == [
        "config.yaml", "history", "VECTOR.md", "personality.md", "sessions", "exports/s1.md",
        "memory/apartment", "assets/g1", "cache/scenes"]
    monkeypatch.delenv("VECTOR_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    assert paths.home() == tmp_path / "h" / ".vector"  # the catalogued default
    monkeypatch.setenv("VECTOR_HOME", "~/state")
    assert paths.home() == tmp_path / "h" / "state"


def _home_reads(tree: ast.AST) -> list[int]:
    """Lines that call Path.home() or any expanduser (an imported one included)."""
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and (node.attr == "expanduser" or (
                node.attr == "home" and isinstance(node.value, ast.Name) and node.value.id == "Path")):
            bad.append(node.lineno)
        if isinstance(node, ast.ImportFrom) and any(a.name == "expanduser" for a in node.names):
            bad.append(node.lineno)
    return bad


def test_only_schemas_paths_reads_the_home_directory() -> None:
    offenders = []
    for path in PKG.rglob("*.py"):
        if "third_party" in path.parts or path == PKG / "schemas" / "paths.py":
            continue
        offenders += [f"{path.relative_to(_REPO)}:{n}" for n in _home_reads(ast.parse(path.read_text()))]
    assert not offenders, "home-directory reads outside schemas/paths.py:\n" + "\n".join(offenders)


@pytest.mark.parametrize("src", ["Path.home() / '.x'", "os.path.expanduser('~')", "Path('~').expanduser()",
                                 "from os.path import expanduser"])
def test_the_ban_catches_every_spelling(src: str) -> None:
    assert _home_reads(ast.parse(src))


# ---------------------------------------------------------------------------
# The one-time move of ~/.vector_os_nano/worlds/<world>/ into memory/<world>/
# ---------------------------------------------------------------------------

@pytest.fixture
def old(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    monkeypatch.setenv("VECTOR_HOME", str(tmp_path / "vh"))
    worlds = tmp_path / "h" / ".vector_os_nano" / "worlds"
    for world in ("apartment", "tabletop"):
        (worlds / world).mkdir(parents=True)
        (worlds / world / "scene_graph.yaml").write_text(f"rooms: {world}\n")
    return worlds


def test_old_memory_is_moved_in_once_with_one_notice(old, tmp_path) -> None:
    note = paths.migrate_old_memory()
    assert note == ("moved the scene memory of apartment, tabletop from ~/.vector_os_nano/worlds "
                    "to $VECTOR_HOME/memory")
    assert (paths.memory("apartment") / "scene_graph.yaml").read_text() == "rooms: apartment\n"
    assert not (tmp_path / "h" / ".vector_os_nano").exists()  # emptied, so removed
    assert paths.migrate_old_memory() == ""  # once


def test_an_existing_world_memory_is_never_overwritten(old) -> None:
    paths.memory("apartment").mkdir(parents=True)
    (paths.memory("apartment") / "scene_graph.yaml").write_text("rooms: newer\n")
    (old.parent / "go2_room").mkdir()  # an unrelated leftover keeps the old folder
    note = paths.migrate_old_memory()
    assert "moved the scene memory of tabletop" in note and "left apartment in" in note
    assert (paths.memory("apartment") / "scene_graph.yaml").read_text() == "rooms: newer\n"
    assert (old / "apartment" / "scene_graph.yaml").read_text() == "rooms: apartment\n"
    assert str(old) not in note  # no machine path: the notice reaches the LLM provider


def test_a_failed_move_deletes_nothing(old, monkeypatch) -> None:
    def broken(src: str, dst: str) -> None:
        Path(dst).mkdir(parents=True)  # a partial copy, then the disk fills
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(shutil, "move", broken)
    note = paths.migrate_old_memory()
    assert "left apartment, tabletop in" in note and "moved" not in note
    for world in ("apartment", "tabletop"):
        assert (old / world / "scene_graph.yaml").read_text() == f"rooms: {world}\n"


# ---------------------------------------------------------------------------
# VECTOR_HOME moves everything: a real dev-world turn through the CLI
# ---------------------------------------------------------------------------

def test_a_dev_turn_writes_only_under_vector_home(tmp_path) -> None:
    """-p with the script LLM in a child: HOME is a stand-in for the real home; the
    session lands in VECTOR_HOME and nothing at all is written under HOME."""
    real, state, work = tmp_path / "real", tmp_path / "vh", tmp_path / "work"
    real.mkdir()
    work.mkdir()
    script = tmp_path / "script.json"
    script.write_text(json.dumps({"turns": [
        {"tool_calls": [{"name": "file_write", "input": {"file_path": "m.txt", "content": "ok\n"}}]},
        {"tool_calls": [{"name": "verify", "input": {"expr": "path_contains('m.txt', 'ok')"}}]},
        {"tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]}))
    env = {**os.environ, **_NO_KEYS}
    env.update(HOME=str(real), VECTOR_HOME=str(state), PYTHONPATH=str(_REPO), VECTOR_FAKE_LLM_TOOLS=str(script))
    proc = subprocess.run(
        [sys.executable, "-m", "vector_os_nano.vcli.cli", "-p", "write m.txt", "--json", "--no-permission"],
        cwd=work, env=env, capture_output=True, text=True, timeout=150,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert len(list((state / "sessions").glob("*.jsonl"))) == 1
    assert (work / "m.txt").read_text() == "ok\n"
    assert list(real.iterdir()) == []


@pytest.mark.sim
def test_a_simulator_start_writes_only_under_vector_home(tmp_path) -> None:
    """A real go2 start under -p: the session, the world's memory (moved in from the old
    ~/.vector_os_nano, with its notice) and the generated scenes all land in
    VECTOR_HOME; the stand-in home keeps no state directory of any release."""
    real, state = tmp_path / "real", tmp_path / "vh"
    old = real / ".vector_os_nano" / "worlds" / "apartment" / "scene_graph.yaml"
    old.parent.mkdir(parents=True)
    old.write_text("rooms: {}\n")
    script = tmp_path / "script.json"
    script.write_text(json.dumps({"turns": [{"text": "standing by"}]}))
    env = {**os.environ, **_NO_KEYS}
    env.update(HOME=str(real), VECTOR_HOME=str(state), PYTHONPATH=str(_REPO), VECTOR_NAV_STACK="",
               VECTOR_FAKE_LLM_TOOLS=str(script))
    proc = subprocess.run(
        [sys.executable, "-m", "vector_os_nano.vcli.cli", "-p", "hi", "--embodiment", "go2", "--headless", "--json"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=400,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-3000:]
    said = " ".join(proc.stderr.split())  # rich wraps the console
    assert "moved the scene memory of apartment from ~/.vector_os_nano/worlds to $VECTOR_HOME/memory" in said, said
    assert "vlm=none" in said, said  # no key reached the child
    assert len(list((state / "sessions").glob("*.jsonl"))) == 1
    assert (state / "memory" / "apartment" / "scene_graph.yaml").is_file()
    assert list((state / "cache" / "scenes").glob("*.xml"))
    left = {p.relative_to(real).as_posix() for p in real.rglob("*")}
    assert not {p for p in left if p.split("/")[0] in (".vector", ".vector_os_nano")
                or p.startswith(".cache/vector_os_nano")}, sorted(left)
