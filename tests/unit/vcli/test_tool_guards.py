# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Adversarial tests for the shared path guard of the read/search/write tools (R3).

The model's arguments are attacker-steerable (any text it reads can inject them), so
every class of bypass gets a test: credential paths, '..' traversal, symlinks into a
secret, absolute paths into .git, option injection into ripgrep, glob escapes, and the
permission layering (an auto-allowed read outside the project must ask, not allow).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import pytest

from vector_os_nano.vcli.permissions import PermissionContext
from vector_os_nano.vcli.tools.base import ToolContext
from vector_os_nano.vcli.tools.file_tools import FileEditTool, FileReadTool, FileWriteTool
from vector_os_nano.vcli.tools.search_tools import GlobTool, GrepTool

SECRET = "sk-ant-DO-NOT-LEAK-1234"


@pytest.fixture()
def box(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """A fake $HOME holding credential stores, and a project dir outside it."""
    home = tmp_path / "home"
    project = tmp_path / "project"
    outside = tmp_path / "elsewhere"
    for d in (home / ".vector", home / ".ssh", home / ".aws", home / ".config" / "gh",
              home / ".claude", project / ".git" / "hooks", outside):
        d.mkdir(parents=True)
    (home / ".vector" / "config.yaml").write_text(f"anthropic_api_key: {SECRET}\n")
    (home / ".vector" / "oauth_credentials.json").write_text(f'{{"token": "{SECRET}"}}')
    (home / ".vector" / "personality.md").write_text("be kind\n")
    (home / ".ssh" / "id_ed25519").write_text(SECRET)
    (home / ".aws" / "credentials").write_text(SECRET)
    (home / ".config" / "gh" / "hosts.yml").write_text(SECRET)
    (home / ".claude" / ".credentials.json").write_text(SECRET)
    (project / ".env").write_text(f"ANTHROPIC_API_KEY={SECRET}\n")
    (project / ".env.example").write_text("ANTHROPIC_API_KEY=\n")
    (project / "server.pem").write_text(SECRET)
    (project / "id_rsa").write_text(SECRET)
    (project / "id_rsa.pub").write_text("ssh-rsa AAAA public\n")
    # Upper-case spellings: distinct files on Linux, the real secrets on a
    # case-insensitive filesystem (macOS, Windows).
    for name in (".ENV", "server.PEM", "api.Key", "ID_RSA", ".Env.local"):
        (project / name).write_text(SECRET)
    for rel in (".AWS/credentials", ".SSH/config", ".Vector/config.yaml",
                ".Config/gh/hosts.yml", ".NETRC", ".Git-Credentials"):
        (home / rel).parent.mkdir(parents=True, exist_ok=True)
        (home / rel).write_text(SECRET)
    (project / ".GIT").mkdir()
    (project / ".GIT" / "config").write_text(SECRET)
    (project / ".ENV.Example").write_text("ANTHROPIC_API_KEY=\n")
    (project / ".git" / "config").write_text(f"url = https://x:{SECRET}@github.com/a/b\n")
    (project / ".git" / "hooks" / "pre-commit").write_text("#!/bin/sh\n")
    (project / "main.py").write_text("def main():\n    return 1\n")
    (outside / "notes.txt").write_text("outside the project\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    return {"home": home, "project": project, "outside": outside}


def _ctx(cwd: Path, session: Any = None) -> ToolContext:
    return ToolContext(agent=None, cwd=cwd, session=session, permissions=None,
                       abort=threading.Event())


def _decide(tool: Any, params: dict[str, Any], cwd: Path) -> str:
    return PermissionContext().check(tool, params, _ctx(cwd)).behavior


# ---------------------------------------------------------------------------
# Credentials: refused outright, by the permission gate AND by execute()
# ---------------------------------------------------------------------------

def _credential_paths(box: dict[str, Path]) -> list[str]:
    h, p = box["home"], box["project"]
    return [
        str(h / ".vector" / "config.yaml"),
        "~/.vector/oauth_credentials.json",
        str(h / ".ssh" / "id_ed25519"),
        str(h / ".aws" / "credentials"),
        str(h / ".config" / "gh" / "hosts.yml"),
        str(h / ".claude" / ".credentials.json"),
        str(p / ".env"),
        ".env",                                   # relative, inside the project
        str(p / "server.pem"),
        str(p / "id_rsa"),
        str(p / ".git" / "config"),
        "/etc/shadow",
        ".ENV", "server.PEM", "api.Key", "ID_RSA", ".Env.local", ".GIT/config",
        str(h / ".AWS" / "credentials"), str(h / ".SSH" / "config"),
        str(h / ".Vector" / "config.yaml"), str(h / ".Config" / "gh" / "hosts.yml"),
        str(h / ".NETRC"), str(h / ".Git-Credentials"),
    ]


def test_file_read_refuses_every_credential_path(box: dict[str, Path]) -> None:
    tool, cwd = FileReadTool(), box["project"]
    for path in _credential_paths(box):
        assert _decide(tool, {"file_path": path}, cwd) == "deny", path
        # --no-permission cannot lift it either (an intrinsic deny is a hard stop)
        nop = PermissionContext(no_permission=True).check(tool, {"file_path": path}, _ctx(cwd))
        assert nop.behavior == "deny", path
        res = tool.execute({"file_path": path}, _ctx(cwd))
        assert res.is_error and SECRET not in res.content, path


def test_traversal_into_a_secret_is_resolved_first(box: dict[str, Path]) -> None:
    sneaky = str(box["project"] / ".." / "home" / ".vector" / "config.yaml")
    assert _decide(FileReadTool(), {"file_path": sneaky}, box["project"]) == "deny"
    res = FileReadTool().execute({"file_path": sneaky}, _ctx(box["project"]))
    assert res.is_error and SECRET not in res.content


def test_symlink_inside_the_project_to_a_secret_is_refused(box: dict[str, Path]) -> None:
    link = box["project"] / "docs.txt"
    link.symlink_to(box["home"] / ".vector" / "config.yaml")
    assert _decide(FileReadTool(), {"file_path": str(link)}, box["project"]) == "deny"
    res = FileReadTool().execute({"file_path": str(link)}, _ctx(box["project"]))
    assert res.is_error and SECRET not in res.content


def test_templates_and_public_keys_stay_readable(box: dict[str, Path]) -> None:
    for name in (".env.example", ".ENV.Example", "id_rsa.pub", "main.py"):
        path = str(box["project"] / name)
        assert _decide(FileReadTool(), {"file_path": path}, box["project"]) == "allow", name
        assert not FileReadTool().execute({"file_path": path}, _ctx(box["project"])).is_error


# ---------------------------------------------------------------------------
# Project root: allow inside, ask outside (not auto-allowed by the read-only flag)
# ---------------------------------------------------------------------------

def test_read_inside_project_is_allowed_outside_asks(box: dict[str, Path]) -> None:
    cwd = box["project"]
    assert _decide(FileReadTool(), {"file_path": "main.py"}, cwd) == "allow"
    assert _decide(FileReadTool(), {"file_path": str(box["outside"] / "notes.txt")}, cwd) == "ask"
    # ~/.vector holds more than secrets: the rest is not denied, but it asks
    assert _decide(FileReadTool(), {"file_path": "~/.vector/personality.md"}, cwd) == "ask"


def test_a_symlink_out_of_the_project_asks(box: dict[str, Path]) -> None:
    link = box["project"] / "notes_link.txt"
    link.symlink_to(box["outside"] / "notes.txt")
    assert _decide(FileReadTool(), {"file_path": str(link)}, box["project"]) == "ask"


def test_home_as_the_project_root_never_auto_allows(box: dict[str, Path]) -> None:
    assert _decide(FileReadTool(), {"file_path": "~/.vector/personality.md"}, box["home"]) == "ask"


def test_search_tools_ask_outside_the_project(box: dict[str, Path]) -> None:
    cwd = box["project"]
    for tool, params in ((GlobTool(), {"pattern": "*"}), (GrepTool(), {"pattern": "x"})):
        assert _decide(tool, params, cwd) == "allow"  # default path = the project
        assert _decide(tool, {**params, "path": str(box["outside"])}, cwd) == "ask"
        assert _decide(tool, {**params, "path": str(box["home"])}, cwd) == "ask"
        assert _decide(tool, {**params, "path": str(box["home"] / ".ssh")}, cwd) == "deny"
        assert _decide(tool, {**params, "path": str(cwd / ".." / "home" / ".aws")}, cwd) == "deny"


# ---------------------------------------------------------------------------
# grep / glob
# ---------------------------------------------------------------------------

def test_grep_never_returns_secret_file_contents(box: dict[str, Path]) -> None:
    """The project holds .env, a .pem, a private key and .git/config: grep -r over the
    project must not surface any of them (the python walk reads every file)."""
    (box["project"] / "leak_link.txt").symlink_to(box["home"] / ".vector" / "config.yaml")
    res = GrepTool().execute({"pattern": "sk-ant"}, _ctx(box["project"]))
    assert SECRET not in res.content, res.content


def test_grep_does_not_follow_a_symlink_out_of_the_search_root(box: dict[str, Path]) -> None:
    (box["project"] / "out.txt").symlink_to(box["outside"] / "notes.txt")
    res = GrepTool().execute({"pattern": "outside the project"}, _ctx(box["project"]))
    assert "outside the project" not in res.content, res.content


def test_grep_pattern_and_path_never_reach_rg_as_options(box: dict[str, Path], monkeypatch) -> None:
    """'--pre=CMD' as a pattern makes rg run CMD on every file: argv must carry the
    pattern after -e and the path after --, and ignore any rg config file."""
    from vector_os_nano.vcli.tools import search_tools

    seen: list[list[str]] = []

    def fake_run(argv: list[str], **kw: Any) -> Any:
        seen.append(list(argv))
        return type("P", (), {"returncode": 1, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(search_tools.subprocess, "run", fake_run)
    evil = "--pre=sh"
    GrepTool().execute({"pattern": evil, "path": str(box["project"])}, _ctx(box["project"]))
    argv = seen[0]
    assert "--no-config" in argv
    i_e, i_dd = argv.index("-e"), argv.index("--")
    assert argv[i_e + 1] == evil and i_e + 1 < i_dd  # the pattern is an -e operand
    assert argv[i_dd + 1:] == [str(box["project"])]  # the path is after --
    assert evil not in argv[:i_e]                       # never as a bare option


def test_grep_rg_hits_in_secret_files_are_dropped(box: dict[str, Path], monkeypatch) -> None:
    """rg (when installed) prints NUL-separated file names; hits in protected files
    are filtered out even though rg read them."""
    from vector_os_nano.vcli.tools import search_tools

    p = box["project"]
    out = f"{p / '.env'}\0" "1:ANTHROPIC_API_KEY=" + SECRET + "\n" + f"{p / 'main.py'}\0" "1:def main():\n"
    monkeypatch.setattr(search_tools.subprocess, "run", lambda argv, **kw: type(
        "P", (), {"returncode": 0, "stdout": out, "stderr": ""})())
    res = GrepTool().execute({"pattern": "."}, _ctx(p))
    assert SECRET not in res.content
    assert res.content == f"{p / 'main.py'}:1:def main():"


def test_grep_option_like_pattern_is_searched_literally(box: dict[str, Path]) -> None:
    (box["project"] / "flags.txt").write_text("use --pre=sh with care\n")
    res = GrepTool().execute({"pattern": "--pre=sh"}, _ctx(box["project"]))
    assert "flags.txt" in res.content


def test_glob_pattern_cannot_climb_out_of_the_root(box: dict[str, Path]) -> None:
    res = GlobTool().execute({"pattern": "../home/.ssh/*"}, _ctx(box["project"]))
    assert res.is_error and "id_ed25519" not in res.content


def test_glob_execute_refuses_a_protected_root(box: dict[str, Path]) -> None:
    res = GlobTool().execute({"pattern": "*", "path": "~/.ssh"}, _ctx(box["project"]))
    assert res.is_error and "id_ed25519" not in res.content


@pytest.mark.parametrize("pattern", ["cfg/*", "*/*", "**/*"])
def test_glob_does_not_list_through_a_symlinked_dir(box: dict[str, Path], pattern: str) -> None:
    """A dir symlink inside the project leaks neither ~/.ssh names nor an outside dir."""
    (box["project"] / "cfg").symlink_to(box["home"] / ".ssh")
    (box["project"] / "ext").symlink_to(box["outside"])
    assert _decide(GlobTool(), {"pattern": pattern}, box["project"]) == "allow"
    res = GlobTool().execute({"pattern": pattern}, _ctx(box["project"]))
    assert "id_ed25519" not in res.content and "notes.txt" not in res.content


def test_glob_still_lists_ordinary_project_files(box: dict[str, Path]) -> None:
    res = GlobTool().execute({"pattern": "*.py"}, _ctx(box["project"]))
    assert not res.is_error and "main.py" in res.content


# ---------------------------------------------------------------------------
# dev-world verify oracles: model-authored, run with no prompt -> same guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expr", [
    "not path_contains({p!r}, 'sk-ant')",
    "grep_count('sk-ant', {p!r}) == 0",
    "grep_count('sk-ant', '.') == 0",          # the walk meets .env / server.pem
    "not path_contains('../elsewhere/notes.txt', 'outside')",
])
@pytest.mark.parametrize("p", [".env", ".git/config", "cfg_link"])
def test_negated_dev_oracle_on_a_refused_path_verifies_false(box: dict[str, Path],
                                                             expr: str, p: str) -> None:
    """A refused read must fail closed in BOTH polarities: if path_contains/grep_count
    returned False/0 for a secret, 'not path_contains(...)' would verify True whatever
    the file holds, and a task that never happened could grade GROUNDED."""
    from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
    from vector_os_nano.vcli.worlds.dev import dev_verify_namespace

    (box["project"] / "cfg_link").symlink_to(box["home"] / ".vector" / "config.yaml")
    assert GoalVerifier(dev_verify_namespace()).verify(expr.format(p=p)) is False


def test_dev_oracles_cannot_probe_secret_files(box: dict[str, Path]) -> None:
    from vector_os_nano.vcli.worlds.dev import file_exists, grep_count, path_contains

    (box["project"] / "cfg_link").symlink_to(box["home"] / ".vector" / "config.yaml")
    (box["project"] / "src").mkdir()
    (box["project"] / "src" / "app.py").write_text("def main():\n    pass\n")
    for path in (".env", ".git/config", "server.pem", "cfg_link"):
        with pytest.raises(PermissionError):          # refused, never a False/0 answer
            path_contains(path, "sk-ant")
        with pytest.raises(PermissionError):
            grep_count("sk-ant", path)
    with pytest.raises(PermissionError):              # the walk meets .env: no partial count
        grep_count("sk-ant", ".")
    assert grep_count("^def main", "src") == 1        # ordinary files still searched
    assert grep_count("^def main", "main.py") == 1
    assert path_contains("main.py", "def main")
    assert file_exists(".env")                        # existence is not content


# ---------------------------------------------------------------------------
# write / edit: the protected-path guard resolves before comparing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool_cls", [FileWriteTool, FileEditTool])
def test_write_guard_resolves_dotdot_symlinks_and_absolute_git(box, tool_cls) -> None:
    p = box["project"]
    (p / "hooks_link").symlink_to(p / ".git" / "hooks")
    attempts = [
        str(p / ".git" / "hooks" / "pre-commit"),                 # absolute into .git
        str(p / "sub" / ".." / ".git" / "hooks" / "pre-commit"),  # '..' into .git
        str(p / "hooks_link" / "pre-commit"),                     # symlink into .git
        str(p / "x" / ".." / ".." / "home" / ".ssh" / "authorized_keys"),
        "~/.vector/config.yaml",
    ]
    params = {"content": "pwned", "old_string": "#!/bin/sh", "new_string": "pwned"}
    for path in attempts:
        full = {**params, "file_path": path}
        assert PermissionContext(no_permission=True).check(tool_cls(), full, _ctx(p)).behavior == "deny", path
        assert tool_cls().execute(full, _ctx(p)).is_error, path
    assert (p / ".git" / "hooks" / "pre-commit").read_text() == "#!/bin/sh\n"
    assert not (box["home"] / ".ssh" / "authorized_keys").exists()


def test_write_overwrite_guard_is_keyed_on_the_real_path(box: dict[str, Path]) -> None:
    """Reading 'a/../main.py' then writing 'main.py' is the same file."""
    p = box["project"]
    session = type("S", (), {"read_files": set()})()
    (p / "a").mkdir()
    assert not FileReadTool().execute({"file_path": "a/../main.py"}, _ctx(p, session)).is_error
    res = FileWriteTool().execute({"file_path": str(p / "main.py"), "content": "x = 2\n"}, _ctx(p, session))
    assert not res.is_error, res.content
    assert (p / "main.py").read_text() == "x = 2\n"


def test_ordinary_in_project_edit_still_works(box: dict[str, Path]) -> None:
    p = box["project"]
    assert _decide(FileEditTool(), {"file_path": "main.py", "old_string": "1", "new_string": "2"}, p) == "ask"
    res = FileEditTool().execute({"file_path": "main.py", "old_string": "return 1", "new_string": "return 2"}, _ctx(p))
    assert not res.is_error and "return 2" in (p / "main.py").read_text()


def test_only_a_no_prompt_search_joins_a_parallel_batch(box: dict[str, Path]) -> None:
    """The engine runs concurrency-safe calls on a thread pool; a call that will ask
    must run alone, or several permission prompts interleave on the terminal."""
    for tool in (GlobTool(), GrepTool()):
        assert tool.is_concurrency_safe({"pattern": "x"}) is True
        assert tool.is_concurrency_safe({"pattern": "x", "path": str(box["outside"])}) is False
        assert tool.is_concurrency_safe({"pattern": "x", "path": "~/.ssh"}) is False


def test_guard_holds_with_no_tool_context(box: dict[str, Path]) -> None:
    """PermissionContext.check(tool, params) with no context uses the process cwd."""
    assert PermissionContext().check(FileReadTool(), {"file_path": ".env"}).behavior == "deny"
    assert PermissionContext().check(FileReadTool(), {"file_path": "main.py"}).behavior == "allow"
    assert os.getcwd() == str(box["project"])
