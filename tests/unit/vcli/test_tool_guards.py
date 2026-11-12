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
from types import SimpleNamespace
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
    # only .vector's credential files are refused (R3CV-2); its other files are outside: ask
    assert _decide(FileReadTool(), {"file_path": "~/.vector/personality.md"}, cwd) == "ask"


def test_a_symlink_out_of_the_project_asks(box: dict[str, Path]) -> None:
    link = box["project"] / "notes_link.txt"
    link.symlink_to(box["outside"] / "notes.txt")
    assert _decide(FileReadTool(), {"file_path": str(link)}, box["project"]) == "ask"


def test_home_as_the_project_root_never_auto_allows(box: dict[str, Path]) -> None:
    (box["home"] / "notes.md").write_text("plain\n")
    assert _decide(FileReadTool(), {"file_path": "~/notes.md"}, box["home"]) == "ask"
    assert _decide(FileReadTool(), {"file_path": "~/.vector/personality.md"}, box["home"]) == "ask"


@pytest.mark.parametrize("pseudo", ["/proc", "/proc/self", "/sys", "/dev"])
def test_an_ancestor_of_home_or_a_pseudo_fs_as_the_root_never_auto_allows(
        box: dict[str, Path], pseudo: str) -> None:
    """cwd=/home with HOME=/home/alice: 'alice/.npmrc' lies inside the root, yet it is
    a home dotfile secret and must not auto-allow; neither may /proc/self/environ."""
    home = box["home"]
    (home / ".npmrc").write_text(SECRET)
    (home / ".cache" / "huggingface").mkdir(parents=True)
    (home / ".cache" / "huggingface" / "token").write_text(SECRET)
    for cwd in (home.parent, home.parent.parent):
        rel = os.path.relpath(home, cwd)
        for f in (".npmrc", ".bash_history", ".cache/huggingface/token"):
            assert _decide(FileReadTool(), {"file_path": f"{rel}/{f}"}, cwd) != "allow", (cwd, f)
        for tool, params in ((GlobTool(), {"pattern": "**/*"}), (GrepTool(), {"pattern": "x"})):
            assert _decide(tool, {**params, "path": "."}, cwd) != "allow", (cwd, tool)
    assert _decide(FileReadTool(), {"file_path": "self/environ"}, Path("/proc")) != "allow"
    assert _decide(GrepTool(), {"pattern": "PATH=", "path": "."}, Path(pseudo)) != "allow"


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


def test_glob_an_absolute_pattern_is_a_clean_error(box: dict[str, Path]) -> None:
    res = GlobTool().execute({"pattern": "/etc/*"}, _ctx(box["project"]))
    assert res.is_error and "shadow" not in res.content


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
    "grep_count('sk-ant', '.') == 0",          # a directory: never walked, refused
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
    with pytest.raises(OSError, match="one regular file"):   # a directory is never walked
        grep_count("sk-ant", ".")
    assert grep_count("^def main", "src/app.py") == 1 # ordinary files still searched
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


# ---------------------------------------------------------------------------
# Credential names are matched by path component anywhere, not under one $HOME
# ---------------------------------------------------------------------------

_BOB = {".aws/credentials": "aws_secret_access_key", ".vector/config.yaml": "anthropic_api_key",
        ".ssh/id_ed25519": "PRIVATE KEY", ".config/gh/hosts.yml": "oauth_token",
        ".config/gcloud/credentials.db": "c", ".docker/config.json": "auth", ".kube/config": "token",
        ".azure/accessTokens.json": "t", ".gnupg/secring.gpg": "k", ".npmrc": "authToken",
        ".pypirc": "password", ".netrc": "password", ".pgpass": "pw", ".git-credentials": "https",
        ".bash_history": "export", ".zsh_history": "export", ".python_history": "key",
        ".claude/.credentials.json": "token", ".cache/huggingface/token": "hf_"}


_STORES = (".aws", ".ssh", ".gnupg", ".kube", ".azure", ".vector", ".docker")
# Files in those stores that no exact-name rule lists (R8a-V1): the store refuses
# everything but its named project files, so each one is refused all the same.
_UNLISTED = {".kube/config-prod": "token", ".kube/configs/staging.yaml": "token",
             ".vector/exports/s1.md": "**You:**", ".vector/history": "+key",
             ".vector/sessions/VECTOR.md": "**You:**", ".vector/exports/personality.md": "**You:**",
             ".azure/service_principal_entries.bin": "secret", ".docker/daemon-auth.json": "auth",
             ".docker/contexts/meta/m.json": "host", ".docker/machine/certs/ca.pem": "CERT"}


@pytest.fixture()
def homes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """cwd holds several users' homes (e.g. root started in /home), $HOME lies elsewhere."""
    root = tmp_path / "home"
    for rel, word in {**_BOB, **_UNLISTED}.items():
        (root / "bob" / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / "bob" / rel).write_text(f"{word}={SECRET}\n")
    (root / "bob" / "docs").mkdir()
    (root / "bob" / "docs" / "aws-notes.md").write_text("how we use aws\n")
    (root / "bob" / "src").mkdir()
    (root / "bob" / "src" / "ssh_utils.py").write_text("def ssh_connect():\n    pass\n")
    (tmp_path / "root").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "root"))
    monkeypatch.chdir(root)
    return root


@pytest.mark.parametrize("rel", list(_BOB))
def test_another_users_credentials_are_refused_with_home_elsewhere(homes: Path, rel: str) -> None:
    from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
    from vector_os_nano.vcli.worlds.dev import dev_verify_namespace

    path = f"bob/{rel}"
    assert _decide(FileReadTool(), {"file_path": path}, homes) == "deny"
    assert FileReadTool().execute({"file_path": path}, _ctx(homes)).is_error
    gv = GoalVerifier(dev_verify_namespace())
    for expr in (f"path_contains({path!r}, {SECRET!r})", f"not path_contains({path!r}, 'nope')",
                 f"grep_count({SECRET!r}, {path!r}) == 1", f"grep_count('nope', {path!r}) == 0"):
        assert gv.verify(expr) is False, expr
    for tool in (GrepTool(), GlobTool()):
        assert _decide(tool, {"pattern": "*", "path": path}, homes) == "deny"
    if rel.split("/")[0] in _STORES:   # a credential store dir: searching it is refused too
        store = path.rsplit("/", 1)[0]
        for tool in (GrepTool(), GlobTool()):
            assert _decide(tool, {"pattern": "*", "path": store}, homes) == "deny", store


@pytest.mark.parametrize("rel", list(_UNLISTED))
def test_an_unlisted_file_in_a_credential_store_is_refused(homes: Path, rel: str) -> None:
    """R8a-V1: bob's home is not a passwd home, so in_project auto-allows anything
    is_protected lets through; a kubeconfig variant or a .vector transcript must not."""
    from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
    from vector_os_nano.vcli.worlds.dev import dev_verify_namespace

    path = f"bob/{rel}"
    assert _decide(FileReadTool(), {"file_path": path}, homes) == "deny"
    assert FileReadTool().execute({"file_path": path}, _ctx(homes)).is_error
    gv = GoalVerifier(dev_verify_namespace())
    for expr in (f"path_contains({path!r}, {SECRET!r})", f"grep_count({SECRET!r}, {path!r}) == 1"):
        assert gv.verify(expr) is False, expr
    store = "bob/" + rel.split("/")[0]
    for tool in (GrepTool(), GlobTool()):
        assert _decide(tool, {"pattern": "*", "path": store}, homes) == "deny", store
    hits = GrepTool().execute({"pattern": SECRET, "path": "bob"}, _ctx(homes)).content
    assert SECRET not in hits, hits


def test_a_search_over_several_homes_never_returns_credentials(homes: Path) -> None:
    res = GrepTool().execute({"pattern": SECRET, "path": "."}, _ctx(homes))
    assert SECRET not in res.content, res.content
    listed = GlobTool().execute({"pattern": "**/*"}, _ctx(homes)).content
    for rel in _BOB:
        assert f"bob/{rel}" not in listed, rel
    assert "aws-notes.md" in listed and "ssh_utils.py" in listed


def test_a_name_that_merely_contains_a_credential_word_stays_readable(homes: Path) -> None:
    from vector_os_nano.vcli.worlds.dev import path_contains

    for path, text in (("bob/docs/aws-notes.md", "use aws"), ("bob/src/ssh_utils.py", "ssh_connect")):
        assert _decide(FileReadTool(), {"file_path": path}, homes) == "allow", path
        assert text in FileReadTool().execute({"file_path": path}, _ctx(homes)).content
        assert text in GrepTool().execute({"pattern": text, "path": path}, _ctx(homes)).content
        assert path_contains(path, text)


def test_a_case_variant_ancestor_of_home_never_auto_allows(tmp_path: Path, monkeypatch) -> None:
    """On a case-insensitive volume cwd=/users holds /Users/alice: the root test folds case."""
    (tmp_path / "Users" / "alice").mkdir(parents=True)
    (tmp_path / "users" / "alice").mkdir(parents=True)
    (tmp_path / "users" / "alice" / "notes.md").write_text("plain\n")
    monkeypatch.setenv("HOME", str(tmp_path / "Users" / "alice"))
    for cwd in (tmp_path / "users", tmp_path / "users" / "alice"):
        assert _decide(FileReadTool(), {"file_path": str(tmp_path / "users/alice/notes.md")}, cwd) == "ask"


def test_case_folding_never_widens_the_project(tmp_path: Path, monkeypatch) -> None:
    """Containment stays exact: /x/PROJ is a sibling of the root /x/proj on Linux."""
    (tmp_path / "proj").mkdir()
    (tmp_path / "PROJ").mkdir()
    (tmp_path / "PROJ" / "a.txt").write_text("x\n")
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    assert _decide(FileReadTool(), {"file_path": str(tmp_path / "PROJ" / "a.txt")}, tmp_path / "proj") == "ask"


# ---------------------------------------------------------------------------
# grep's Python fallback (the path taken when ripgrep is absent)
# ---------------------------------------------------------------------------

@pytest.fixture()
def no_rg(monkeypatch: pytest.MonkeyPatch) -> None:
    from vector_os_nano.vcli.tools import search_tools

    def missing(*a: Any, **kw: Any) -> Any:
        raise FileNotFoundError("rg")

    monkeypatch.setattr(search_tools.subprocess, "run", missing)


def test_grep_fallback_reports_an_invalid_regex(box: dict[str, Path], no_rg: None) -> None:
    res = GrepTool().execute({"pattern": "[unclosed"}, _ctx(box["project"]))
    assert res.is_error and "invalid regex" in res.content and "unterminated" in res.content


def test_grep_fallback_skips_a_fifo(box: dict[str, Path], no_rg: None) -> None:
    os.mkfifo(box["project"] / "pipe")
    out: list[Any] = []
    t = threading.Thread(target=lambda: out.append(
        GrepTool().execute({"pattern": "def main"}, _ctx(box["project"]))), daemon=True)
    t.start()
    t.join(timeout=10)
    assert not t.is_alive(), "grep blocked opening a FIFO"
    assert "main.py" in out[0].content


# ---------------------------------------------------------------------------
# R3 verification follow-ups (R3CV-1..3, 5; R3CV-4 is in test_web_tool.py)
# ---------------------------------------------------------------------------

def test_a_root_holding_any_accounts_home_never_auto_allows(tmp_path: Path, monkeypatch) -> None:
    """R3CV-1: cwd holds bob's home (a passwd entry) and $HOME is elsewhere: bob's
    unlisted credentials (.cargo/credentials.toml) must ask, never auto-allow."""
    from vector_os_nano.vcli.tools import file_tools

    bob = tmp_path / "home" / "bob"
    (bob / ".cargo").mkdir(parents=True)
    (bob / ".cargo" / "credentials.toml").write_text(f"token = {SECRET}\n")
    monkeypatch.setenv("HOME", str(tmp_path / "root"))
    monkeypatch.setattr(file_tools.pwd, "getpwall", lambda: [SimpleNamespace(pw_dir=str(bob))])
    for cwd in (tmp_path / "home", bob):
        assert _decide(FileReadTool(), {"file_path": str(bob / ".cargo" / "credentials.toml")}, cwd) == "ask"
    (bob / "proj").mkdir()
    assert _decide(FileReadTool(), {"file_path": str(bob / "proj" / "a.py")}, bob / "proj") == "allow"


def test_content_free_dir_names_refuse_only_their_credentials(box: dict[str, Path]) -> None:
    """R3CV-2 / R8a-V1: .kube and .azure are refused whole; .docker and .vector refuse
    everything but their named project files (Dockerfile-style; VECTOR.md and personality.md
    as the store's direct children). A '.vector' that is not the resolved $VECTOR_HOME keeps
    no folder: its embodiments/ and worlds/ are refused (C1a-V-D1)."""
    project = box["project"]
    readable = (".docker/Dockerfile", ".docker/php/Dockerfile.dev", ".docker/web.dockerfile",
                ".docker/docker-compose.yml", ".docker/.dockerignore",
                ".vector/VECTOR.md", ".vector/personality.md")
    refused = (".docker/config.json", ".docker/entrypoint.sh", ".docker/contexts/meta/m.json",
               ".kube/config", ".kube/README.md", ".azure/accessTokens.json", ".azure/pipelines.yml",
               ".azure/msal_token_cache.json", ".azure/service_principal_entries.json",
               ".vector/config.yaml", ".vector/oauth_credentials.json", ".vector/history",
               ".vector/exports/s1.md", ".vector/sessions/s1.json", ".vector/traces/t1.jsonl",
               ".vector/sessions/VECTOR.md.jsonl", ".vector/sessions/VECTOR.md",
               ".vector/exports/personality.md", ".vector/memory/apartment/VECTOR.md",
               ".vector/assets/embodiments/robot.yaml", ".vector/embodiments/bot/robot.yaml",
               ".vector/worlds/w/world.yaml", ".vector/Embodiments/b/robot.yaml",
               ".vector/WORLDS/w/world.yaml", ".vector/vector.md.d/notes.md")
    for rel in readable + refused:
        (project / rel).parent.mkdir(parents=True, exist_ok=True)
        (project / rel).write_text(f"x {SECRET}\n")
    for rel in readable:
        assert _decide(FileReadTool(), {"file_path": rel}, project) == "allow", rel
    for rel in refused:
        assert _decide(FileReadTool(), {"file_path": rel}, project) == "deny", rel
    (box["home"] / ".vector" / "VECTOR.md").write_text("notes\n")
    assert _decide(FileReadTool(), {"file_path": "~/.vector/VECTOR.md"}, project) == "ask"
    assert _decide(FileReadTool(), {"file_path": "~/.vector/config.yaml"}, project) == "deny"


def test_a_unicode_case_variant_of_a_credential_dir_is_refused(box: dict[str, Path]) -> None:
    """R3CV-3: lower() leaves U+017F (long s) alone, casefold() folds it to 's': on a
    case-insensitive volume .ſsh IS .ssh."""
    variant = box["project"] / ".ſsh" / "id_ed25519_backup"
    variant.parent.mkdir()
    variant.write_text(SECRET)
    assert _decide(FileReadTool(), {"file_path": str(variant)}, box["project"]) == "deny"


def test_the_dev_oracle_decides_and_reads_on_one_resolution(box: dict[str, Path], monkeypatch) -> None:
    """R3CV-5: a path that resolves to a plain file when the permission is decided and
    to a credential when it is read (a symlink swap in between) must not leak."""
    from vector_os_nano.vcli.tools import file_tools
    from vector_os_nano.vcli.worlds import dev

    benign = box["project"] / "main.py"
    secret = box["home"] / ".vector" / "config.yaml"
    calls: list[str] = []

    def swapping(path_str: str, cwd: Any = None) -> Path:
        calls.append(path_str)
        return benign if len(calls) == 1 else secret

    monkeypatch.setattr(file_tools, "real_path", swapping)
    monkeypatch.setattr(dev, "real_path", swapping)
    assert dev.path_contains("main.py", "def main")  # it read what it decided on
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# The state directory is the RESOLVED $VECTOR_HOME wherever it lives (R8a-F1-m1/m2)
# ---------------------------------------------------------------------------

_STATE_REFUSED = ("config.yaml", "history", "oauth_credentials.json", "sessions/s1.jsonl",
                  "sessions/VECTOR.md", "exports/s1.md", "exports/personality.md",
                  "memory/apartment/scene_graph.yaml", "memory/apartment/VECTOR.md",
                  "cache/scenes/room.xml", "runs/20261112/crash.log", "assets/g1/LICENSE")
_STATE_OWN = ("VECTOR.md", "personality.md", "embodiments/bot/robot.yaml", "worlds/w/world.yaml")


@pytest.fixture()
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """VECTOR_HOME in a MIXED-CASE folder whose name says nothing ('Data/VH', no '.vector'
    component), reached through a symlink; the project lies elsewhere. Mixed case so the
    store's own case folding is exercised (C1a-V-D3): on macOS realpath keeps the typed
    case, so a lower-case spelling of the store that does not resolve IS the store."""
    real, project = tmp_path / "Data" / "VH", tmp_path / "project"
    for rel in _STATE_REFUSED + _STATE_OWN:
        (real / rel).parent.mkdir(parents=True, exist_ok=True)
        (real / rel).write_text(f"x {SECRET}\n")
    project.mkdir()
    (tmp_path / "link").symlink_to(real)
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("VECTOR_HOME", str(tmp_path / "link"))
    monkeypatch.chdir(project)
    return {"real": real, "link": tmp_path / "link", "project": project}


@pytest.mark.parametrize("rel", _STATE_REFUSED)
def test_vector_home_anywhere_refuses_its_state(state: dict[str, Path], rel: str) -> None:
    """Refused (never asked) through its real path, its VECTOR_HOME spelling, a case
    variant and a project symlink, for reads, searches and writes."""
    p = state["project"]
    (p / "leak").symlink_to(state["real"] / rel)
    for path in (str(state["real"] / rel), str(state["link"] / rel), str(state["real"] / rel).upper(),
                 str(state["real"] / rel).lower(), str(state["link"]).swapcase() + "/" + rel, "leak"):
        assert _decide(FileReadTool(), {"file_path": path}, p) == "deny", path
        for tool in (FileWriteTool(), FileEditTool()):
            full = {"file_path": path, "content": "x", "old_string": "x", "new_string": "y"}
            assert PermissionContext(no_permission=True).check(tool, full, _ctx(p)).behavior == "deny", path
    assert SECRET not in FileReadTool().execute({"file_path": "leak"}, _ctx(p)).content
    for tool in (GrepTool(), GlobTool()):  # the store itself is refused whole
        assert _decide(tool, {"pattern": "*", "path": str(state["link"])}, p) == "deny"
    assert (state["real"] / rel).read_text() == f"x {SECRET}\n"


@pytest.mark.parametrize("rel", _STATE_OWN)
def test_vector_home_user_files_are_askable_never_refused(state: dict[str, Path], rel: str) -> None:
    """VECTOR.md, personality.md, embodiments/ and worlds/ are the user's own data: the
    agent may read them and (asking) edit them, like any path outside the project."""
    p = state["project"]
    for path in (str(state["link"] / rel), str(state["real"] / rel)):
        assert _decide(FileReadTool(), {"file_path": path}, p) == "ask", path
        assert _decide(FileWriteTool(), {"file_path": path, "content": "x"}, p) == "ask", path
    assert SECRET in FileReadTool().execute({"file_path": str(state["link"] / rel)}, _ctx(p)).content


def test_a_user_folder_link_into_the_state_is_refused(state: dict[str, Path]) -> None:
    """embodiments/ is askable, but a symlink in it resolves before the rule applies."""
    sneaky = state["real"] / "embodiments" / "bot" / "transcript.jsonl"
    sneaky.symlink_to(state["real"] / "sessions" / "s1.jsonl")
    assert _decide(FileReadTool(), {"file_path": str(sneaky)}, state["project"]) == "deny"


def test_only_the_store_itself_is_matched_never_a_name_like_it(state: dict[str, Path]) -> None:
    """A sibling that shares the store's name as a prefix, or a project folder called
    'vh' or 'sessions', is an ordinary path."""
    p = state["project"]
    for rel in ("../Data/VH2/config.yaml", "vh/config.yaml", "sessions/config.yaml"):
        (p / rel).parent.mkdir(parents=True, exist_ok=True)
        (p / rel).write_text("plain\n")
    assert _decide(FileReadTool(), {"file_path": "../Data/VH2/config.yaml"}, p) == "ask"
    assert _decide(FileReadTool(), {"file_path": "vh/config.yaml"}, p) == "allow"
    assert _decide(FileReadTool(), {"file_path": "sessions/config.yaml"}, p) == "allow"


def test_another_users_store_keeps_only_its_direct_user_files(homes: Path) -> None:
    """The literal '.vector' rule stays for other users' stores as before: it keeps only
    VECTOR.md and personality.md as direct children; their embodiments/ and worlds/ are
    refused (C1a-V-D1: only the resolved $VECTOR_HOME keeps those folders)."""
    bob = homes / "bob" / ".vector"
    for rel in ("VECTOR.md", "personality.md", "embodiments/x/robot.yaml", "worlds/y/world.yaml"):
        (bob / rel).parent.mkdir(parents=True, exist_ok=True)
        (bob / rel).write_text("data\n")
    for rel in ("VECTOR.md", "personality.md"):
        assert _decide(FileReadTool(), {"file_path": f"bob/.vector/{rel}"}, homes) == "allow", rel
    for rel in ("embodiments/x/robot.yaml", "worlds/y/world.yaml", "memory/a/VECTOR.md"):
        assert _decide(FileReadTool(), {"file_path": f"bob/.vector/{rel}"}, homes) == "deny", rel
        top = f"bob/.vector/{rel.split('/')[0]}"
        assert _decide(GlobTool(), {"pattern": "*", "path": top}, homes) == "deny", top
