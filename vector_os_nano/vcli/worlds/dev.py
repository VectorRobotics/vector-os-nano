# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Default "dev" world — a general coding/automation agent (no robot).

Ships with the kernel. Provides:
- the general dev persona (from prompt.py),
- deterministic, side-effect-free verify predicates over the project tree. The
  first paragraph of each predicate's docstring is what the model reads in the
  verify tool's description.

Verify predicates are strictly observational (``file_exists``, ``grep_count``,
``path_contains``). Running a command (e.g. a test suite) is an ACTION — the
permission-gated ``bash`` tool — never a verify predicate.
"""

from __future__ import annotations

import re
from typing import Any

from vector_os_nano.vcli.cognitive.oracle_registry import oracle
from vector_os_nano.vcli.prompt import DEV_ROLE_PROMPT, DEV_TOOL_INSTRUCTIONS
from vector_os_nano.vcli.tools.file_tools import in_project, project_root, real_path, resolved_permission

# A predicate runs inside the GoalVerifier 5s sandbox: one bounded file per call.
_MAX_BYTES_PER_FILE = 1_000_000


def _read_file(path: str) -> str:
    """The text of the ONE regular file at *path*, if the shared read rule
    (file_tools.read_permission) auto-allows it. A model-authored predicate runs with
    no prompt, so it must never become a content oracle on what file_read would ask
    or refuse, and it never walks a directory (a walk that skips or stops part-way
    lets a negated count verify True).

    Anything else RAISES (GoalVerifier turns that into False): a refused path, a
    directory, a missing path, a non-regular file (FIFO, socket) or one over the size
    cap. Returning False/0 would make 'not path_contains(...)' or
    'grep_count(...) == 0' verify True on a file that was never examined."""
    root = project_root(None)
    p = real_path(path, root)  # resolved ONCE: the decision and the read use this path
    perm = resolved_permission(p, root, path)
    if perm.behavior != "allow":
        raise PermissionError(f"verify may not read {path}: {perm.reason}")
    if not p.is_file():
        if not p.exists():
            raise FileNotFoundError(f"verify found no file {path}: state absence with not file_exists(...)")
        raise OSError(f"verify reads one regular file per call; {path} is not one")
    if p.stat().st_size > _MAX_BYTES_PER_FILE:
        raise OSError(f"verify will not read {path}: larger than {_MAX_BYTES_PER_FILE} bytes")
    return p.read_text(encoding="utf-8", errors="ignore")


@oracle(scope="dev", domain="bool")
def file_exists(path: str) -> bool:
    """True if the file *path* (relative to cwd, inside the project) exists; state
    absence as not file_exists(path). A directory (the project root '.' included)
    verifies False either way: name a file.

    Existence is not content, so a protected path inside the project may be asked
    about; anything read_permission would not auto-allow for its location ($HOME,
    an ancestor of it or a pseudo-fs as cwd, outside cwd) raises."""
    p = real_path(path)
    if not in_project(p, project_root(None)):
        raise PermissionError(f"verify may not look outside the project: {path}")
    if p.is_dir():  # vacuous: '.' (the project root) always exists
        raise IsADirectoryError(f"file_exists names a file; {path} is a directory")
    return p.exists()


@oracle(scope="dev", domain="bool")
def path_contains(path: str, substr: str) -> bool:
    """True if the ONE regular file at *path* contains the non-empty *substr*; a
    directory, a missing, secret or non-regular path, or an empty substr verifies
    False. One file per call: to state something about several files, 'and' the
    calls together."""
    if not substr:  # vacuous: every file contains ''
        raise ValueError("path_contains needs a non-empty substring")
    return substr in _read_file(path)


@oracle(scope="dev", domain="count")
def grep_count(pattern: str, path: str) -> int:
    """Number of regex *pattern* matches in the ONE regular file at *path* (path has
    no default); a directory, a missing, secret or non-regular path, a file over
    1 MB, an empty or an invalid regex verifies False. One file per call: to state
    something about several files, 'and' the calls together."""
    if not pattern:  # vacuous: '' matches at every position
        raise ValueError("grep_count needs a non-empty pattern")
    return len(re.compile(pattern).findall(_read_file(path)))


def dev_verify_namespace() -> dict[str, Any]:
    """Return the dev world's read-only verify predicates."""
    return {
        "file_exists": file_exists,
        "grep_count": grep_count,
        "path_contains": path_contains,
    }


class DevWorld:
    """The default, robot-free world: a general coding/automation agent."""

    name = "dev"

    def is_robot(self) -> bool:
        return False

    def persona_blocks(self, agent: Any = None) -> tuple[str, str]:
        return DEV_ROLE_PROMPT, DEV_TOOL_INSTRUCTIONS

    def build_verify_namespace(self, agent: Any) -> dict[str, Any]:
        return dev_verify_namespace()

    def register_capabilities(self, registry: Any, agent: Any, backend: Any) -> None:
        # The model driving the native loop is the dev world's only model.
        return None
