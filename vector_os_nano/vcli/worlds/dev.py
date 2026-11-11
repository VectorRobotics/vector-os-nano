# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Default "dev" world — a general coding/automation agent (no robot).

Ships with the kernel. Provides:
- the general dev persona (from prompt.py),
- deterministic, side-effect-free verify predicates over the project tree,
- a software-task decompose vocabulary for the GoalDecomposer.

Verify predicates are strictly observational (``file_exists``, ``grep_count``,
``path_contains``). Running a command (e.g. a test suite) is an ACTION — the
permission-gated ``bash`` tool — never a verify predicate.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from vector_os_nano.vcli.cognitive.oracle_registry import oracle
from vector_os_nano.vcli.prompt import DEV_ROLE_PROMPT, DEV_TOOL_INSTRUCTIONS
from vector_os_nano.vcli.tools.file_tools import is_protected
from vector_os_nano.vcli.worlds.base import DecomposeVocab

# Bounds for read-only predicates (keep evaluation cheap + safe within the
# GoalVerifier 5s sandbox).
_MAX_FILES = 2000
_MAX_BYTES_PER_FILE = 1_000_000
_SKIP_DIRS = {
    ".git",
    ".venv",
    ".venv-nano",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "dist",
    "build",
}


def _safe_path(path: str) -> Path | None:
    """Resolve *path* under the current working directory.

    Returns the resolved Path if it stays within cwd, else None (reject
    traversal / absolute escapes). Predicates are observational and must not
    read outside the project tree.
    """
    try:
        root = Path.cwd().resolve()
        target = (root / path).resolve()
        if target == root or root in target.parents:
            return target
        return None
    except OSError:
        return None


def _readable_path(path: str) -> Path:
    """_safe_path, minus secret files: a model-authored predicate runs with no
    prompt, so it must not become a content oracle on .env or .git/config.

    A refused path RAISES (GoalVerifier turns that into False): returning
    False/0 would make ``not path_contains('.env', ...)`` verify True whatever
    the file holds. Fail closed in both polarities."""
    p = _safe_path(path)
    if p is None or is_protected(p):
        raise PermissionError(f"verify may not read {path} (outside the project or a protected file)")
    return p


@oracle(scope="dev")
def file_exists(path: str) -> bool:
    """True if *path* (relative to cwd) exists."""
    p = _safe_path(path)
    return bool(p is not None and p.exists())


def _read_bounded(p: Path) -> str:
    """The text of *p*. A file over the size cap RAISES, like an unreadable one: a
    truncated read would make 'not path_contains(...)' verify True on a match past it."""
    if p.stat().st_size > _MAX_BYTES_PER_FILE:
        raise OSError(f"verify will not read {p.name}: larger than {_MAX_BYTES_PER_FILE} bytes")
    return p.read_text(encoding="utf-8", errors="ignore")


@oracle(scope="dev")
def path_contains(path: str, substr: str) -> bool:
    """True if the text file at *path* contains *substr* (bounded read)."""
    p = _readable_path(path)
    return p.is_file() and substr in _read_bounded(p)


def _walk(root: Path) -> list[Path]:
    """Every file under *root*, minus _SKIP_DIRS judged RELATIVE to root (a project
    that itself lives under build/ is still walked). Over _MAX_FILES it raises: a
    partial count would let a negated count verify True."""
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            p = _readable_path(os.path.join(dirpath, name))  # a refused file fails the count
            if not p.is_file():  # dangling link, FIFO, socket: no text to count
                continue
            if len(files) == _MAX_FILES:
                raise OSError(f"verify will not walk {root.name or root}: more than {_MAX_FILES} files")
            files.append(p)
    return files


@oracle(scope="dev")
def grep_count(pattern: str, path: str = ".") -> int:
    """Count regex *pattern* matches in text files under *path* (bounded). An invalid
    regex, an unreadable or oversized file, or a refused path raises."""
    rx = re.compile(pattern)
    root = _readable_path(path)
    if not root.exists():
        return 0
    files = [root] if root.is_file() else _walk(root)
    return sum(len(rx.findall(_read_bounded(fp))) for fp in files)


def dev_verify_namespace() -> dict[str, Any]:
    """Return the dev world's read-only verify predicates."""
    return {
        "file_exists": file_exists,
        "grep_count": grep_count,
        "path_contains": path_contains,
    }


# --- Decompose vocabulary --------------------------------------------------

_DEV_VERIFY_SIGNATURES: dict[str, str] = {
    "file_exists": "file_exists(path: str) -> bool  # path (relative to cwd) exists",
    "grep_count": (
        "grep_count(pattern: str, path: str = '.') -> int  # regex match count under path; "
        "path '.' walks the whole tree and is refused (verifies False) when the tree holds "
        f"a secret file such as .env, a file over {_MAX_BYTES_PER_FILE} bytes or more than "
        f"{_MAX_FILES} files: name a file or a subdirectory"
    ),
    "path_contains": "path_contains(path: str, substr: str) -> bool  # file at path contains substr",
}

_DEV_EXAMPLE = """\
Task: "add a greet() function to greet.py and make sure it is defined"
Response:
{
  "goal": "add a greet() function to greet.py and make sure it is defined",
  "sub_goals": [
    {
      "name": "create_greet_file",
      "description": "create greet.py",
      "verify": "file_exists('greet.py')",
      "strategy": "",
      "timeout_sec": 30,
      "depends_on": [],
      "strategy_params": {},
      "fail_action": ""
    },
    {
      "name": "define_greet",
      "description": "define a greet function in greet.py",
      "verify": "grep_count('def greet', 'greet.py') > 0",
      "strategy": "",
      "timeout_sec": 30,
      "depends_on": ["create_greet_file"],
      "strategy_params": {},
      "fail_action": ""
    }
  ],
  "context_snapshot": ""
}"""

# Phase B: the dev world can now *act* via a single tool-backed strategy. A
# "tool_call" sub-goal dispatches one kernel tool (file_write/file_edit/bash/...)
# through the permission gate + the dev allowlist below. Leave strategy empty for
# steps that only need verification; use tool_call when a step must change the tree.
_DEV_TOOL_EXAMPLE = """\
Task: "create config.txt containing the word ready"
Response:
{
  "goal": "create config.txt containing the word ready",
  "sub_goals": [
    {
      "name": "write_config",
      "description": "write config.txt with the content 'ready'",
      "verify": "path_contains('config.txt', 'ready')",
      "strategy": "tool_call",
      "timeout_sec": 30,
      "depends_on": [],
      "strategy_params": {"tool": "file_write", "args": {"file_path": "config.txt", "content": "ready\\n"}},
      "fail_action": ""
    }
  ],
  "context_snapshot": ""
}"""

# Tools a dev sub-goal may invoke autonomously (the kernel's domain-general
# "code" toolset). Anything off this list is denied by ToolDispatcher before any
# permission check — robot/diag/system tools are never reachable from the dev world.
DEV_TOOL_ALLOWLIST: frozenset[str] = frozenset({
    "file_read",
    "file_write",
    "file_edit",
    "bash",
    "glob",
    "grep",
})

_DEV_STRATEGY_PARAMS_HELP = """\
  - tool_call: {"tool": "<tool_name>", "args": {<tool-specific arguments>}}
      Dispatches one kernel tool through the permission gate. Allowed tools:
        file_write  args: {"file_path": str, "content": str}
        file_edit   args: {"file_path": str, "old_string": str, "new_string": str}
        bash        args: {"command": str}
      Use tool_call ONLY for steps that must change the project; otherwise leave
      strategy empty and rely on the verify predicate alone."""

DEV_VOCAB = DecomposeVocab(
    planner_intro=(
        "You are a software task planner. Decompose the user's task into verifiable "
        "sub-goals, each with a deterministic verify predicate over the project. "
        "For steps that must modify the project, set strategy to \"tool_call\"."
    ),
    verify_functions=frozenset(_DEV_VERIFY_SIGNATURES.keys()),
    verify_fn_signatures=_DEV_VERIFY_SIGNATURES,
    strategy_descriptions={
        "tool_call": "Invoke one kernel tool (file_write/file_edit/bash/...) to act on the project",
        "chat": "Ask the chat LLM for free-form text (no project side effect; verify must still hold)",
    },
    strategies=frozenset({"tool_call", "chat"}),
    strategy_params_help=_DEV_STRATEGY_PARAMS_HELP,
    examples=_DEV_EXAMPLE + "\n\n" + _DEV_TOOL_EXAMPLE,
    fallback_verify="True",
)


class DevWorld:
    """The default, robot-free world: a general coding/automation agent."""

    name = "dev"

    def is_robot(self) -> bool:
        return False

    def persona_blocks(self) -> tuple[str, str]:
        return DEV_ROLE_PROMPT, DEV_TOOL_INSTRUCTIONS

    def register_tools(self, registry: Any, agent: Any) -> None:
        # General code/web tools are registered by the CLI for every world.
        return None

    def build_verify_namespace(self, agent: Any) -> dict[str, Any]:
        return dev_verify_namespace()

    def register_capabilities(self, registry: Any, agent: Any, backend: Any) -> None:
        # The chat LLM as one routable capability (Phase C seam). No-op without a
        # backend (e.g. before /login), keeping the dev path inert until then.
        if backend is None:
            return
        from vector_os_nano.vcli.cognitive.capabilities import LLMChatCapability
        registry.register(LLMChatCapability(backend))

    def decompose_vocab(self) -> DecomposeVocab:
        return DEV_VOCAB

    def derive_vocab_from_registry(self) -> bool:
        # Dev world ships an explicit DEV_VOCAB; no registry derivation.
        return False
