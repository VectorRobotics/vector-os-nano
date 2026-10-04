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

import re
from pathlib import Path
from typing import Any

from vector_os_nano.vcli.cognitive.oracle_registry import oracle
from vector_os_nano.vcli.prompt import DEV_ROLE_PROMPT, DEV_TOOL_INSTRUCTIONS
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


@oracle(scope="dev")
def file_exists(path: str) -> bool:
    """True if *path* (relative to cwd) exists."""
    p = _safe_path(path)
    return bool(p is not None and p.exists())


@oracle(scope="dev")
def path_contains(path: str, substr: str) -> bool:
    """True if the text file at *path* contains *substr* (bounded read)."""
    p = _safe_path(path)
    if p is None or not p.is_file():
        return False
    try:
        if p.stat().st_size > _MAX_BYTES_PER_FILE:
            with p.open("r", encoding="utf-8", errors="ignore") as fh:
                return substr in fh.read(_MAX_BYTES_PER_FILE)
        return substr in p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


@oracle(scope="dev")
def grep_count(pattern: str, path: str = ".") -> int:
    """Count regex *pattern* matches in text files under *path* (bounded)."""
    root = _safe_path(path)
    if root is None or not root.exists():
        return 0
    try:
        rx = re.compile(pattern)
    except re.error:
        return 0
    files: list[Path] = []
    if root.is_file():
        files = [root]
    else:
        for p in root.rglob("*"):
            if len(files) >= _MAX_FILES:
                break
            if p.is_file() and not (_SKIP_DIRS & set(p.parts)):
                files.append(p)
    total = 0
    for fp in files:
        try:
            if fp.stat().st_size > _MAX_BYTES_PER_FILE:
                continue
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        total += len(rx.findall(text))
    return total


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
    "grep_count": "grep_count(pattern: str, path: str = '.') -> int  # regex match count under path",
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

    def persona_blocks(self, agent: Any = None) -> tuple[str, str]:
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
