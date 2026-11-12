# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The state directory (charter section 5): the ONE owner of every path in it.

``$VECTOR_HOME`` (default ``~/.vector``) is read on every call, never at import, so a
test or a gate run that points it elsewhere moves everything. Nothing else in the
package calls ``Path.home()`` or ``expanduser`` (tests/unit/test_state_dir.py).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from vector_os_nano.schemas import env

# Charter section 5, entry -> askable. True: the user's own files, which the file tools
# read and (asking first) edit like any path outside the project. False: what Vector
# writes, or what may hold a secret; the file tools refuse it, and every other name
# directly under the store.
LAYOUT: dict[str, bool] = {
    "config.yaml": False,
    "history": False,
    "VECTOR.md": True,
    "personality.md": True,
    "sessions/<id>.jsonl": False,
    "exports/<id>.md": False,
    "memory/<world>/": False,
    "embodiments/<id>/": True,
    "worlds/<id>/": True,
    "assets/<name>/": False,
    "cache/scenes/": False,
    "runs/<stamp>/": False,
}
SCENE_GRAPH, TERRAIN_MAP = "scene_graph.yaml", "terrain_map.npz"  # the files of memory/<world>/

_OLD_MEMORY = "~/.vector_os_nano/worlds"  # where releases before 1.0 kept memory/<world>/


def user_home() -> Path:
    """The account's home directory."""
    return Path.home()


def expand_user(path: str) -> str:
    """*path* with a leading ``~`` expanded: the one spelling for a user-typed path."""
    return os.path.expanduser(path)


def home() -> Path:
    """``$VECTOR_HOME``, else ``~/.vector``."""
    return Path(expand_user(env.read("VECTOR_HOME")))


def config_file() -> Path:
    return home() / "config.yaml"


def history_file() -> Path:
    return home() / "history"


def user_context() -> Path:
    return home() / "VECTOR.md"


def personality() -> Path:
    return home() / "personality.md"


def sessions() -> Path:
    return home() / "sessions"


def export_file(session_id: str) -> Path:
    return home() / "exports" / f"{session_id}.md"


def memory(world: str) -> Path:
    """One world's spatial memory (SCENE_GRAPH, TERRAIN_MAP), shared by every body in it."""
    return home() / "memory" / world


def assets(name: str) -> Path:
    """A robot's fetched assets (scripts/fetch_assets.py)."""
    return home() / "assets" / name


def scene_cache() -> Path:
    """Generated scene files (regenerable)."""
    return home() / "cache" / "scenes"


def migrate_old_memory() -> str:
    """Move each ``~/.vector_os_nano/worlds/<world>/`` into ``memory/<world>/``, once.

    Returns one notice, '' when there was nothing to move. A world whose
    ``memory/<world>/`` already exists, or whose move fails, is left where it is and
    named (shutil.move removes the source only after a complete copy).
    """
    old = Path(expand_user(_OLD_MEMORY))
    if not old.is_dir():
        return ""
    moved, kept = [], []
    for src in sorted(p for p in old.iterdir() if p.is_dir()):
        dst = memory(src.name)
        try:
            if dst.exists():
                raise FileExistsError(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved.append(src.name)
        except OSError:
            kept.append(src.name)
    for empty in (old, old.parent):
        try:
            empty.rmdir()  # only when nothing is left in it
        except OSError:
            break
    # no machine path: the notice reaches the LLM provider inside the sim summary
    parts = []
    if moved:
        parts.append(f"moved the scene memory of {', '.join(moved)} from {_OLD_MEMORY} to $VECTOR_HOME/memory")
    if kept:
        parts.append(f"left {', '.join(kept)} in {_OLD_MEMORY} ($VECTOR_HOME/memory/<world> exists or the move "
                     "failed; merge it by hand)")
    return "; ".join(parts)
