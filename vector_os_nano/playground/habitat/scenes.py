# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Scene-reference resolution for habitat scenarios (ADR-009).

Scene assets are EXTERNAL runtime data (never vendored into git — ADR-008).
A scenario carries a backend-generic ``scene_ref``; this module resolves it
to a real file under the data root:

- absolute refs pass through (power users / tests),
- relative refs resolve against ``VECTOR_HABITAT_DATA`` (default: the spike
  download location), fail-loud with the searched path when missing.
"""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_DATA_ROOT = (
    Path.home() / "sandbox" / "habitat-spike" / "data" / "scene_datasets"
)


def habitat_data_root() -> Path:
    """Directory habitat scene refs resolve against (env-overridable)."""
    return Path(os.environ.get("VECTOR_HABITAT_DATA", str(_DEFAULT_DATA_ROOT)))


def resolve_scene_ref(scene_ref: str) -> str:
    """Resolve a scenario ``scene_ref`` to an existing scene file, fail-loud."""
    if not scene_ref:
        raise FileNotFoundError("scenario has an empty scene_ref")
    p = Path(scene_ref)
    if not p.is_absolute():
        p = habitat_data_root() / p
    if not p.exists():
        raise FileNotFoundError(
            f"habitat scene not found: {p} — set VECTOR_HABITAT_DATA or download "
            f"the dataset (license-free test scenes: habitat_test_scenes)"
        )
    return str(p)
