# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Where cross-process flags and logs live (default /tmp).

The sim, bridge, nav scripts and skills coordinate through a few flag files
(vector_nav_active, vector_nav_stalled, vector_reset_pose, ...). They used to be
hardcoded under /tmp, so two sims on one machine drove each other's robots.
Set VECTOR_RUN_DIR per sim (scripts honour ``${VECTOR_RUN_DIR:-/tmp}``).
"""
from __future__ import annotations

import os


def run_path(name: str) -> str:
    """Absolute path of ``name`` inside the run dir (created on first use)."""
    d = os.environ.get("VECTOR_RUN_DIR") or "/tmp"
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)
