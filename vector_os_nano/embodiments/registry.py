# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Embodiment registry: every ``embodiments/<id>/robot.yaml`` is a startable body.

Adding a robot = adding a manifest directory (Rule 11); the NL ``start_simulation``
enum and the ``--embodiment`` CLI choices are generated from this listing.
"""
from __future__ import annotations

from vector_os_nano.embodiments import config as _config
from vector_os_nano.embodiments.config import EmbodimentConfig, load_embodiment_config


def embodiment_ids() -> list[str]:
    """Ids of every embodiment that ships a manifest, sorted."""
    return sorted(p.parent.name for p in _config._EMBODIMENTS_DIR.glob("*/robot.yaml"))


def load(embodiment_id: str) -> EmbodimentConfig:
    """Load one manifest; unknown id fails loud with the valid set (Rule 8)."""
    ids = embodiment_ids()
    if embodiment_id not in ids:
        raise _config.EmbodimentConfigError(
            f"unknown embodiment '{embodiment_id}'; valid: {ids}"
        )
    return load_embodiment_config(embodiment_id)
