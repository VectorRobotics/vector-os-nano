# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Embodiment registry: every ``embodiments/<id>/robot.yaml`` is a startable body.

Adding a robot = adding a manifest directory (Rule 11); the NL ``start_simulation``
enum and the ``--embodiment`` CLI choices are generated from this listing.
"""
from __future__ import annotations

from vector_os_nano.schemas import robot as _config
from vector_os_nano.schemas.robot import EmbodimentConfig, EmbodimentConfigError, load_embodiment_config
from typing import Any


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
    return validate(load_embodiment_config(embodiment_id), ctx=f"{embodiment_id}/robot.yaml")


def validate(cfg: EmbodimentConfig, *, ctx: str = "robot.yaml") -> EmbodimentConfig:
    """Validate owner-specific policy data against the installed registries before startup."""
    from vector_os_nano.sim.backends import BACKENDS

    if cfg.base is not None:
        policy = cfg.base.policy
        pctx = f"{ctx}.base.policy"
        for name in policy.by_backend:
            if name not in BACKENDS:
                raise EmbodimentConfigError(f"{pctx}.by_backend: unknown backend {name!r}; valid: {sorted(BACKENDS)}")
        _check_policy_row(policy.ref, policy.params, pctx)
        for name in policy.by_backend:
            _check_policy_row(*policy.on(name), f"{pctx}.by_backend.{name}")
    return cfg


def _check_policy_row(ref: str, params: dict[str, Any], ctx: str) -> None:
    """The row *ref* exists and *params* are exactly its keys (``sim.policies.owned``)."""
    from vector_os_nano.sim.policies import owned  # a name table: no simulator import

    try:
        owned(ref, params)
    except KeyError as exc:
        raise EmbodimentConfigError(f"{ctx}.ref: {exc.args[0]}") from None
    except ValueError as exc:
        raise EmbodimentConfigError(f"{ctx}: {exc}") from None
