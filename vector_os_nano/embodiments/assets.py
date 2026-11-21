# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The ONE asset-folder resolver (charter section 5, "Assets not shipped").

A robot whose manifest has an ``assets:`` block names its model and policy
relative to an asset folder: the robot's own folder when the files are shipped
there, else ``$VECTOR_HOME/assets/<assets.name>`` as written by
``scripts/fetch_assets.py``. Every declared file must be present and match its
pinned sha256; otherwise the start fails with the exact fetch command.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from vector_os_nano.schemas.robot import AssetsSpec, EmbodimentConfig
from vector_os_nano.schemas import paths


class AssetsMissing(RuntimeError):
    """A robot's assets are not shipped and not provisioned (or fail their sha256)."""


def fetch_command(embodiment_id: str) -> str:
    return f"python scripts/fetch_assets.py {embodiment_id}"


MISSING, MISMATCH = "missing", "does not match its pinned sha256"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def problems(spec: AssetsSpec, folder: Path) -> dict[str, str]:
    """``{file: MISSING | MISMATCH}`` for every file of *spec* not verified in *folder*."""
    out = {}
    for rel, digest in spec.sha256.items():
        path = folder / rel
        if not path.is_file():
            out[rel] = MISSING
        elif sha256(path) != digest:
            out[rel] = MISMATCH
    return out


def fetched_dir(spec: AssetsSpec) -> Path:
    return paths.assets(spec.name)


def asset_dir(cfg: EmbodimentConfig) -> Path:
    """The folder *cfg*'s model and policy paths are relative to (raises AssetsMissing)."""
    own = cfg.path.parent
    if cfg.assets is None or not problems(cfg.assets, own):
        return own
    folder = fetched_dir(cfg.assets)
    bad = problems(cfg.assets, folder)
    if bad:
        rel, why = next(iter(bad.items()))
        more = f" and {len(bad) - 1} more" if len(bad) > 1 else ""
        # no machine path: this sentence reaches the LLM provider as a tool result
        raise AssetsMissing(
            f"{cfg.id} cannot start: its model and policy are neither shipped nor fetched "
            f"({rel} {why}{more}); fetch them with: {fetch_command(cfg.id)}"
        )
    return folder


def model_root(cfg: EmbodimentConfig) -> Path:
    """Resolve model and policy files in the robot folder, then verified fetched assets."""
    return model_root(cfg.compose.base) if cfg.compose else asset_dir(cfg)


def policy_root(cfg: EmbodimentConfig) -> Path:
    """Inherited policies retain their base folder; replacement blocks use this robot's."""
    if cfg.compose and cfg.base is cfg.compose.base.base:
        return policy_root(cfg.compose.base)
    return asset_dir(cfg)
