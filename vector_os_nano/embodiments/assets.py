# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The ONE asset-folder resolver (charter section 5, "Assets not shipped").

A robot whose manifest has an ``assets:`` block names its model and policy
relative to an asset folder: the robot's own folder when the files are shipped
there, else ``STATE_DIR/assets/<assets.name>`` as written by
``scripts/fetch_assets.py``. Every declared file must be present and match its
pinned sha256; otherwise the start fails with the exact fetch command.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from vector_os_nano.embodiments.config import _EMBODIMENTS_DIR, AssetsSpec, EmbodimentConfig

# ponytail: ~/.vector until C1's schemas/paths.py owns the state dir ($VECTOR_HOME).
STATE_DIR: Path = Path.home() / ".vector"


class AssetsMissing(RuntimeError):
    """A robot's assets are not shipped and not provisioned (or fail their sha256)."""


def fetch_command(embodiment_id: str) -> str:
    return f"python scripts/fetch_assets.py {embodiment_id}"


def problems(spec: AssetsSpec, folder: Path) -> list[str]:
    """Every file of *spec* that is missing from *folder* or fails its sha256."""
    out = []
    for rel, digest in spec.sha256.items():
        path = folder / rel
        if not path.is_file():
            out.append(f"{rel} missing")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            out.append(f"{rel} does not match its pinned sha256")
    return out


def fetched_dir(spec: AssetsSpec) -> Path:
    return STATE_DIR / "assets" / spec.name


def asset_dir(cfg: EmbodimentConfig) -> Path:
    """The folder *cfg*'s model and policy paths are relative to (raises AssetsMissing)."""
    own = _EMBODIMENTS_DIR / cfg.id
    if cfg.assets is None or not problems(cfg.assets, own):
        return own
    folder = fetched_dir(cfg.assets)
    bad = problems(cfg.assets, folder)
    if bad:
        more = f" and {len(bad) - 1} more" if len(bad) > 1 else ""
        raise AssetsMissing(
            f"{cfg.id} cannot start: its model and policy are not shipped and {folder} "
            f"does not hold them ({bad[0]}{more}); fetch them with: {fetch_command(cfg.id)}"
        )
    return folder
