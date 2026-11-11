# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Perception weights: resolved through the Hugging Face cache only (charter section 5).

Each model class names its :class:`Weights` (``repo_id`` + a pinned ``revision``).
Cached: the snapshot folder, no network. Not cached and fetching allowed: fetched
on first use after ONE notice (what, how big, where, and that ``HF_HUB_OFFLINE=1``
forbids it). Not cached and ``HF_HUB_OFFLINE=1``: :class:`WeightsUnavailable` with
the pre-fetch command. Nothing here writes ``HF_HUB_OFFLINE`` or any other variable.
"""
from __future__ import annotations

import fnmatch
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

FETCH_COMMAND = "python scripts/fetch_assets.py --weights"
# configs, tokenizer and safetensors weights; never the pytorch_model.bin twin
_PATTERNS = ("*.json", "*.safetensors", "*.txt")


@dataclass(frozen=True)
class Weights:
    repo_id: str
    revision: str  # a full commit: what was measured is what loads

    def __str__(self) -> str:
        return f"{self.repo_id}@{self.revision[:7]}"


class WeightsUnavailable(RuntimeError):
    """The weights are not cached, and fetching them is forbidden or failed."""


def default_stack() -> tuple[Weights, ...]:
    """The weights of the default perception stack: the detector, then the segmenter."""
    from vector_os_nano.perception.grounding_dino import GroundingDinoDetector
    from vector_os_nano.perception.tracker import EdgeTAMTracker

    return (GroundingDinoDetector.WEIGHTS, EdgeTAMTracker.WEIGHTS)


def offline() -> bool:
    """``HF_HUB_OFFLINE`` as huggingface_hub reads it."""
    return os.environ.get("HF_HUB_OFFLINE", "").strip().upper() in {"1", "ON", "YES", "TRUE"}


def cached_dir(w: Weights) -> str | None:
    """The complete cached snapshot of *w*, or None (never touches the network)."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(w.repo_id, revision=w.revision, allow_patterns=list(_PATTERNS),
                                 local_files_only=True)
    except LocalEntryNotFoundError:
        return None


def status(w: Weights) -> str:
    """One word or clause for the capability line (no network)."""
    if cached_dir(w):
        return "cached"
    if offline():
        return f"unavailable: not cached and HF_HUB_OFFLINE=1 ({FETCH_COMMAND})"
    return "not cached: fetched on first use"


def local_dir(w: Weights) -> str:
    """The cached snapshot folder of *w*, fetched first when allowed (one notice)."""
    path = cached_dir(w)
    if path:
        return path
    from huggingface_hub import HfApi, constants, snapshot_download

    if offline():
        raise WeightsUnavailable(
            f"{w} is not in the Hugging Face cache ({constants.HF_HUB_CACHE}) and HF_HUB_OFFLINE=1 "
            f"forbids fetching it; pre-fetch it with: {FETCH_COMMAND}"
        )
    try:
        info = HfApi().model_info(w.repo_id, revision=w.revision, files_metadata=True)
        size = sum(s.size or 0 for s in info.siblings
                   if any(fnmatch.fnmatch(s.rfilename, p) for p in _PATTERNS))
        logger.warning(
            "[weights] fetching %s (%.0f MB) into %s on first use; HF_HUB_OFFLINE=1 forbids "
            "this (pre-fetch with: %s)", w, size / 1e6, constants.HF_HUB_CACHE, FETCH_COMMAND,
        )
        return snapshot_download(w.repo_id, revision=w.revision, allow_patterns=list(_PATTERNS))
    except Exception as exc:  # noqa: BLE001 — one honest error, whatever the hub said
        raise WeightsUnavailable(
            f"fetching {w} into {constants.HF_HUB_CACHE} failed ({type(exc).__name__}: {exc}); "
            f"pre-fetch it with: {FETCH_COMMAND}"
        ) from exc
