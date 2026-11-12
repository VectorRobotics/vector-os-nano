# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Perception weights: resolved through the Hugging Face cache only (charter section 5).

Each model class names its ``WEIGHTS``: one :class:`Weights` (``repo_id`` + a pinned
``revision``) per repository it loads. Cached: the snapshot folder, no network.
Not cached and fetching allowed: fetched on first use after ONE notice (what, how
big, where, and that ``HF_HUB_OFFLINE=1`` forbids it). Not cached and
``HF_HUB_OFFLINE=1``: :class:`WeightsUnavailable` with the pre-fetch command.
A load (:func:`loading`) holds huggingface_hub's offline flag for its own duration
only; nothing here writes ``HF_HUB_OFFLINE`` or any other environment variable.
"""
from __future__ import annotations

import fnmatch
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)

FETCH_COMMAND = "python scripts/fetch_assets.py --weights"
# configs, tokenizer and safetensors weights; never the pytorch_model.bin twin
_PATTERNS = ("*.json", "*.safetensors", "*.txt")


@dataclass(frozen=True)
class Weights:
    repo_id: str
    revision: str  # a full commit: what was measured is what loads
    files: tuple[str, ...] = _PATTERNS
    # The branch the loading library itself asks for (transformers resolves some
    # dependencies by "main"); it must point at ``revision`` in the cache.
    ref: str = ""

    def __str__(self) -> str:
        return f"{self.repo_id}@{self.revision[:7]}"


class WeightsUnavailable(RuntimeError):
    """The weights are not cached, and fetching them is forbidden or failed."""


def default_stack() -> tuple[Weights, ...]:
    """Every repository the default perception stack loads: the detector, the segmenter."""
    from vector_os_nano.perception.grounding_dino import GroundingDinoDetector
    from vector_os_nano.perception.tracker import EdgeTAMTracker

    return GroundingDinoDetector.WEIGHTS + EdgeTAMTracker.WEIGHTS


def offline() -> bool:
    """``HF_HUB_OFFLINE`` as huggingface_hub reads it."""
    return os.environ.get("HF_HUB_OFFLINE", "").strip().upper() in {"1", "ON", "YES", "TRUE"}


def cached_dir(w: Weights) -> str | None:
    """The complete cached snapshot of *w*, or None (never touches the network)."""
    from huggingface_hub import snapshot_download, try_to_load_from_cache
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = snapshot_download(w.repo_id, revision=w.revision, allow_patterns=list(w.files),
                                 local_files_only=True)
    except LocalEntryNotFoundError:
        return None
    if w.ref:  # the library's own lookup by ref must land on the pinned snapshot
        hit = try_to_load_from_cache(w.repo_id, w.files[0], revision=w.ref)
        if not isinstance(hit, str) or f"/snapshots/{w.revision}/" not in hit:
            return None
    return path


def status(ws: tuple[Weights, ...]) -> str:
    """A clause for the capability line (no network)."""
    if all(cached_dir(w) for w in ws):
        return "cached"
    if offline():
        return f"unavailable: not cached and HF_HUB_OFFLINE=1 ({FETCH_COMMAND})"
    return "not cached: fetched on first use"


def resolve(ws: tuple[Weights, ...]) -> list[str]:
    """The cached snapshot folder of each of *ws*; what is missing is fetched first,
    when allowed, after ONE notice for all of it."""
    missing = [w for w in ws if not cached_dir(w)]
    if missing:
        _fetch(missing)
    return [cached_dir(w) or _moved(w) for w in ws]


@contextmanager
def loading(ws: tuple[Weights, ...]):
    """:func:`resolve` *ws*, then hold the hub offline while the library loads them.

    Everything the load reads is then cached, yet transformers still asks the hub on
    its own (EdgeTAM builds a default config, which looks up the timm backbone by
    "main", whenever a config is printed or diffed): offline, it reads the cache.
    """
    from huggingface_hub import constants

    paths = resolve(ws)
    # ponytail: process-wide for the load's duration; a hub call on another thread meanwhile reads the cache only.
    was, constants.HF_HUB_OFFLINE = constants.HF_HUB_OFFLINE, True
    try:
        yield paths
    finally:
        constants.HF_HUB_OFFLINE = was


def local_dir(w: Weights) -> str:
    return resolve((w,))[0]


def _fetch(missing: list[Weights]) -> None:
    from huggingface_hub import HfApi, constants, snapshot_download

    what = ", ".join(map(str, missing))
    if offline():
        raise WeightsUnavailable(
            f"{what} {'is' if len(missing) == 1 else 'are'} not in the Hugging Face cache "
            f"({constants.HF_HUB_CACHE}) and HF_HUB_OFFLINE=1 forbids fetching; pre-fetch with: "
            f"{FETCH_COMMAND}"
        )
    try:
        size = sum(s.size or 0 for w in missing
                   for s in HfApi().model_info(w.repo_id, revision=w.revision, files_metadata=True).siblings
                   if any(fnmatch.fnmatch(s.rfilename, p) for p in w.files))
        logger.warning(
            "[weights] fetching %s (%.1f MB) into %s on first use; HF_HUB_OFFLINE=1 forbids "
            "this (pre-fetch with: %s)", what, size / 1e6, constants.HF_HUB_CACHE, FETCH_COMMAND,
        )
        for w in missing:
            snapshot_download(w.repo_id, revision=w.ref or w.revision, allow_patterns=list(w.files))
    except Exception as exc:  # noqa: BLE001 — one honest error, whatever the hub said
        raise WeightsUnavailable(
            f"fetching {what} into {constants.HF_HUB_CACHE} failed ({type(exc).__name__}: {exc}); "
            f"pre-fetch with: {FETCH_COMMAND}"
        ) from exc


def _moved(w: Weights) -> str:
    raise WeightsUnavailable(
        f"{w.repo_id} {w.ref} no longer points at the pinned {w.revision[:7]}: "
        "the pin needs a review before it can load"
    )
