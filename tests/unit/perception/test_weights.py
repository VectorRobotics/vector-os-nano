# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""C0 — perception weights through the Hugging Face cache only (charter section 5).

The hub client is faked: cached -> no network; not cached + HF_HUB_OFFLINE=1 -> the
exact pre-fetch sentence; not cached + allowed -> ONE notice, then the pinned fetch.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from vector_os_nano.perception import weights
from vector_os_nano.perception.grounding_dino import GroundingDinoDetector
from vector_os_nano.perception.tracker import EdgeTAMTracker

W = weights.Weights("org/model", "0123456789abcdef0123456789abcdef01234567")


class _Hub:
    """A fake hub: *cached* decides the local lookup; every call is recorded."""

    def __init__(self, cached: bool) -> None:
        self.cached, self.calls = cached, []

    def snapshot_download(self, repo_id, *, revision, allow_patterns, local_files_only=False):
        self.calls.append(("snapshot", repo_id, revision, local_files_only))
        if local_files_only and not self.cached:
            raise LocalEntryNotFoundError("not cached")
        return f"/cache/{repo_id}/{revision}"

    def model_info(self, repo_id, *, revision, files_metadata):
        self.calls.append(("info", repo_id, revision))
        return SimpleNamespace(siblings=[
            SimpleNamespace(rfilename="model.safetensors", size=55_000_000),
            SimpleNamespace(rfilename="config.json", size=1_000_000),
            SimpleNamespace(rfilename="pytorch_model.bin", size=690_000_000),  # not fetched
        ])


@pytest.fixture
def hub(monkeypatch):
    def make(cached: bool) -> _Hub:
        h = _Hub(cached)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", h.snapshot_download)
        monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(model_info=h.model_info))
        return h
    return make


def test_each_default_model_class_names_a_pinned_revision() -> None:
    assert weights.default_stack() == (GroundingDinoDetector.WEIGHTS, EdgeTAMTracker.WEIGHTS)
    for w in weights.default_stack():
        assert re.fullmatch(r"[0-9a-f]{40}", w.revision), w
    assert GroundingDinoDetector.WEIGHTS.repo_id == "IDEA-Research/grounding-dino-tiny"
    assert EdgeTAMTracker.WEIGHTS.repo_id == "yonigozlan/EdgeTAM-hf"


def test_cached_weights_load_without_any_network_call(hub, monkeypatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    h = hub(cached=True)
    assert weights.local_dir(W) == f"/cache/org/model/{W.revision}"
    assert h.calls == [("snapshot", "org/model", W.revision, True)]
    assert weights.status(W) == "cached"


def test_offline_and_not_cached_fails_with_the_prefetch_command(hub, monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    h = hub(cached=False)
    with pytest.raises(weights.WeightsUnavailable) as exc:
        weights.local_dir(W)
    msg = str(exc.value)
    assert msg.startswith("org/model@0123456 is not in the Hugging Face cache (")
    assert msg.endswith("HF_HUB_OFFLINE=1 forbids fetching it; pre-fetch it with: "
                        "python scripts/fetch_assets.py --weights")
    assert [c[0] for c in h.calls] == ["snapshot"]  # never asked the hub
    assert "HF_HUB_OFFLINE=1" in weights.status(W)


def test_first_use_fetch_prints_one_notice_then_fetches_the_pin(hub, monkeypatch, caplog) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    h = hub(cached=False)
    with caplog.at_level(logging.INFO, logger="vector_os_nano.perception.weights"):
        assert weights.local_dir(W) == f"/cache/org/model/{W.revision}"
    notices = [r for r in caplog.records if r.name == "vector_os_nano.perception.weights"]
    assert len(notices) == 1 and notices[0].levelno == logging.WARNING
    text = notices[0].getMessage()
    # what, how big (the .bin twin is not counted), where, and the way to forbid it
    assert "org/model@0123456" in text and "(56 MB)" in text
    assert str(huggingface_hub.constants.HF_HUB_CACHE) in text and "HF_HUB_OFFLINE=1 forbids" in text
    assert h.calls[-1] == ("snapshot", "org/model", W.revision, False)
    assert weights.status(W) == "not cached: fetched on first use"


def test_a_failed_fetch_is_one_honest_error(hub, monkeypatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    h = hub(cached=False)

    def down(*_a, **_k):
        raise OSError("connection refused")

    h.model_info = down
    with pytest.raises(weights.WeightsUnavailable, match="fetching org/model@0123456 into .* failed "
                       r"\(OSError: connection refused\); pre-fetch it with"):
        weights.local_dir(W)


def test_the_models_load_from_the_resolved_snapshot_or_raise(monkeypatch) -> None:
    pytest.importorskip("transformers")
    from vector_os_nano.perception import grounding_dino, tracker

    def unavailable(w):
        raise weights.WeightsUnavailable(f"{w} unavailable")

    for mod, load in ((grounding_dino, lambda: GroundingDinoDetector()._ensure_loaded()),
                      (tracker, lambda: EdgeTAMTracker()._load_model())):
        monkeypatch.setattr(mod, "local_dir", unavailable)
        with pytest.raises(weights.WeightsUnavailable):
            load()


def test_the_capability_line_says_which_model_or_why_not(monkeypatch) -> None:
    pytest.importorskip("transformers")
    from vector_os_nano.perception import rynnbrain
    from vector_os_nano.sim import session

    monkeypatch.setattr(rynnbrain.RynnBrainDetector, "available", lambda self: False)
    monkeypatch.setattr(weights, "status", lambda w: "cached" if "dino" in w.repo_id else "UNAVAILABLE")
    agent = SimpleNamespace(_perception=SimpleNamespace(has_self_mask=True), _vlm=None)
    line = session._perception_note(agent, "no OPENROUTER_API_KEY or VECTOR_VLM_URL")
    assert re.fullmatch(r"perception: detector=grounding-dino@(cuda|cpu) \(cached\), segmenter=edgetam "
                        r"\(UNAVAILABLE\), self_mask=yes, vlm=none \(no OPENROUTER_API_KEY or "
                        r"VECTOR_VLM_URL\), rynnbrain=down", line), line
    g1 = session._perception_note(agent, "g1 has no scene VLM", segmenter=False)
    assert "segmenter" not in g1 and "vlm=none (g1 has no scene VLM)" in g1


def test_nothing_in_the_package_writes_the_offline_flags() -> None:
    """Loading a model never sets HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE process-wide."""
    pkg = Path(weights.__file__).resolve().parents[1]
    write = re.compile(r"""(environ\[|setdefault\(|putenv\(|setenv\()\s*["'](HF_HUB_OFFLINE|TRANSFORMERS_OFFLINE)""")
    hits = [str(p) for p in pkg.rglob("*.py") if write.search(p.read_text(encoding="utf-8"))]
    assert hits == []
    for legacy in ("VECTOR_MODEL_DIR", "vector_os/models", '"vector_os" / "models"'):
        assert not [p for p in pkg.rglob("*.py") if legacy in p.read_text(encoding="utf-8")], legacy
