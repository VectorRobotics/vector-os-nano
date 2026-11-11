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
    """A fake hub over a fake cache: {(repo, commit)} snapshots and {(repo, ref): commit}."""

    def __init__(self, cached: bool, upstream_main: str = W.revision) -> None:
        self.snapshots = {("org/model", W.revision)} if cached else set()
        self.refs: dict = {}
        self.upstream_main, self.calls = upstream_main, []

    def snapshot_download(self, repo_id, *, revision, allow_patterns, local_files_only=False):
        self.calls.append(("snapshot", repo_id, revision, local_files_only))
        if local_files_only:
            if (repo_id, revision) not in self.snapshots:
                raise LocalEntryNotFoundError("not cached")
        elif revision == "main":
            revision = self.refs[(repo_id, "main")] = self.upstream_main
        self.snapshots.add((repo_id, revision))
        return f"/cache/{repo_id}/snapshots/{revision}"

    def try_to_load_from_cache(self, repo_id, filename, *, revision):
        commit = self.refs.get((repo_id, revision))
        return f"/cache/{repo_id}/snapshots/{commit}/{filename}" if commit else None

    def model_info(self, repo_id, *, revision, files_metadata):
        self.calls.append(("info", repo_id, revision))
        return SimpleNamespace(siblings=[
            SimpleNamespace(rfilename="model.safetensors", size=55_000_000),
            SimpleNamespace(rfilename="config.json", size=1_000_000),
            SimpleNamespace(rfilename="pytorch_model.bin", size=690_000_000),  # not fetched
        ])


@pytest.fixture
def hub(monkeypatch):
    def make(cached: bool, **kw) -> _Hub:
        h = _Hub(cached, **kw)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", h.snapshot_download)
        monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", h.try_to_load_from_cache)
        monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(model_info=h.model_info))
        return h
    return make


def test_each_default_model_class_names_pinned_revisions() -> None:
    assert weights.default_stack() == GroundingDinoDetector.WEIGHTS + EdgeTAMTracker.WEIGHTS
    for w in weights.default_stack():
        assert re.fullmatch(r"[0-9a-f]{40}", w.revision), w
    assert [w.repo_id for w in weights.default_stack()] == [
        "IDEA-Research/grounding-dino-tiny", "yonigozlan/EdgeTAM-hf", "timm/repvit_m1.dist_in1k"]


def test_cached_weights_load_without_any_network_call(hub, monkeypatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    h = hub(cached=True)
    assert weights.local_dir(W) == f"/cache/org/model/snapshots/{W.revision}"
    assert h.calls and all(c == ("snapshot", "org/model", W.revision, True) for c in h.calls)  # local only
    assert weights.status((W,)) == "cached"


def test_offline_and_not_cached_fails_with_the_prefetch_command(hub, monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    h = hub(cached=False)
    with pytest.raises(weights.WeightsUnavailable) as exc:
        weights.local_dir(W)
    msg = str(exc.value)
    assert msg.startswith("org/model@0123456 is not in the Hugging Face cache (")
    assert msg.endswith("HF_HUB_OFFLINE=1 forbids fetching; pre-fetch with: "
                        "python scripts/fetch_assets.py --weights")
    assert [c[0] for c in h.calls] == ["snapshot"]  # never asked the hub
    assert "HF_HUB_OFFLINE=1" in weights.status((W,))


def test_first_use_fetch_prints_one_notice_then_fetches_the_pin(hub, monkeypatch, caplog) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    h = hub(cached=False)
    assert weights.status((W,)) == "not cached: fetched on first use"
    with caplog.at_level(logging.INFO, logger="vector_os_nano.perception.weights"):
        assert weights.local_dir(W) == f"/cache/org/model/snapshots/{W.revision}"
    notices = [r for r in caplog.records if r.name == "vector_os_nano.perception.weights"]
    assert len(notices) == 1 and notices[0].levelno == logging.WARNING
    text = notices[0].getMessage()
    # what, how big (the .bin twin is not counted), where, and the way to forbid it
    assert "org/model@0123456" in text and "(56.0 MB)" in text
    assert str(huggingface_hub.constants.HF_HUB_CACHE) in text and "HF_HUB_OFFLINE=1 forbids" in text
    assert ("snapshot", "org/model", W.revision, False) in h.calls
    assert weights.status((W,)) == "cached"


def test_a_ref_resolved_dependency_must_land_on_its_pin(hub, monkeypatch) -> None:
    """A dependency transformers asks for by "main" is cached only if main is the pin."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    dep = weights.Weights("org/model", W.revision, files=("config.json",), ref="main")
    h = hub(cached=True)  # the commit is cached but no ref points at it: not usable
    assert weights.cached_dir(dep) is None
    assert weights.local_dir(dep).endswith(W.revision) and h.refs[("org/model", "main")] == W.revision
    moved = hub(cached=False, upstream_main="f" * 40)
    with pytest.raises(weights.WeightsUnavailable, match="main no longer points at the pinned 0123456"):
        weights.local_dir(dep)
    assert moved.refs[("org/model", "main")] == "f" * 40


def test_one_notice_covers_every_missing_repository(hub, monkeypatch, caplog) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    hub(cached=False)
    other = weights.Weights("org/other", "f" * 40)
    with caplog.at_level(logging.INFO, logger="vector_os_nano.perception.weights"):
        assert len(weights.resolve((W, other))) == 2
    notices = [r.getMessage() for r in caplog.records if r.name == "vector_os_nano.perception.weights"]
    assert len(notices) == 1 and "org/model@0123456, org/other@fffffff (112.0 MB)" in notices[0]


def test_a_failed_fetch_is_one_honest_error(hub, monkeypatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    h = hub(cached=False)

    def down(*_a, **_k):
        raise OSError("connection refused")

    h.model_info = down
    with pytest.raises(weights.WeightsUnavailable, match="fetching org/model@0123456 into .* failed "
                       r"\(OSError: connection refused\); pre-fetch with"):
        weights.local_dir(W)


def test_the_models_load_from_the_resolved_snapshot_or_raise(monkeypatch) -> None:
    pytest.importorskip("transformers")
    def unavailable(ws):
        raise weights.WeightsUnavailable(f"{ws[0]} unavailable")

    monkeypatch.setattr(weights, "resolve", unavailable)
    for load in (lambda: GroundingDinoDetector()._ensure_loaded(),
                 lambda: EdgeTAMTracker()._load_model()):
        with pytest.raises(weights.WeightsUnavailable):
            load()


def test_the_capability_line_says_which_model_or_why_not(monkeypatch) -> None:
    pytest.importorskip("transformers")
    from vector_os_nano.perception import rynnbrain
    from vector_os_nano.sim import session

    monkeypatch.setattr(rynnbrain.RynnBrainDetector, "available", lambda self: False)
    monkeypatch.setattr(weights, "status", lambda ws: "cached" if "dino" in ws[0].repo_id else "UNAVAILABLE")
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


def test_a_real_cached_load_opens_no_socket_with_the_offline_flag_unset(monkeypatch) -> None:
    """Cached means no network, also inside transformers (EdgeTAM's default vision config
    asks the hub for the timm backbone by "main" every time a config is printed or diffed)."""
    pytest.importorskip("transformers")
    import socket

    import torch

    if weights.status(weights.default_stack()) != "cached":
        pytest.skip("the default perception weights are not in the Hugging Face cache")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)  # as an unset env reads
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # cpu: leave the GPU alone
    attempts = []

    def blocked(*args, **_kw):
        attempts.append(args[:1])
        raise OSError("network blocked by the test")

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    GroundingDinoDetector()._ensure_loaded()
    EdgeTAMTracker(device="cpu")._load_model()
    assert attempts == []
    assert huggingface_hub.constants.HF_HUB_OFFLINE is False  # the scope ended with the load
