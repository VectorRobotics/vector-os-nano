# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""perception:edgetam-provisioning-silent-fallback + offline-env-leak: where EdgeTAM
loads from, and that loading the models never mutates the process environment."""
from __future__ import annotations

from vector_os_nano.perception.tracker import EdgeTAMTracker


def test_weights_resolve_from_the_model_dir_else_the_hf_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTOR_MODEL_DIR", str(tmp_path))
    assert EdgeTAMTracker._resolve_model_source("yonigozlan/EdgeTAM-hf") == "yonigozlan/EdgeTAM-hf"
    (tmp_path / "EdgeTAM-hf").mkdir()
    (tmp_path / "EdgeTAM-hf" / "config.json").write_text("{}")
    assert EdgeTAMTracker._resolve_model_source("yonigozlan/EdgeTAM-hf") == str(tmp_path / "EdgeTAM-hf")
    assert EdgeTAMTracker.weights_available("yonigozlan/EdgeTAM-hf")
    assert not EdgeTAMTracker.weights_available("nobody/absent-model")


def test_no_module_sets_the_offline_flags_process_wide():
    import inspect

    from vector_os_nano.perception import grounding_dino, tracker

    for mod in (grounding_dino, tracker):
        src = inspect.getsource(mod)
        assert "HF_HUB_OFFLINE" not in src and "TRANSFORMERS_OFFLINE" not in src, mod.__name__
