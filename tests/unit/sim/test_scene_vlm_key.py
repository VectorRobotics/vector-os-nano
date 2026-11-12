# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""C1a / R5V-D6 — the scene VLM's key comes from the environment like every provider's.

A key left in config/user.yaml (llm.api_key, vlm.api_key) is neither read nor sent to
the OpenRouter VLM; one notice names the variable to set instead, never the value.
"""
from __future__ import annotations

from vector_os_nano.sim.session import _scene_vlm_config

SECRET = "sk-or-SECRET-in-user-yaml"


def test_a_key_in_user_yaml_is_ignored_with_one_notice(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = {"llm": {"api_key": SECRET}, "vlm": {"api_key": SECRET, "model": "m"}}
    key, vlm, note = _scene_vlm_config(cfg, None)
    assert key == "" and vlm == {"model": "m"}  # the VLM is built from these: no key reaches it
    assert note == ("config/user.yaml llm.api_key and vlm.api_key ignored (keys come only from the "
                    "environment): set OPENROUTER_API_KEY")
    assert SECRET not in note
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-env")
    assert _scene_vlm_config(cfg, None)[0] == "or-env"
    assert cfg["vlm"]["api_key"] == SECRET  # the caller's dict is left as it was


def test_no_key_in_the_file_no_notice(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-env")
    assert _scene_vlm_config({"llm": {"provider": "x"}}, None) == ("or-env", {}, "")
    assert _scene_vlm_config({}, None) == ("or-env", {}, "")
