# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Explicit credentials (charter section 4): the provider is chosen by flag > the
config file's ``llm:`` block > the default, and its key comes ONLY from the
environment. The config file never supplies a key; another application's credentials
are never read.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vector_os_nano.vcli import config as cfg

_ENV = ("DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "MY_KEY")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway HOME with no .env on the way: only what a test writes is read."""
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VECTOR_HOME", str(tmp_path / ".vector"))
    for var in _ENV:
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _config(home: Path, text: str) -> Path:
    path = home / ".vector" / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("schema: 1\n" + text)
    return path


def test_the_default_provider_reads_its_own_variable(home, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")  # present, but not the chosen provider
    c = cfg.resolve_credentials()
    assert (c.provider, c.key_env, c.api_key) == ("deepseek", "DEEPSEEK_API_KEY", "ds-key")
    assert c.model == "deepseek-v4-flash" and c.base_url == "https://api.deepseek.com"


def test_flag_over_config_over_default(home, monkeypatch) -> None:
    _config(home, "llm:\n  provider: anthropic\n  model: claude-haiku-4-5\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    c = cfg.resolve_credentials()
    assert (c.provider, c.model, c.api_key) == ("anthropic", "claude-haiku-4-5", "ant-key")
    # A flag picks another provider: that provider's key and defaults, not the file's model.
    c = cfg.resolve_credentials("openrouter")
    assert (c.provider, c.model, c.api_key) == ("openrouter", "anthropic/claude-sonnet-4-6", "or-key")
    assert cfg.resolve_credentials("anthropic", "claude-opus-4-1").model == "claude-opus-4-1"


@pytest.mark.parametrize(("text", "named"), [
    ("anthropic_api_key: {s}\n", "anthropic_api_key looks like a secret.*set ANTHROPIC_API_KEY"),
    ("llm:\n  provider: anthropic\n  api_key: {s}\n", "llm.api_key looks like a secret.*set ANTHROPIC_API_KEY"),
    ("llm:\n  provider: openai_compat\n  api_key_env: MY_KEY\n  token: {s}\n", "llm.token .*set MY_KEY"),
])
def test_a_key_in_the_config_file_is_refused_never_used_nor_printed(home, monkeypatch, text, named) -> None:
    """cli-entry:config-keys-world-readable / docs-packaging:credential-handling: the file
    holds no key; one that does is refused naming the variable to set instead, its value
    never echoed, and the file is left untouched."""
    secret = "sk-ant-api03-SECRET-in-the-file"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
    path = _config(home, text.format(s=secret))
    before = path.read_bytes()
    with pytest.raises(ValueError, match=named) as err:
        cfg.resolve_credentials()
    assert str(path) in str(err.value) and secret not in str(err.value)
    assert path.read_bytes() == before


def test_another_applications_token_is_never_borrowed(home) -> None:
    """cli-entry:oauth-outranks-api-key / cli-entry:oauth-token-as-x-api-key: a logged-in
    Claude Code (or a vector OAuth file) never supplies the key."""
    for f in (home / ".claude" / ".credentials.json", home / ".vector" / "oauth_credentials.json"):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-borrowed", "expiresAt": 0},
                                 "accessToken": "sk-ant-oat01-borrowed"}))
    c = cfg.resolve_credentials("anthropic")
    assert c.api_key == "" and "ANTHROPIC_API_KEY" in c.instruction()


def test_the_key_variable_can_be_named_in_the_config(home, monkeypatch) -> None:
    _config(home, "llm:\n  provider: openai_compat\n  model: qwen\n  base_url: http://127.0.0.1:8000/v1\n"
                  "  api_key_env: MY_KEY\n")
    monkeypatch.setenv("MY_KEY", "local")
    c = cfg.resolve_credentials()
    assert (c.provider, c.model, c.base_url, c.key_env, c.api_key) == (
        "openai_compat", "qwen", "http://127.0.0.1:8000/v1", "MY_KEY", "local")


def test_fail_loud_on_an_unknown_provider_or_an_unaddressed_compat_endpoint(home) -> None:
    with pytest.raises(ValueError, match="one of"):
        cfg.resolve_credentials("claude")
    # an OpenAI-compatible key is never sent to a default host
    with pytest.raises(ValueError, match="base_url"):
        cfg.resolve_credentials("openai_compat", "m")


def test_no_key_names_the_exact_variable(home) -> None:
    c = cfg.resolve_credentials("openrouter")
    assert c.api_key == "" and c.instruction().startswith("No OPENROUTER_API_KEY set")
