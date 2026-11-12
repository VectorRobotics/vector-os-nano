# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""LLM credentials: the provider is a choice, its key comes from the environment.

``provider``, ``model`` and ``base_url`` resolve as: CLI flag > the ``llm:`` block of
``~/.vector/config.yaml`` > the built-in default. The key is read ONLY from the
provider's environment variable (a repo-root ``.env`` is loaded first); the config
file never holds a key, and no other application's credentials are ever read.

    # ~/.vector/config.yaml
    llm:
      provider: deepseek        # deepseek | openrouter | anthropic | openai_compat
      model: deepseek-v4-flash  # optional: the provider's default otherwise
      base_url: ...             # optional
      api_key_env: MY_KEY       # optional: read the key from this variable instead
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path.home() / ".vector" / "config.yaml"


@dataclass(frozen=True)
class Provider:
    key_env: str
    model: str
    base_url: str | None


PROVIDERS: dict[str, Provider] = {
    "deepseek": Provider("DEEPSEEK_API_KEY", "deepseek-v4-flash", "https://api.deepseek.com"),
    "openrouter": Provider("OPENROUTER_API_KEY", "anthropic/claude-sonnet-4-6", "https://openrouter.ai/api/v1"),
    "anthropic": Provider("ANTHROPIC_API_KEY", "claude-sonnet-4-6", None),
    "openai_compat": Provider("OPENAI_API_KEY", "", None),
}
DEFAULT_PROVIDER = "deepseek"

# Keys an older config file held; each names the variable it belongs in now.
_LEGACY_KEYS = {"anthropic_api_key": "ANTHROPIC_API_KEY", "openrouter_api_key": "OPENROUTER_API_KEY",
                "deepseek_api_key": "DEEPSEEK_API_KEY"}


@dataclass(frozen=True)
class Credentials:
    """The resolved LLM selection. ``api_key`` is '' when its variable is unset."""

    provider: str
    model: str
    base_url: str | None
    key_env: str
    api_key: str = field(default="", repr=False)
    notice: str = ""  # one line about an old config file, else ''

    def instruction(self) -> str:
        """What to do when the key is missing: the exact variable to set."""
        return (f"No {self.key_env} set: the {self.provider} provider reads its key from the "
                f"environment. Add {self.key_env}=<your key> to .env in the repo root (or export "
                f"it), then type /login. Another provider: vector-cli --provider <name>, or "
                f"llm.provider in {_CONFIG_PATH}.")


def load_config() -> dict[str, Any]:
    """The config file as a dict ({} when it is absent or unreadable)."""
    if not _CONFIG_PATH.exists():
        return {}
    try:
        import yaml  # noqa: PLC0415

        raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — an unreadable file is an absent one
        return {}
    return raw if isinstance(raw, dict) else {}


def _legacy_notice(config: dict[str, Any], key_env: str) -> str:
    """One line naming key material an old config file still holds (never its value)."""
    llm = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    moves = [f"{k} -> {v}" for k, v in _LEGACY_KEYS.items() if config.get(k)]
    if llm.get("api_key"):
        moves.append(f"llm.api_key -> {key_env}")
    if not moves:
        return ""
    return (f"{_CONFIG_PATH} holds an API key, which is not read (keys come only from the "
            f"environment); move it: {', '.join(moves)}. The file is left as it is.")


def resolve_credentials(
    provider: str | None = None, model: str | None = None, base_url: str | None = None,
) -> Credentials:
    """Resolve the LLM selection (flag > config ``llm:`` > default) and read its key.

    A repo-root ``.env`` is loaded into the environment first (an already-set
    variable wins). An unknown provider raises with the valid set.
    """
    from dotenv import load_dotenv  # noqa: PLC0415

    load_dotenv()
    config = load_config()
    llm = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    name = provider or llm.get("provider") or DEFAULT_PROVIDER
    if name not in PROVIDERS:
        raise ValueError(f"unknown LLM provider {name!r}; one of {sorted(PROVIDERS)}")
    spec = PROVIDERS[name]
    # The file's model / base_url / key variable belong to the file's provider: a flag
    # that picks another provider starts from that provider's defaults.
    own = llm if name == (llm.get("provider") or DEFAULT_PROVIDER) else {}
    key_env = str(own.get("api_key_env") or spec.key_env)
    creds = Credentials(
        provider=name,
        model=str(model or own.get("model") or spec.model),
        base_url=base_url or own.get("base_url") or spec.base_url,
        key_env=key_env,
        api_key=os.environ.get(key_env, "").strip(),
        notice=_legacy_notice(config, key_env),
    )
    if not (creds.model and creds.base_url) and name == "openai_compat":
        # Never default an OpenAI-compatible key to some other host.
        raise ValueError("the openai_compat provider needs a model and a base_url "
                         f"(--model / --base-url, or llm.model / llm.base_url in {_CONFIG_PATH})")
    return creds
