"""Persistent configuration for Vector CLI.

Stores API keys, default model, provider preferences in ~/.vector/config.yaml.
Also discovers Claude Code OAuth tokens from ~/.claude/.credentials.json.

Config file location: ~/.vector/config.yaml
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_CONFIG_DIR = Path.home() / ".vector"
_CONFIG_PATH = _CONFIG_DIR / "config.yaml"
_CLAUDE_CREDS_PATH = Path.home() / ".claude" / ".credentials.json"

# Defaults when no config file exists
_DEFAULTS: dict[str, Any] = {
    "provider": "openrouter",
    "model": "claude-haiku-4-5",
    "anthropic_api_key": "",
    "openrouter_api_key": "",
    "base_url": "",
}


def load_config() -> dict[str, Any]:
    """Load config from ~/.vector/config.yaml, merging with defaults."""
    config = dict(_DEFAULTS)
    if not _CONFIG_PATH.exists():
        return config
    try:
        import yaml  # noqa: PLC0415
        raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            config.update(raw)
    except ImportError:
        # No PyYAML — fall back to simple key=value parsing
        config.update(_load_simple(_CONFIG_PATH))
    except Exception:
        pass
    return config


def save_config(config: dict[str, Any]) -> None:
    """Write config to ~/.vector/config.yaml."""
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # noqa: PLC0415
        _CONFIG_PATH.write_text(
            yaml.dump(config, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
    except ImportError:
        _save_simple(_CONFIG_PATH, config)


def _load_simple(path: Path) -> dict[str, str]:
    """Parse a simple key: value file (YAML subset)."""
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            result[k.strip()] = v.strip().strip("'\"")
    return result


def _save_simple(path: Path, config: dict[str, Any]) -> None:
    """Write a simple key: value file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}: {v}" for k, v in config.items() if v]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_claude_oauth() -> dict[str, Any] | None:
    """Load Claude Code OAuth credentials from ~/.claude/.credentials.json.

    Returns the OAuth data dict if valid and not expired, else None.
    """
    if not _CLAUDE_CREDS_PATH.exists():
        return None
    try:
        raw = json.loads(_CLAUDE_CREDS_PATH.read_text(encoding="utf-8"))
        oauth = raw.get("claudeAiOauth")
        if not isinstance(oauth, dict):
            return None
        access_token = oauth.get("accessToken", "")
        expires_at = oauth.get("expiresAt", 0)
        if not access_token:
            return None
        # Check expiry (expiresAt is ms timestamp)
        if expires_at and time.time() * 1000 > expires_at:
            return None
        return oauth
    except (json.JSONDecodeError, OSError, KeyError):
        return None


def resolve_credentials(
    cli_api_key: str | None = None,
    cli_base_url: str | None = None,
    cli_model: str | None = None,
) -> tuple[str, str, str, str | None]:
    """Resolve API key, provider, model, and base_url from all sources.

    Priority: CLI flags > Claude OAuth > env vars > config file > OpenRouter.

    Returns:
        (api_key, provider, model, base_url)
    """
    import os  # noqa: PLC0415

    config = load_config()

    api_key = cli_api_key or ""
    provider = "anthropic"
    base_url = cli_base_url

    if not api_key:
        # Claude Code OAuth (auto-discovered from ~/.claude/.credentials.json)
        oauth = load_claude_oauth()
        if oauth:
            api_key = oauth["accessToken"]
            provider = "anthropic"

    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        api_key = config.get("anthropic_api_key", "")

    if api_key:
        provider = "anthropic"
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            api_key = config.get("openrouter_api_key", "")
        if api_key:
            provider = "openrouter"
            if not base_url:
                base_url = config.get("base_url", "") or "https://openrouter.ai/api/v1"

    # Model resolution: CLI flag > config > default
    model = cli_model or config.get("model", "claude-sonnet-4-6")

    # Auto-prefix for OpenRouter, strip prefix for Anthropic direct
    if provider == "openrouter" and "/" not in model:
        model = f"anthropic/{model}"
    elif provider == "anthropic" and "/" in model:
        model = model.split("/", 1)[1]

    return api_key, provider, model, base_url or None
