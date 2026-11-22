# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""``$VECTOR_HOME/config.yaml`` (charter section 4): the user's settings, strictly parsed.

    schema: 1
    llm:
      provider: deepseek        # deepseek | openrouter | anthropic | openai_compat
      model: deepseek-v4-flash  # optional: the provider's default otherwise
      base_url: ...             # optional
      api_key_env: MY_KEY       # optional: read the key from this variable instead

Precedence: CLI flag > this file > the built-in default (vcli/config.py). The file never
holds a key: a key that looks like a secret is refused, naming the variable to set.
The ``defaults`` block selects simulation startup; ``perception`` selects its models.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from vector_os_nano.schemas.strict import yaml_loader

SCHEMA = 1


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

DETECTORS = {
    'grounding-dino': 'vector_os_nano.perception.grounding_dino:get_shared_detector',
    'rynnbrain': 'vector_os_nano.perception.rynnbrain:RynnBrainDetector',
}
SEGMENTERS = {'edgetam': 'vector_os_nano.perception.tracker:EdgeTAMTracker'}


@dataclass(frozen=True)
class Llm:
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None


@dataclass(frozen=True)
class Defaults:
    embodiment: str = "so101"
    backend: str = "mujoco"
    world: str | None = None
    gui: bool = True


@dataclass(frozen=True)
class SceneVlm:
    model: str = 'google/gemma-4-31b-it'
    base_url: str = 'https://openrouter.ai/api/v1'
    api_key_env: str | None = 'OPENROUTER_API_KEY'


@dataclass(frozen=True)
class Perception:
    detector: str = 'grounding-dino'
    segmenter: str = 'edgetam'
    scene_vlm: SceneVlm | None = SceneVlm()


@dataclass(frozen=True)
class UserConfig:
    llm: Llm = Llm()
    defaults: Defaults = Defaults()
    perception: Perception = Perception()


class UserConfigError(ValueError):
    """config.yaml is not a valid schema-1 file (the message names the file and the key)."""


# A key name that says it holds a secret (api_key_env names a variable, not a secret).
_SECRET_WORD = r"api_?key|key|token|secret|password|passwd|credentials?"
_SECRET = re.compile(rf"(^|_)({_SECRET_WORD})$", re.I)


def _secret_error(path: Path, raw: dict[str, Any]) -> UserConfigError | None:
    """The first secret-looking key, with the variable its value belongs in (never the value)."""
    llm = raw.get("llm") if isinstance(raw.get("llm"), dict) else {}
    provider = PROVIDERS.get(str(llm.get("provider")), PROVIDERS[DEFAULT_PROVIDER])
    own = str(llm.get("api_key_env") or provider.key_env)
    perception = raw.get('perception')
    perception = perception if isinstance(perception, dict) else {}
    scene = perception.get('scene_vlm')
    scene = scene if isinstance(scene, dict) else {}
    blocks = [('', raw, own), ('llm.', llm, own), ('perception.', perception, 'OPENROUTER_API_KEY'),
              ('perception.scene_vlm.', scene, scene.get('api_key_env') or 'OPENROUTER_API_KEY')]
    for prefix, block, own in blocks:
        for key in block:
            if _SECRET.search(str(key)):
                var = own if re.fullmatch(_SECRET_WORD, str(key), re.I) else str(key).upper()
                return UserConfigError(f"{path}: {prefix}{key} looks like a secret, and the config file never holds "
                                       f"one: delete it and set {var} in the environment (or .env) instead")
    return None


def _unknown(path: Path, where: str, keys: Any, valid: list[str]) -> None:
    extra = sorted(str(k) for k in keys if k not in valid)
    if extra:
        raise UserConfigError(f"{path}: unknown key {where}{extra[0]}; valid: {', '.join(valid)}")


def load(path: Path) -> UserConfig:
    """Parse *path*; an absent file is the empty config. Raises UserConfigError."""
    if not path.exists():
        return UserConfig()
    import yaml

    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml_loader(path, UserConfigError))  # noqa: S506 - a SafeLoader
    except (OSError, yaml.YAMLError) as exc:
        raise UserConfigError(f"{path}: unreadable ({type(exc).__name__}: {exc})") from exc
    if not isinstance(raw, dict):
        raise UserConfigError(f"{path}: a mapping with schema: {SCHEMA} is required")
    secret = _secret_error(path, raw)
    if secret is not None:
        raise secret
    schema = raw.get("schema")
    if type(schema) is not int:
        raise UserConfigError(f"{path}: schema: {SCHEMA} is required (got {schema!r})")
    if schema > SCHEMA:
        raise UserConfigError(f"{path}: schema {schema} is newer than this release reads ({SCHEMA}): "
                              "upgrade vector_os_nano")
    if schema < SCHEMA:
        raise UserConfigError(f"{path}: schema {schema} is not a valid schema number; use schema: {SCHEMA}")
    _unknown(path, "", raw, ["schema", "llm", "defaults", "perception"])
    llm = raw.get("llm") or {}
    if not isinstance(llm, dict):
        raise UserConfigError(f"{path}: llm must be a mapping (got {type(llm).__name__})")
    names = [f.name for f in fields(Llm)]
    _unknown(path, "llm.", llm, names)
    for key, value in llm.items():
        if value is not None and not isinstance(value, str):
            raise UserConfigError(f"{path}: llm.{key} must be a string (got {type(value).__name__})")
    if llm.get("provider") is not None and llm["provider"] not in PROVIDERS:
        raise UserConfigError(f"{path}: llm.provider {llm['provider']!r} is not one of {', '.join(PROVIDERS)}")
    defaults = raw.get("defaults", {})
    if not isinstance(defaults, dict):
        raise UserConfigError(f"{path}: defaults must be a mapping")
    _unknown(path, "defaults.", defaults, [f.name for f in fields(Defaults)])
    for key, value in defaults.items():
        if key == "gui":
            if type(value) is not bool:
                raise UserConfigError(f"{path}: defaults.gui must be true or false")
        elif not isinstance(value, str) or not value.strip():
            raise UserConfigError(f"{path}: defaults.{key} must be a non-empty string")
    perception = raw.get('perception', {})
    if not isinstance(perception, dict):
        raise UserConfigError(f'{path}: perception must be a mapping')
    _unknown(path, 'perception.', perception, [f.name for f in fields(Perception)])
    for key, table in (('detector', DETECTORS), ('segmenter', SEGMENTERS)):
        if key in perception and (not isinstance(perception[key], str) or perception[key] not in table):
            raise UserConfigError(f'{path}: perception.{key} must name a model; valid: {", ".join(table)}')
    scene = perception.get('scene_vlm', {})
    if scene is not None:
        if not isinstance(scene, dict):
            raise UserConfigError(f'{path}: perception.scene_vlm must be a mapping or null')
        _unknown(path, 'perception.scene_vlm.', scene, [f.name for f in fields(SceneVlm)])
        for key, value in scene.items():
            if key == 'api_key_env' and value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise UserConfigError(f'{path}: perception.scene_vlm.{key} must be a non-empty string')
        scene = SceneVlm(**scene)
        try:
            url = urlsplit(scene.base_url)
        except ValueError as exc:
            raise UserConfigError(f'{path}: perception.scene_vlm.base_url is not a valid URL') from exc
        if url.scheme not in ('http', 'https') or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise UserConfigError(f'{path}: perception.scene_vlm.base_url must be an HTTP(S) URL without credentials, query or fragment')
        if scene.api_key_env is not None and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', scene.api_key_env):
            raise UserConfigError(f'{path}: perception.scene_vlm.api_key_env must name an environment variable')
    return UserConfig(llm=Llm(**llm), defaults=Defaults(**defaults),
                      perception=Perception(**{**perception, 'scene_vlm': scene}))
