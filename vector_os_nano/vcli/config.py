# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""LLM credentials: the provider is a choice, its key comes from the environment.

``provider``, ``model`` and ``base_url`` resolve as: CLI flag > the ``llm:`` block of
``$VECTOR_HOME/config.yaml`` (schemas/user.py) > the built-in default. The key is read
ONLY from the provider's environment variable (a repo-root ``.env`` is loaded first);
the config file never holds a key, and no other application's credentials are ever read.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from vector_os_nano.schemas import paths, user
from vector_os_nano.schemas.user import DEFAULT_PROVIDER, PROVIDERS


@dataclass(frozen=True)
class Credentials:
    """The resolved LLM selection. ``api_key`` is '' when its variable is unset."""

    provider: str
    model: str
    base_url: str | None
    key_env: str
    api_key: str = field(default="", repr=False)

    def instruction(self) -> str:
        """What to do when the key is missing: the exact variable to set."""
        return (f"No {self.key_env} set: the {self.provider} provider reads its key from the "
                f"environment. Add {self.key_env}=<your key> to .env in the repo root (or export "
                f"it), then type /login. Another provider: vector-cli --provider <name>, or "
                f"llm.provider in {paths.config_file()}.")


def resolve_credentials(
    provider: str | None = None, model: str | None = None, base_url: str | None = None,
) -> Credentials:
    """Resolve the LLM selection (flag > config ``llm:`` > default) and read its key.

    A repo-root ``.env`` is loaded into the environment first (an already-set
    variable wins). An unknown provider, or an invalid config file, raises with the
    valid set.
    """
    from dotenv import load_dotenv  # noqa: PLC0415

    load_dotenv()
    llm = user.load(paths.config_file()).llm
    name = provider or llm.provider or DEFAULT_PROVIDER
    if name not in PROVIDERS:
        raise ValueError(f"unknown LLM provider {name!r}; one of {sorted(PROVIDERS)}")
    spec = PROVIDERS[name]
    # The file's model / base_url / key variable belong to the file's provider: a flag
    # that picks another provider starts from that provider's defaults.
    own = llm if name == (llm.provider or DEFAULT_PROVIDER) else user.Llm()
    key_env = own.api_key_env or spec.key_env
    creds = Credentials(
        provider=name,
        model=model or own.model or spec.model,
        base_url=base_url or own.base_url or spec.base_url,
        key_env=key_env,
        api_key=os.environ.get(key_env, "").strip(),
    )
    if not (creds.model and creds.base_url) and name == "openai_compat":
        # Never default an OpenAI-compatible key to some other host.
        raise ValueError("the openai_compat provider needs a model and a base_url "
                         f"(--model / --base-url, or llm.model / llm.base_url in {paths.config_file()})")
    return creds
