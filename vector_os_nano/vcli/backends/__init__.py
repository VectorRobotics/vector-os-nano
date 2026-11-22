# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Construct provider adapters for the public ``interfaces.LLMBackend`` contract."""
from __future__ import annotations

from vector_os_nano.interfaces.llm import LLMBackend

__all__ = ["create_backend"]


def create_backend(
    provider: str,
    api_key: str,
    model: str,
    base_url: str | None = None,
) -> LLMBackend:
    """Factory: create the right backend for a given provider.

    Args:
        provider: "anthropic" or "openrouter" or "openai_compat".
        api_key:  API key for the provider.
        model:    Model identifier (provider-specific naming).
        base_url: Optional base URL override.

    Returns:
        An LLMBackend instance ready to call().
    """
    if provider == "anthropic":
        from vector_os_nano.vcli.backends.anthropic import AnthropicBackend

        return AnthropicBackend(api_key=api_key, model=model, base_url=base_url)

    # openrouter, openai_compat, local — all use OpenAI-compatible API
    from vector_os_nano.vcli.backends.openai_compat import OpenAICompatBackend

    return OpenAICompatBackend(
        api_key=api_key,
        model=model,
        base_url=base_url or "https://openrouter.ai/api/v1",
    )
