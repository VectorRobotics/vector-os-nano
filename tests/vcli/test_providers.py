# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""W3.3 — Protocol-based provider resolution (kill getattr-by-string).

The kernel asks for a PROTOCOL, not a named private attribute: a baseless
arm world requesting a base capability gets a clear "no base provider met
spec" error (naming the spec and what IS present) instead of an
AttributeError or a silent None. `_base` seam first, per the plan.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vector_os_nano.vcli.providers import (
    BaseMotionProvider,
    BaseStateProvider,
    ProviderError,
    ensure_provider,
    resolve_provider,
)


def _full_base() -> SimpleNamespace:
    return SimpleNamespace(
        get_position=lambda: [0.0, 0.0, 0.0],
        get_heading=lambda: 0.0,
        walk=lambda vx=0.0, vy=0.0, vyaw=0.0, duration=1.0: True,
        set_velocity=lambda vx, vy, vyaw: None,
        stop=lambda: None,
    )


class TestProtocols:
    def test_full_base_satisfies_both(self) -> None:
        b = _full_base()
        assert isinstance(b, BaseStateProvider)
        assert isinstance(b, BaseMotionProvider)

    def test_query_only_base_satisfies_state_not_motion(self) -> None:
        b = SimpleNamespace(get_position=lambda: [0, 0, 0], get_heading=lambda: 0.0)
        assert isinstance(b, BaseStateProvider)
        assert not isinstance(b, BaseMotionProvider)


class TestResolveProvider:
    def test_resolves_conforming_base(self) -> None:
        agent = SimpleNamespace(_base=_full_base())
        assert resolve_provider(agent, BaseMotionProvider, what="base") is agent._base

    def test_baseless_agent_fails_loud_with_spec_name(self) -> None:
        agent = SimpleNamespace(_arm=object())
        with pytest.raises(ProviderError) as ei:
            resolve_provider(agent, BaseMotionProvider, what="base")
        msg = str(ei.value)
        assert "no base provider" in msg
        assert "BaseMotionProvider" in msg

    def test_optional_resolution_returns_none(self) -> None:
        agent = SimpleNamespace()
        assert (
            resolve_provider(agent, BaseStateProvider, what="base", required=False)
            is None
        )

    def test_nonconforming_candidate_error_names_missing_methods(self) -> None:
        agent = SimpleNamespace(_base=SimpleNamespace(get_position=lambda: [0, 0, 0]))
        with pytest.raises(ProviderError) as ei:
            resolve_provider(agent, BaseMotionProvider, what="base")
        msg = str(ei.value)
        assert "get_heading" in msg  # tells the integrator exactly what is missing
        assert "walk" in msg


class TestEnsureProvider:
    def test_none_raises_clear_error(self) -> None:
        with pytest.raises(ProviderError) as ei:
            ensure_provider(None, BaseMotionProvider, what="base")
        assert "no base provider" in str(ei.value)
        assert "BaseMotionProvider" in str(ei.value)

    def test_conforming_passes_through(self) -> None:
        b = _full_base()
        assert ensure_provider(b, BaseMotionProvider, what="base") is b

    def test_provider_error_is_a_runtime_error(self) -> None:
        """Compatibility: existing except-RuntimeError handlers keep working."""
        assert issubclass(ProviderError, RuntimeError)


class TestLocomotionSeamMigrated:
    def test_baseless_primitives_fail_with_spec_error(self) -> None:
        from vector_os_nano.vcli import primitives
        from vector_os_nano.vcli.primitives import locomotion

        old = locomotion._ctx
        locomotion._ctx = primitives.PrimitiveContext(base=None)
        try:
            with pytest.raises(ProviderError) as ei:
                locomotion.get_position()
            assert "no base provider" in str(ei.value)
        finally:
            locomotion._ctx = old
