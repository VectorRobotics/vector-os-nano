# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Navigation tool delegates to the skill; the skill uses the common base contract."""
from __future__ import annotations

import inspect

import pytest




def _navigate_params(cls) -> dict:
    return dict(inspect.signature(cls.navigate_to).parameters)


def test_go2_and_g1_navigate_to_share_the_polymorphic_contract():
    """Both drivers accept the (x, y, timeout=...) call _NativeBaseNavigateTool makes."""
    from vector_os_nano.interfaces import BaseProtocol
    from vector_os_nano.sim.drivers.base import LeggedBase

    for cls in (LeggedBase, BaseProtocol):
        params = _navigate_params(cls)
        assert "x" in params and "y" in params, (
            f"{cls.__name__}.navigate_to must accept x, y; got {list(params)}"
        )
        accepts_timeout = "timeout" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        assert accepts_timeout, (
            f"{cls.__name__}.navigate_to must accept a `timeout` kwarg (or **kwargs) — "
            f"_NativeBaseNavigateTool calls navigate_to(x, y, timeout=...); got {list(params)}"
        )


def test_native_navigate_tool_is_embodiment_agnostic():
    """The kernel navigate tool dispatches via duck-typed base.navigate_to — no concrete-class branch."""
    import vector_os_nano.vcli.native_tools as nl

    src = inspect.getsource(nl._NativeBaseNavigateTool)
    assert "SkillWrapperTool(nav, agent).execute" in src
    assert "base.navigate_to" not in src
    # No coupling to a CONCRETE driver class inside the tool (Rule 11 at the tool layer) —
    # docstring prose may mention 'go2'/'g1' as examples, but the code must not import or
    # isinstance-check a driver class (that would be a per-embodiment fork).
    for concrete in ("LeggedBase", "MuJoCoG1", "isinstance("):
        assert concrete not in src, (
            f"_NativeBaseNavigateTool must stay embodiment-agnostic; found {concrete!r}"
        )
