# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""S3c — the navigate capability is CONVERGED at the tool layer (the invariant guard).

`_NativeBaseNavigateTool` (native_loop.py) dispatches coordinate navigation POLYMORPHICALLY:
it calls ``base.navigate_to(x, y, timeout=...)`` on whatever base is connected — there is
NO go2-vs-g1 branch in the kernel/tool layer (Rule 11 at the tool layer is already met).
Both in-process drivers implement it (``MuJoCoGo2.navigate_to`` / ``MuJoCoG1.navigate_to``).

This test PINS the polymorphic contract both must honor so the one navigate tool keeps
dispatching to either without a per-embodiment branch. A future change that diverges one
driver's call shape fails here.
"""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip("mujoco", reason="the drivers need mujoco")


def _navigate_params(cls) -> dict:
    return dict(inspect.signature(cls.navigate_to).parameters)


def test_go2_and_g1_navigate_to_share_the_polymorphic_contract():
    """Both drivers accept the (x, y, timeout=...) call _NativeBaseNavigateTool makes."""
    from vector_os_nano.hardware.sim.mujoco_g1 import MuJoCoG1
    from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2

    for cls in (MuJoCoGo2, MuJoCoG1):
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
    assert "navigate_to" in src, "the tool must call the base's navigate_to"
    # No coupling to a CONCRETE driver class inside the tool (Rule 11 at the tool layer) —
    # docstring prose may mention 'go2'/'g1' as examples, but the code must not import or
    # isinstance-check a driver class (that would be a per-embodiment fork).
    for concrete in ("MuJoCoGo2", "MuJoCoG1", "isinstance("):
        assert concrete not in src, (
            f"_NativeBaseNavigateTool must stay embodiment-agnostic; found {concrete!r}"
        )
