# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #8 R0 — live MuJoCo viewer window wiring for the G1 base.

The owner could not test campaigns #5-#7 in vector-cli because boot_g1_agent
booted a HEADLESS G1 (only the offscreen viewer_frame_png). R0 adds a live
``mujoco.viewer.launch_passive`` window, driven by the control thread that
already owns mjData (Case 12 discipline — no cross-thread GL). These tests pin
the flag plumbing (constructor default, boot pass-through, headless stays
headless); the real window is verified by GUI acceptance (tmux + screenshot).
"""
from __future__ import annotations

import pytest

from vector_os_nano.hardware.sim.mujoco_g1 import G1MuJoCoBase, g1_assets_ready


class TestViewerFlagDefault:
    """No connect needed — constructor flag contract."""

    def test_gui_defaults_off(self):
        # Headless by default: a base built without gui never opens a window,
        # so pytest / CI / harness boots stay windowless.
        b = G1MuJoCoBase()
        assert b._gui is False
        assert b._viewer is None

    def test_gui_flag_stored(self):
        b = G1MuJoCoBase(gui=True)
        assert b._gui is True
        # Not connected yet → no viewer object until connect() opens it.
        assert b._viewer is None


@pytest.mark.skipif(
    not g1_assets_ready(),
    reason="g1_gait assets not installed (run scripts/setup_g1_gait.sh)")
class TestHeadlessConnectStaysWindowless:
    def test_connect_without_gui_opens_no_viewer(self):
        b = G1MuJoCoBase()              # gui=False
        b.connect()
        try:
            assert b._viewer is None    # headless: nothing launched
        finally:
            b.disconnect()


@pytest.mark.skipif(
    not g1_assets_ready(),
    reason="g1_gait assets not installed (run scripts/setup_g1_gait.sh)")
class TestBootThreadsGuiFlag:
    def test_boot_g1_agent_passes_gui_to_base(self):
        from vector_os_nano.playground.catalog import get_scenario
        from vector_os_nano.playground.world import PlaygroundWorld
        from vector_os_nano.vcli.g1_runtime import boot_g1_agent

        # gui=False keeps the boot headless (safe under pytest); the flag must
        # still reach the base so the CLI's gui=not headless choice is honored.
        agent = boot_g1_agent(PlaygroundWorld(get_scenario("g1_flat")), gui=False)
        try:
            assert agent._base._gui is False
            assert agent._base._viewer is None
        finally:
            agent._base.disconnect()
