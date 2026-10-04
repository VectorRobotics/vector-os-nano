# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Playground (INC9-polish) — v1 polish, all headless & deterministic.

Covers the four small, additive polish items:

1. Mid-session / conversational scenario selection: ``cli.enter_scenario`` routes
   entering a scenario through ``resolve_world_named`` so the playground is
   reachable WITHOUT relaunching with ``--scenario``. Fails loud on unknown id;
   swaps the world into app_state and re-inits VGG via the stored callbacks.
2. CLI startup banner surfaces the ACTIVE scenario name when one is selected.
3. A SECOND catalog scenario (``tabletop_tray``) so ``resolve_world_named`` is
   exercised over more than one preset and ``--scenario tabletop_tray`` resolves.
4. Scene-defined place region: the ``tabletop_tray`` scenario carries the
   tabletop world manifest's ``tray`` drop-zone (``object_on(obj, 'tray')``).

No MuJoCo / network: the arm is a deterministic fake; the engine is a stub that
records the init_vgg world. No live LLM.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vector_os_nano.playground import PlaygroundWorld
from vector_os_nano.playground.catalog import (
    SCENARIOS,
    TABLETOP,
    TABLETOP_TRAY,
    get_scenario,
)
from tests.harness.fake_gt import bind_fake_sim
from vector_os_nano.embodiments import registry

# The SO-101 home pose (its manifest stance; the arm_at_home oracle reads it there).
_HOME_JOINTS = tuple(registry.load("so101").stance.values())
from vector_os_nano.vcli import cli
from vector_os_nano.vcli.worlds import resolve_world_named


_HOME = list(_HOME_JOINTS)


# ---------------------------------------------------------------------------
# Deterministic fakes
# ---------------------------------------------------------------------------


class FakeArm:
    def __init__(self, objects: dict[str, list[float]]) -> None:
        self._objects = objects
        self._connected = True

    def get_object_positions(self) -> dict[str, list[float]]:
        return {k: list(v) for k, v in self._objects.items()}

    def get_joint_positions(self) -> list[float]:
        return list(_HOME)

    def fk(self, joint_positions: list[float]):
        return [0.2, 0.0, 0.2], [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]


def _agent(arm: Any = None) -> SimpleNamespace:
    return SimpleNamespace(_arm=arm, _gripper=None)


class _StubEngine:
    """Records the world flowed into init_vgg (and the view-sink callback)."""

    def __init__(self) -> None:
        self.init_calls: list[dict[str, Any]] = []

    def init_vgg(self, **kwargs: Any) -> None:
        self.init_calls.append(kwargs)


# ---------------------------------------------------------------------------
# 3. Second scenario in the catalog
# ---------------------------------------------------------------------------


class TestSecondScenario:
    def test_tray_scenario_registered_distinct_from_tabletop(self) -> None:
        assert "tabletop" in SCENARIOS
        assert "tabletop_tray" in SCENARIOS
        assert SCENARIOS["tabletop_tray"] is TABLETOP_TRAY
        assert TABLETOP_TRAY.id != TABLETOP.id

    def test_get_scenario_resolves_both(self) -> None:
        assert get_scenario("tabletop").id == "tabletop"
        assert get_scenario("tabletop_tray").id == "tabletop_tray"

    def test_resolve_world_named_resolves_tray(self) -> None:
        # Importing the package (above) registered the scenarios; the kernel
        # registry resolves the new preset by id without a relaunch.
        world = resolve_world_named("tabletop_tray")
        assert isinstance(world, PlaygroundWorld)
        assert world.name == "tabletop_tray"
        assert world.scenario.id == "tabletop_tray"

    def test_both_presets_share_the_so101_scene(self) -> None:
        assert TABLETOP_TRAY.scene_xml == TABLETOP.scene_xml
        assert TABLETOP_TRAY.scene_xml.endswith("so101_mujoco.xml")


# ---------------------------------------------------------------------------
# 4. Scene-defined place region
# ---------------------------------------------------------------------------


class TestSceneDefinedPlaceRegion:
    def test_tabletop_has_no_region_default(self) -> None:
        # The bare tabletop scenario names no drop-zone.
        assert TABLETOP.place_region is None

    def test_tray_scenario_carries_a_region(self) -> None:
        assert TABLETOP_TRAY.place_region == (0.20, 0.0, 0.50, 0.25)

# ---------------------------------------------------------------------------
# 2. Banner surfaces the active scenario name
# ---------------------------------------------------------------------------


class TestBannerScenario:
    def test_format_banner_includes_scenario(self) -> None:
        text = cli.format_banner("gpt", None, scenario="tabletop_tray")
        assert "Scenario: tabletop_tray" in text

    def test_format_banner_omits_scenario_when_none(self) -> None:
        text = cli.format_banner("gpt", None)
        assert "Scenario:" not in text

    def test_print_banner_renders_scenario(self, capsys: Any) -> None:
        cli.print_banner("gpt", "Anthropic", None, scenario="tabletop")
        out = capsys.readouterr().out
        assert "Scenario: tabletop" in out


# ---------------------------------------------------------------------------
# 1. Mid-session / conversational scenario selection
# ---------------------------------------------------------------------------


class TestEnterScenario:
    def test_enters_named_scenario_and_updates_state(self) -> None:
        engine = _StubEngine()
        app_state: dict[str, Any] = {
            "engine": engine,
            "agent": None,
            "skill_registry": None,
            "vgg_step_callback": (lambda s: None),
            "vgg_step_view_callback": (lambda v: None),
            "scenario": None,
            "world": None,
        }
        world = cli.enter_scenario("tabletop_tray", app_state)

        assert isinstance(world, PlaygroundWorld)
        assert world.name == "tabletop_tray"
        # State swapped in.
        assert app_state["world"] is world
        assert app_state["scenario"] == "tabletop_tray"
        # VGG re-inited with the new world + the stored view sink.
        assert len(engine.init_calls) == 1
        call = engine.init_calls[0]
        assert call["world"] is world
        assert call["on_vgg_step_view"] is app_state["vgg_step_view_callback"]

    def test_unknown_scenario_fails_loud_with_valid_set(self) -> None:
        app_state: dict[str, Any] = {"engine": None, "scenario": None, "world": None}
        with pytest.raises(KeyError) as exc:
            cli.enter_scenario("does-not-exist", app_state)
        msg = str(exc.value)
        assert "does-not-exist" in msg
        # Valid set surfaced; both presets present, never a silent fallback.
        assert "tabletop" in msg
        # State NOT mutated on failure.
        assert app_state["scenario"] is None
        assert app_state["world"] is None

    def test_no_engine_still_swaps_world(self) -> None:
        # Before an API key is set there's no engine; the world swap must still
        # stand (it takes effect at the next engine init) and must not crash.
        app_state: dict[str, Any] = {"engine": None, "scenario": None, "world": None}
        world = cli.enter_scenario("tabletop", app_state)
        assert app_state["world"] is world
        assert app_state["scenario"] == "tabletop"

    def test_slash_scenario_switches_live(self) -> None:
        engine = _StubEngine()
        app_state: dict[str, Any] = {
            "engine": engine,
            "agent": None,
            "skill_registry": None,
            "vgg_step_callback": (lambda s: None),
            "vgg_step_view_callback": (lambda v: None),
            "scenario": None,
            "world": None,
        }
        cont = cli._handle_slash_command(
            "scenario", ["tabletop_tray"], registry=_DummyRegistry(), app_state=app_state
        )
        assert cont is True
        assert app_state["scenario"] == "tabletop_tray"
        assert isinstance(app_state["world"], PlaygroundWorld)

    def test_slash_scenario_unknown_id_does_not_crash(self) -> None:
        app_state: dict[str, Any] = {"engine": None, "scenario": None, "world": None}
        cont = cli._handle_slash_command(
            "scenario", ["nope"], registry=_DummyRegistry(), app_state=app_state
        )
        # Reported (fail loud) but the REPL keeps running; state untouched.
        assert cont is True
        assert app_state["scenario"] is None
        assert app_state["world"] is None

    def test_slash_scenario_no_arg_shows_active(self) -> None:
        app_state: dict[str, Any] = {
            "engine": None,
            "scenario": "tabletop_tray",
            "world": None,
        }
        cont = cli._handle_slash_command(
            "scenario", [], registry=_DummyRegistry(), app_state=app_state
        )
        assert cont is True
        # Read-only: showing the active scenario must not change state.
        assert app_state["scenario"] == "tabletop_tray"


class _DummyRegistry:
    """Minimal registry stand-in (the /scenario branch never touches it)."""

    def list_tools(self) -> list[str]:
        return []
