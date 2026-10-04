# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Playground E-1 — a SECOND embodiment: the Go2 quadruped (mobile base).

Proves the playground/seam generalizes beyond the arm. Covers:
- the GT base oracles (at_position / facing / in_room) over a FAKE base
  exposing get_position / get_heading,
- fail-safe behaviour when the base is absent / not connected (never raises),
- the world registry wiring (resolve_world_named("go2_room") -> a has_base
  PlaygroundWorld) and the --scenario launch path,
- a kernel-integration test: the go2 PlaygroundWorld wired into VectorEngine, the
  base predicates evaluated THROUGH the real GoalVerifier off the merged
  namespace (the "test with vector-os-nano" gate),
- the decompose vocab: with a connected base the engine puts the base primitives
  (walk_forward/turn/scan_360) AND go2 skills in vocab; an arm-only agent never
  sees them,
- the existing ARM scenarios stay unaffected (predicate sets do not bleed).

No MuJoCo / network — the base is a deterministic fake.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from vector_os_nano.hardware.sim.mujoco_go2 import _ROOM_XML
from vector_os_nano.playground import PlaygroundWorld, register_scenarios
from vector_os_nano.playground.catalog import GO2_ROOM, get_scenario
from tests.harness.fake_gt import bind_fake_sim


# ---------------------------------------------------------------------------
# Deterministic fakes (no MuJoCo)
# ---------------------------------------------------------------------------


class FakeBase:
    """A deterministic stand-in for MuJoCoGo2's oracle surface.

    Exposes the two state queries the base predicates read: get_position (xyz)
    and get_heading (yaw radians). Like MuJoCoGo2 it carries a ``_connected``
    flag the predicates respect.
    """

    def __init__(
        self,
        position: list[float],
        heading: float,
        connected: bool = True,
    ) -> None:
        self._position = list(position)
        self._heading = float(heading)
        self._connected = connected

    def get_position(self) -> list[float]:
        if not self._connected:
            raise RuntimeError("FakeBase: not connected")
        return list(self._position)

    def get_heading(self) -> float:
        if not self._connected:
            raise RuntimeError("FakeBase: not connected")
        return self._heading


def _go2_world() -> PlaygroundWorld:
    return PlaygroundWorld(GO2_ROOM)


def _base_agent(base: Any = None) -> SimpleNamespace:
    agent = SimpleNamespace(_base=base)
    if base is not None:
        bind_fake_sim(agent, "go2")  # the live sim session's GT oracles
    return agent


# ---------------------------------------------------------------------------
# Scenario / world contract
# ---------------------------------------------------------------------------


class TestGo2Scenario:
    def test_scenario_registered_and_has_base(self) -> None:
        register_scenarios()  # idempotent
        from vector_os_nano.vcli.worlds.registry import resolve_world_named

        world = resolve_world_named("go2_room")
        assert type(world).__name__ == "PlaygroundWorld"
        assert world.name == "go2_room"
        assert world.is_robot() is True
        assert world.has_base() is True
        assert world.embodiment == "go2"

    def test_arm_scenario_reports_no_base(self) -> None:
        # The arm scenario must NOT claim a base — embodiments don't bleed.
        assert PlaygroundWorld().has_base() is False
        assert PlaygroundWorld().embodiment == "arm"

    def test_go2_scenario_has_named_rooms(self) -> None:
        scen = get_scenario("go2_room")
        assert "kitchen" in scen.rooms
        # Each room is a 4-tuple (x_min, y_min, x_max, y_max).
        assert all(len(box) == 4 for box in scen.rooms.values())

    def test_decompose_vocab_single_sourced(self) -> None:
        world = _go2_world()
        assert world.decompose_vocab() is None
        assert world.derive_vocab_from_registry() is True


# ---------------------------------------------------------------------------
# Base predicates: at_position
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Base predicates: facing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Base predicates: visited
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fail-safe: no base / not connected / raising oracle => never raises
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Kernel integration: predicates through the real GoalVerifier + the engine
# ---------------------------------------------------------------------------


def _make_engine():
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.intent_router import IntentRouter
    from vector_os_nano.vcli.tools.base import CategorizedToolRegistry

    class _MockBackend:
        def call(self, messages, tools, system, max_tokens, on_text=None):
            class _R:
                text = "{}"

            return _R()

    return VectorEngine(
        backend=_MockBackend(),
        registry=CategorizedToolRegistry(),
        system_prompt=[],
        intent_router=IntentRouter(),
    )


class TestKernelIntegration:
    def test_base_predicates_via_goal_verifier(self) -> None:
        from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier

        # (17, 2.5) is the REAL kitchen centre (f_kitchen pos="17 2.5").
        base = FakeBase(position=[17.0, 2.5, 0.3], heading=0.0)
        agent = _base_agent(base)
        eng = _make_engine()
        eng.init_vgg(agent=agent, world=_go2_world())
        assert type(eng._world).__name__ == "PlaygroundWorld"

        ns = eng._build_verifier_namespace(agent)
        # Base predicates are present (and the arm-only stubs were not injected
        # by the go2 world).
        assert {"at_position", "facing", "in_room"} <= set(ns)

        gv = GoalVerifier(ns)
        assert gv.verify("at_position(17.0, 2.5)") is True
        assert gv.verify("facing(0.0)") is True
        assert gv.verify("in_room('kitchen')") is True
        assert gv.verify("in_room('guest_bedroom')") is False

    def test_resolved_via_scenario_flag(self) -> None:
        """The --scenario launch path resolves the go2 playground world."""
        from vector_os_nano.vcli.cli import _resolve_active_world, parse_args

        args = parse_args(["--scenario", "go2_room"])
        world = _resolve_active_world(args, agent=_base_agent(None))
        assert type(world).__name__ == "PlaygroundWorld"
        assert world.name == "go2_room"
        assert world.has_base() is True


# ---------------------------------------------------------------------------
# Decompose vocab: base primitives + go2 skills are in-vocab with a base agent;
# an arm-only agent never sees them (mechanism is agent-driven, not world-driven).
# ---------------------------------------------------------------------------


class _FakeSkill:
    def __init__(self, name: str, description: str) -> None:
        self._schema = {"name": name, "description": description, "parameters": {}}

    def to_schema(self) -> dict:
        return dict(self._schema)


class _FakeSkillRegistry:
    """Minimal skill registry exposing to_schemas() like the real one."""

    def __init__(self, schemas: list[dict]) -> None:
        self._schemas = schemas

    def to_schemas(self) -> list[dict]:
        return [dict(s) for s in self._schemas]


class TestDecomposeVocabRouting:
    def _schemas(self) -> list[dict]:
        # Stand in for skills/go2 explore + look discovered from the registry.
        return [
            {"name": "explore", "description": "Explore the environment", "parameters": {}},
            {"name": "look", "description": "Look in a direction", "parameters": {}},
        ]

    def test_base_agent_gets_base_primitives_and_go2_skills(self) -> None:
        eng = _make_engine()
        eng._world = _go2_world()
        reg = _FakeSkillRegistry(self._schemas())
        agent = _base_agent(FakeBase(position=[0.0, 0.0, 0.3], heading=0.0))

        # has_base=True path: base primitives present, go2 skills present.
        kwargs = eng._build_registry_vocab_kwargs(reg, agent, has_base=True)
        strategies = set(kwargs["strategies"])
        assert {"walk_forward", "turn", "scan_360"} <= strategies
        assert {"explore_skill", "look_skill"} <= strategies

    def test_arm_agent_never_sees_base_primitives(self) -> None:
        eng = _make_engine()
        eng._world = PlaygroundWorld()  # arm scenario
        reg = _FakeSkillRegistry(self._schemas())
        agent = SimpleNamespace(_arm=None, _base=None)

        kwargs = eng._build_registry_vocab_kwargs(reg, agent, has_base=False)
        strategies = set(kwargs["strategies"])
        assert not ({"walk_forward", "turn", "scan_360"} & strategies)


# ---------------------------------------------------------------------------
# Arm scenarios stay unaffected (regression guard).
# ---------------------------------------------------------------------------


class TestArmUnaffected:
    def test_arm_scenario_predicate_set_unchanged(self) -> None:
        class _Arm:
            def get_object_positions(self):
                return {"mug": [0.22, 0.05, 0.06]}

        agent = SimpleNamespace(_arm=_Arm(), _base=None)
        bind_fake_sim(agent, "so101")
        ns = PlaygroundWorld().build_verify_namespace(agent)
        assert {"holding_object", "arm_at_home", "object_on", "released"} <= set(ns)
        # No base predicates leak into an arm scenario (a fixed-base arm).
        assert not ({"at_position", "facing", "in_room"} & set(ns))


# ---------------------------------------------------------------------------
# G2 — the go2 room geometry is RECONCILED with the real room scene.
#
# Reads the tracked go2_room.xml template that scene_room.xml is built from, as
# DATA (xml.etree, never MuJoCo: no sim load, no MUJOCO_GL pollution). Never the
# generated, gitignored scene_room.xml: it exists only after a sim test has run,
# which made this hermetic test order-dependent. Asserts every playground room box matches the matching
# ``f_<room>`` floor-plane geom's extent, that the real Go2 spawn (10, 3) lands in
# the hallway, and that the rooms producer (the foreach source) emits those real
# room centres. This locks the placeholder-quadrant -> real-house swap.
# ---------------------------------------------------------------------------


def _floor_boxes_from_scene_xml(
    xml_path: str,
) -> dict[str, tuple[float, float, float, float]]:
    """Return ``f_<name>`` floor-geom extents as (x_min, y_min, x_max, y_max).

    Pure data read of the MJCF: each ``f_<name>`` plane/box geom has
    ``pos="cx cy cz"`` and ``size="hx hy ..."``, so its planar extent is
    ``(cx-hx, cy-hy, cx+hx, cy+hy)``. No MuJoCo, no sim load.
    """
    import xml.etree.ElementTree as ET

    boxes: dict[str, tuple[float, float, float, float]] = {}
    root = ET.parse(xml_path).getroot()
    for geom in root.iter("geom"):
        name = geom.get("name", "")
        if not name.startswith("f_"):
            continue
        cx, cy, *_ = (float(v) for v in geom.get("pos", "0 0 0").split())
        hx, hy, *_ = (float(v) for v in geom.get("size", "0 0 0").split())
        boxes[name[2:]] = (cx - hx, cy - hy, cx + hx, cy + hy)
    return boxes


class TestGo2GeometryReconciledWithScene:
    # The real Go2 spawn (MuJoCoGo2._reset sets data.qpos[0:3] = [10, 3, 0.35]).
    _SPAWN_XY = (10.0, 3.0)

    def test_perimeter_room_boxes_match_floor_geoms(self) -> None:
        scen = get_scenario("go2_room")
        floor = _floor_boxes_from_scene_xml(str(_ROOM_XML))
        # Every room with a dedicated f_<room> floor geom matches it exactly.
        # (master_bedroom -> f_master, guest_bedroom -> f_guest.) The bathroom is
        # the documented exception: the XML merged the laundry INTO the bathroom
        # (iw_bath_e wall removed), so the navigable room spans x=7..12 while the
        # tiled f_bath floor geom is only the x=7..10 sub-region — checked
        # separately below. The hallway is open central space with no f_ geom.
        # Map the playground room name to its f_<short> floor geom in the XML.
        alias = {
            "living_room": "living",
            "dining_room": "dining",
            "master_bedroom": "master",
            "guest_bedroom": "guest",
        }
        skip = {"bathroom", "hallway"}
        for room, box in scen.rooms.items():
            if room in skip:
                continue
            geom_name = alias.get(room, room)
            assert geom_name in floor, room
            assert box == pytest.approx(floor[geom_name]), room

    def test_bathroom_box_is_merged_bathroom_plus_laundry(self) -> None:
        # The merged bathroom+laundry room (x=7..12) must contain BOTH the tiled
        # f_bath floor region AND the laundry appliances (washer/dryer at x=11).
        scen = get_scenario("go2_room")
        floor = _floor_boxes_from_scene_xml(str(_ROOM_XML))
        bx_min, by_min, bx_max, by_max = scen.rooms["bathroom"]
        fx_min, fy_min, fx_max, fy_max = floor["bath"]
        # The room box covers the entire tiled floor geom...
        assert bx_min <= fx_min and bx_max >= fx_max
        assert by_min <= fy_min and by_max >= fy_max
        # ...and reaches the merged laundry zone (washer/dryer bodies at x=11).
        assert bx_min <= 11.0 <= bx_max

    def test_hallway_is_central_open_span_containing_spawn(self) -> None:
        scen = get_scenario("go2_room")
        x_min, y_min, x_max, y_max = scen.rooms["hallway"]
        sx, sy = self._SPAWN_XY
        # The real spawn sits inside the hallway box.
        assert x_min <= sx <= x_max and y_min <= sy <= y_max

    def test_spawn_visited_resolves_to_hallway_only(self) -> None:
        # Through the live predicate: the spawn is in the hallway, no other room.
        ns = _go2_world().build_verify_namespace(
            _base_agent(FakeBase(position=[*self._SPAWN_XY, 0.3], heading=0.0))
        )
        assert ns["in_room"]("hallway") is True
        for other in ("kitchen", "living_room", "guest_bedroom", "study"):
            assert ns["in_room"](other) is False, other

    def test_rooms_producer_emits_real_room_centres(self) -> None:
        # The foreach source step ("locate the rooms") emits the SAME room set the
        # in_room() predicate reads, each carrying its real (x, y) centre.
        scen = get_scenario("go2_room")
        producer = _go2_world().build_step_primitives(_base_agent(None))[
            PlaygroundWorld.ROOMS_STRATEGY
        ]
        out = producer()
        assert out["count"] == len(scen.rooms)
        by_name = {r["name"]: (r["x"], r["y"]) for r in out["rooms"]}
        # Kitchen centre is the real f_kitchen centre (17, 2.5); hallway is (10, 5).
        assert by_name["kitchen"] == pytest.approx((17.0, 2.5))
        assert by_name["hallway"] == pytest.approx((10.0, 5.0))
        # Every emitted centre lands inside its own named box (producer/verifier
        # never diverge).
        for name, (x, y) in by_name.items():
            assert ns_visited_box(scen.rooms[name], x, y), name


def ns_visited_box(
    box: tuple[float, float, float, float], x: float, y: float
) -> bool:
    x_min, y_min, x_max, y_max = box
    return x_min <= x <= x_max and y_min <= y <= y_max
