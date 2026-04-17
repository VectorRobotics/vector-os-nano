"""Tests for Gazebo Harmonic world files: empty_room.sdf and apartment.sdf.

Validates that world SDF files exist, are valid XML, and contain all required
structural elements (plugins, lights, ground plane, walls, furniture, objects).

Level: Unit — pure file-parsing, no Gazebo runtime required.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
_WORLD_FILE = _REPO_ROOT / "gazebo" / "worlds" / "empty_room.sdf"
_APARTMENT_FILE = _REPO_ROOT / "gazebo" / "worlds" / "apartment.sdf"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sdf_tree() -> ET.ElementTree:
    """Parse empty_room.sdf once for the whole module."""
    return ET.parse(str(_WORLD_FILE))


@pytest.fixture(scope="module")
def sdf_root(sdf_tree: ET.ElementTree) -> ET.Element:
    """Return the root <sdf> element."""
    return sdf_tree.getroot()


@pytest.fixture(scope="module")
def world_elem(sdf_root: ET.Element) -> ET.Element:
    """Return the <world> element."""
    elem = sdf_root.find("world")
    assert elem is not None, "<world> element not found in SDF"
    return elem


# ---------------------------------------------------------------------------
# Existence and parse tests
# ---------------------------------------------------------------------------


class TestEmptyRoom:
    def test_empty_room_sdf_exists(self) -> None:
        """gazebo/worlds/empty_room.sdf must exist."""
        assert _WORLD_FILE.is_file(), f"File not found: {_WORLD_FILE}"

    def test_empty_room_sdf_valid_xml(self) -> None:
        """empty_room.sdf must be parseable as valid XML."""
        tree = ET.parse(str(_WORLD_FILE))
        assert tree.getroot() is not None

    # -----------------------------------------------------------------------
    # Structure
    # -----------------------------------------------------------------------

    def test_empty_room_has_world_element(self, sdf_root: ET.Element) -> None:
        """Root <sdf> element must contain a <world> child."""
        world = sdf_root.find("world")
        assert world is not None, "<world> element not found under <sdf>"
        assert world.get("name") == "empty_room", (
            f"Expected world name='empty_room', got '{world.get('name')}'"
        )

    # -----------------------------------------------------------------------
    # Plugins
    # -----------------------------------------------------------------------

    def test_empty_room_has_physics_plugin(self, sdf_root: ET.Element) -> None:
        """World must declare the Physics plugin."""
        plugin_names = {
            p.get("name", "") for p in sdf_root.findall(".//plugin")
        }
        assert "gz::sim::systems::Physics" in plugin_names, (
            f"gz::sim::systems::Physics not found. Plugins: {plugin_names}"
        )

    def test_empty_room_has_sensors_plugin(self, sdf_root: ET.Element) -> None:
        """World must declare the Sensors plugin."""
        plugin_names = {
            p.get("name", "") for p in sdf_root.findall(".//plugin")
        }
        assert "gz::sim::systems::Sensors" in plugin_names, (
            f"gz::sim::systems::Sensors not found. Plugins: {plugin_names}"
        )

    # -----------------------------------------------------------------------
    # Models
    # -----------------------------------------------------------------------

    def test_empty_room_has_ground_plane(self, sdf_root: ET.Element) -> None:
        """World must contain a model named 'ground_plane'."""
        model_names = {
            m.get("name", "") for m in sdf_root.findall(".//model")
        }
        assert "ground_plane" in model_names, (
            f"'ground_plane' model not found. Models: {model_names}"
        )

    def test_empty_room_has_walls(self, sdf_root: ET.Element) -> None:
        """World must contain at least 4 models with 'wall' in the name."""
        wall_models = [
            m for m in sdf_root.findall(".//model")
            if "wall" in m.get("name", "").lower()
        ]
        assert len(wall_models) >= 4, (
            f"Expected >= 4 wall models, found {len(wall_models)}: "
            f"{[m.get('name') for m in wall_models]}"
        )

    # -----------------------------------------------------------------------
    # Lights
    # -----------------------------------------------------------------------

    def test_empty_room_has_directional_light(self, sdf_root: ET.Element) -> None:
        """World must contain at least one directional light."""
        dir_lights = sdf_root.findall(".//light[@type='directional']")
        assert dir_lights, "No <light type='directional'> found in SDF"

    def test_empty_room_has_point_light(self, sdf_root: ET.Element) -> None:
        """World must contain at least one point light (ceiling lamp)."""
        point_lights = sdf_root.findall(".//light[@type='point']")
        assert point_lights, "No <light type='point'> found in SDF"

    # -----------------------------------------------------------------------
    # Dimensions
    # -----------------------------------------------------------------------

    def test_empty_room_walls_height_gte_2m(self, sdf_root: ET.Element) -> None:
        """All wall box geometries must have z-size >= 2.0 m."""
        wall_models = [
            m for m in sdf_root.findall(".//model")
            if "wall" in m.get("name", "").lower()
        ]
        assert wall_models, "No wall models found"
        for wall in wall_models:
            size_elem = wall.find(".//geometry/box/size")
            assert size_elem is not None, (
                f"No <geometry><box><size> in wall '{wall.get('name')}'"
            )
            z_size = float(size_elem.text.split()[2])
            assert z_size >= 2.0, (
                f"Wall '{wall.get('name')}' height {z_size} < 2.0 m"
            )

    def test_empty_room_floor_area_gte_20m2(self, sdf_root: ET.Element) -> None:
        """Ground plane size must be >= 4m x 5m (area >= 20 m²)."""
        ground = sdf_root.find(".//model[@name='ground_plane']")
        assert ground is not None, "'ground_plane' model not found"
        # plane geometry uses <size> with 2 values: width depth
        size_elem = ground.find(".//geometry/plane/size")
        assert size_elem is not None, (
            "No <geometry><plane><size> in ground_plane"
        )
        parts = size_elem.text.split()
        assert len(parts) >= 2, f"Unexpected plane size format: '{size_elem.text}'"
        width, depth = float(parts[0]), float(parts[1])
        area = width * depth
        assert area >= 20.0, (
            f"Ground plane area {area} m² < 20 m² (need >= 4x5)"
        )
        assert width >= 4.0, f"Ground plane width {width} < 4.0 m"
        assert depth >= 5.0 or width >= 5.0, (
            f"Neither dimension >= 5.0 m (got {width} x {depth})"
        )


# ---------------------------------------------------------------------------
# Fixtures — apartment.sdf
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def apt_tree() -> ET.ElementTree:
    """Parse apartment.sdf once for the whole module."""
    return ET.parse(str(_APARTMENT_FILE))


@pytest.fixture(scope="module")
def apt_root(apt_tree: ET.ElementTree) -> ET.Element:
    """Return the root <sdf> element of apartment.sdf."""
    return apt_tree.getroot()


@pytest.fixture(scope="module")
def apt_world(apt_root: ET.Element) -> ET.Element:
    """Return the <world> element of apartment.sdf."""
    elem = apt_root.find("world")
    assert elem is not None, "<world> element not found in apartment.sdf"
    return elem


# ---------------------------------------------------------------------------
# TestApartment
# ---------------------------------------------------------------------------


class TestApartment:
    """Validates the furnished apartment.sdf world for navigation + VLN testing."""

    # ------------------------------------------------------------------
    # Existence and parse
    # ------------------------------------------------------------------

    def test_apartment_sdf_exists(self) -> None:
        """gazebo/worlds/apartment.sdf must exist."""
        assert _APARTMENT_FILE.is_file(), f"File not found: {_APARTMENT_FILE}"

    def test_apartment_sdf_valid_xml(self) -> None:
        """apartment.sdf must be parseable as valid XML."""
        tree = ET.parse(str(_APARTMENT_FILE))
        assert tree.getroot() is not None

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    def test_apartment_has_world_element(self, apt_root: ET.Element) -> None:
        """Root <sdf> element must contain a <world name='apartment'> child."""
        world = apt_root.find("world")
        assert world is not None, "<world> element not found under <sdf>"
        assert world.get("name") == "apartment", (
            f"Expected world name='apartment', got '{world.get('name')}'"
        )

    # ------------------------------------------------------------------
    # Plugins
    # ------------------------------------------------------------------

    def test_apartment_has_physics_plugin(self, apt_root: ET.Element) -> None:
        """World must declare the gz::sim::systems::Physics plugin."""
        plugin_names = {
            p.get("name", "") for p in apt_root.findall(".//plugin")
        }
        assert "gz::sim::systems::Physics" in plugin_names, (
            f"gz::sim::systems::Physics not found. Plugins: {plugin_names}"
        )

    # ------------------------------------------------------------------
    # Room count (via point lights as per-room proxy)
    # ------------------------------------------------------------------

    def test_apartment_room_count_gte_5(self, apt_root: ET.Element) -> None:
        """There must be at least 5 point lights (one per room).

        Rooms: living room, kitchen, bedroom, bathroom, hallway.
        """
        point_lights = apt_root.findall(".//light[@type='point']")
        count = len(point_lights)
        assert count >= 5, (
            f"Expected >= 5 point lights (one per room), found {count}"
        )

    # ------------------------------------------------------------------
    # Door openings (wall segment count)
    # ------------------------------------------------------------------

    def test_apartment_has_door_openings(self, apt_root: ET.Element) -> None:
        """Walls must be split into segments to create door gaps.

        Requiring >= 8 wall-named models ensures multi-segment walls exist,
        meaning door openings (0.9 m gaps) are modelled.
        """
        wall_models = [
            m for m in apt_root.findall(".//model")
            if "wall" in m.get("name", "").lower()
        ]
        count = len(wall_models)
        assert count >= 8, (
            f"Expected >= 8 wall segments (doors implied by splits), found {count}: "
            f"{[m.get('name') for m in wall_models]}"
        )

    # ------------------------------------------------------------------
    # Furniture count
    # ------------------------------------------------------------------

    def test_apartment_has_furniture_gte_10(self, apt_root: ET.Element) -> None:
        """World must have at least 10 static furniture models.

        Furniture names follow room_itemname convention (e.g. sofa, bed, counter).
        Static models that are not ground planes, walls, or lights are counted.
        """
        _FURNITURE_KEYWORDS = {
            "sofa", "table", "bookshelf", "tv_stand", "counter", "cabinet",
            "chair", "bed", "nightstand", "dresser", "desk",
            "bathtub", "sink", "toilet",
        }
        furniture = [
            m for m in apt_root.findall(".//model")
            if any(kw in m.get("name", "").lower() for kw in _FURNITURE_KEYWORDS)
        ]
        count = len(furniture)
        assert count >= 10, (
            f"Expected >= 10 furniture models, found {count}: "
            f"{[m.get('name') for m in furniture]}"
        )

    # ------------------------------------------------------------------
    # Graspable objects (dynamic models)
    # ------------------------------------------------------------------

    def test_apartment_has_graspable_objects_gte_8(
        self, apt_root: ET.Element
    ) -> None:
        """World must have at least 8 dynamic (graspable) models.

        A dynamic model has <static>false</static>.
        """
        dynamic_models = [
            m for m in apt_root.findall(".//model")
            if m.find("static") is not None
            and m.find("static").text is not None
            and m.find("static").text.strip().lower() == "false"
        ]
        count = len(dynamic_models)
        assert count >= 8, (
            f"Expected >= 8 dynamic models, found {count}: "
            f"{[m.get('name') for m in dynamic_models]}"
        )

    # ------------------------------------------------------------------
    # Per-room lighting
    # ------------------------------------------------------------------

    def test_apartment_has_per_room_lighting(self, apt_root: ET.Element) -> None:
        """World must have at least 5 point lights (ceiling lamps, one per room)."""
        point_lights = apt_root.findall(".//light[@type='point']")
        count = len(point_lights)
        assert count >= 5, (
            f"Expected >= 5 point lights, found {count}"
        )

    # ------------------------------------------------------------------
    # Distinct floor materials
    # ------------------------------------------------------------------

    def test_apartment_floor_materials_differ(
        self, apt_root: ET.Element
    ) -> None:
        """Floor/ground visuals must use at least 3 distinct ambient colors.

        Each room has a different floor material so the VLM can distinguish rooms.
        """
        floor_models = [
            m for m in apt_root.findall(".//model")
            if "floor" in m.get("name", "").lower()
        ]
        # Collect ambient color strings from floor model visuals
        ambient_colors: set[str] = set()
        for floor in floor_models:
            for ambient in floor.findall(".//material/ambient"):
                if ambient.text:
                    ambient_colors.add(ambient.text.strip())
        assert len(ambient_colors) >= 3, (
            f"Expected >= 3 distinct floor ambient colors, found {len(ambient_colors)}: "
            f"{ambient_colors}"
        )

    # ------------------------------------------------------------------
    # Graspable objects have mass
    # ------------------------------------------------------------------

    def test_graspable_objects_have_mass(self, apt_root: ET.Element) -> None:
        """Every dynamic model must have a <mass> element with value > 0."""
        dynamic_models = [
            m for m in apt_root.findall(".//model")
            if m.find("static") is not None
            and m.find("static").text is not None
            and m.find("static").text.strip().lower() == "false"
        ]
        assert dynamic_models, "No dynamic models found"
        for model in dynamic_models:
            mass_elem = model.find(".//inertial/mass")
            assert mass_elem is not None, (
                f"Dynamic model '{model.get('name')}' missing <inertial><mass>"
            )
            assert mass_elem.text is not None, (
                f"Dynamic model '{model.get('name')}' has empty <mass>"
            )
            mass_val = float(mass_elem.text.strip())
            assert mass_val > 0, (
                f"Dynamic model '{model.get('name')}' has mass={mass_val} (must be > 0)"
            )

    # ------------------------------------------------------------------
    # Graspable objects have collision
    # ------------------------------------------------------------------

    def test_graspable_objects_have_collision(
        self, apt_root: ET.Element
    ) -> None:
        """Every dynamic model must have at least one <collision> element."""
        dynamic_models = [
            m for m in apt_root.findall(".//model")
            if m.find("static") is not None
            and m.find("static").text is not None
            and m.find("static").text.strip().lower() == "false"
        ]
        assert dynamic_models, "No dynamic models found"
        for model in dynamic_models:
            collisions = model.findall(".//collision")
            assert collisions, (
                f"Dynamic model '{model.get('name')}' has no <collision> element"
            )
