"""Enhanced skill schema tests — TDD for failure_modes in Skill protocol + to_schemas()."""
from __future__ import annotations


from vector_os_nano.core.skill import SkillRegistry, Skill


def test_to_schemas_includes_failure_modes():
    """to_schemas() output includes failure_modes when present on skill."""

    class TestSkill:
        name = "test"
        description = "test skill"
        parameters = {}
        preconditions = []
        postconditions = []
        effects = {}
        failure_modes = ["no_arm", "ik_unreachable"]
        __skill_aliases__ = []
        __skill_direct__ = False
        __skill_auto_steps__ = []

        def execute(self, params, context):
            pass

    registry = SkillRegistry()
    registry.register(TestSkill())
    schemas = registry.to_schemas()
    assert len(schemas) == 1
    assert "failure_modes" in schemas[0]
    assert schemas[0]["failure_modes"] == ["no_arm", "ik_unreachable"]


def test_to_schemas_omits_empty_failure_modes():
    """to_schemas() does not include failure_modes key when list is empty."""

    class TestSkill:
        name = "test"
        description = "test"
        parameters = {}
        preconditions = []
        postconditions = []
        effects = {}
        failure_modes = []
        __skill_aliases__ = []
        __skill_direct__ = False
        __skill_auto_steps__ = []

        def execute(self, params, context):
            pass

    registry = SkillRegistry()
    registry.register(TestSkill())
    schemas = registry.to_schemas()
    assert "failure_modes" not in schemas[0]


def test_to_schemas_backward_compat_no_failure_modes():
    """Skills without failure_modes attr still work (getattr default)."""

    class OldSkill:
        name = "old"
        description = "old"
        parameters = {}
        preconditions = []
        postconditions = []
        effects = {}
        # No failure_modes attribute!
        __skill_aliases__ = []
        __skill_direct__ = False
        __skill_auto_steps__ = []

        def execute(self, params, context):
            pass

    registry = SkillRegistry()
    registry.register(OldSkill())
    schemas = registry.to_schemas()
    assert "failure_modes" not in schemas[0]


def test_all_default_skills_have_failure_modes():
    """Every skill from get_default_skills() has failure_modes attribute."""
    from vector_os_nano.skills import get_default_skills

    for skill in get_default_skills():
        assert hasattr(skill, "failure_modes"), f"{skill.name} missing failure_modes"
        assert isinstance(skill.failure_modes, list), (
            f"{skill.name}.failure_modes is not a list"
        )


# ---------------------------------------------------------------------------
# T11: enum and source annotations for pick.py and place.py
# ---------------------------------------------------------------------------

def test_pick_schema_has_mode_enum():
    """pick.parameters['mode'] has an enum with exactly ['drop', 'hold']."""
    from vector_os_nano.skills.pick import PickSkill
    skill = PickSkill()
    assert "enum" in skill.parameters["mode"]
    assert skill.parameters["mode"]["enum"] == ["drop", "hold"]


def test_pick_schema_has_source_annotation():
    """pick.parameters['object_id'] and ['object_label'] have source annotations."""
    from vector_os_nano.skills.pick import PickSkill
    skill = PickSkill()
    assert skill.parameters["object_label"].get("source") == "world_model.objects.label"
    assert skill.parameters["object_id"].get("source") == "world_model.objects.object_id"


def test_place_schema_has_location_enum():
    """place.parameters['location'] has an enum with all 9 named locations."""
    from vector_os_nano.skills.place import PlaceSkill
    skill = PlaceSkill()
    assert "enum" in skill.parameters["location"]
    enum_values = skill.parameters["location"]["enum"]
    assert "front" in enum_values
    assert "left" in enum_values
    assert "back_right" in enum_values
    assert len(enum_values) == 9  # 9 named locations


def test_place_schema_location_enum_matches_location_map():
    """place.parameters['location']['enum'] is exactly the keys of _LOCATION_MAP."""
    from vector_os_nano.skills.place import PlaceSkill, _LOCATION_MAP
    skill = PlaceSkill()
    assert set(skill.parameters["location"]["enum"]) == set(_LOCATION_MAP.keys())


def test_place_schema_has_source_annotation():
    """place.parameters['location'] has source='static'."""
    from vector_os_nano.skills.place import PlaceSkill
    skill = PlaceSkill()
    assert skill.parameters["location"].get("source") == "static"


def test_mcp_tool_has_failure_modes():
    """skill_schema_to_mcp_tool passes through failure_modes."""
    from vector_os_nano.mcp.tools import skill_schema_to_mcp_tool
    schema = {
        "name": "pick",
        "description": "Pick up",
        "parameters": {},
        "failure_modes": ["no_arm", "ik_unreachable"],
    }
    tool = skill_schema_to_mcp_tool(schema)
    assert tool.get("failure_modes") == ["no_arm", "ik_unreachable"]


def test_mcp_tool_no_failure_modes_when_absent():
    """skill_schema_to_mcp_tool doesn't add failure_modes when not in schema."""
    from vector_os_nano.mcp.tools import skill_schema_to_mcp_tool
    schema = {"name": "test", "description": "test", "parameters": {}}
    tool = skill_schema_to_mcp_tool(schema)
    assert "failure_modes" not in tool


def test_mcp_tool_schema_has_enum():
    """skill_schema_to_mcp_tool passes through enum values into inputSchema."""
    from vector_os_nano.mcp.tools import skill_schema_to_mcp_tool
    schema = {
        "name": "pick",
        "description": "Pick up object",
        "parameters": {
            "mode": {
                "type": "string",
                "enum": ["drop", "hold"],
                "default": "drop",
                "description": "...",
            }
        },
    }
    tool = skill_schema_to_mcp_tool(schema)
    assert tool["inputSchema"]["properties"]["mode"]["enum"] == ["drop", "hold"]
