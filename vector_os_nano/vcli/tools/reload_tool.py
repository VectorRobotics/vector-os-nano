# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Hot reload tool for robot skills.

Reloads a skill's Python module and re-registers the skill in the
SkillRegistry without restarting the simulation.

Limitations (documented in tool description):
- Cannot reload modules with active background threads (explore.py)
- Frozen dataclass instances created before reload keep old class
- Module-level singletons (e.g., _explore_thread) reset on reload
"""
from __future__ import annotations

import importlib
import logging
import sys
from typing import Any

from vector_os_nano.vcli.tools.base import ToolContext, ToolResult, live_agent, tool

logger = logging.getLogger(__name__)


@tool(name="skill_reload",
      description=(
          "Hot-reload a robot skill module after code changes. "
          "Re-imports the Python module and re-registers the skill. "
          "Use after editing a skill file to apply changes without restart. "
          "LIMITATION: Do not reload modules with background threads (explore.py) "
          "or modules that hold persistent state."
      ),
      read_only=False, permission="ask", category="system")
class SkillReloadTool:
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "Name of the skill to reload (e.g., 'stand', 'walk', 'navigate').",
            },
        },
        "required": ["skill_name"],
    }

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        skill_name: str = params["skill_name"]

        # The live agent's registry: the one the skill tools resolve from at call time
        # (a registry cached at REPL start goes stale after a sim switch).
        registry = getattr(live_agent(context), "_skill_registry", None)
        if registry is None:
            return ToolResult(content="No skill registry available.", is_error=True)

        # Look up the current skill instance.
        skill = registry.get(skill_name)
        if skill is None:
            available = registry.list_skills()
            return ToolResult(
                content=(
                    f"Unknown skill: '{skill_name}'. Available: {', '.join(available)}"
                ),
                is_error=True,
            )

        # Derive module and class from the current instance.
        module_name: str = skill.__class__.__module__
        class_name: str = skill.__class__.__name__

        if module_name not in sys.modules:
            return ToolResult(
                content=f"Module '{module_name}' not in sys.modules. Cannot reload.",
                is_error=True,
            )

        # Reload module — catches syntax errors separately for a cleaner message.
        try:
            importlib.invalidate_caches()
            module = importlib.reload(sys.modules[module_name])
        except SyntaxError as exc:
            return ToolResult(
                content=f"Syntax error in {module_name}: {exc}",
                is_error=True,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                content=f"Failed to reload {module_name}: {exc}",
                is_error=True,
            )

        # Retrieve the class from the freshly-reloaded module.
        new_class = getattr(module, class_name, None)
        if new_class is None:
            return ToolResult(
                content=(
                    f"Class '{class_name}' not found in reloaded module '{module_name}'."
                ),
                is_error=True,
            )

        # Instantiate and re-register.
        try:
            new_instance = new_class()
            registry.register(new_instance)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                content=f"Failed to re-register {skill_name}: {exc}",
                is_error=True,
            )
        if registry.get(skill_name) is not new_instance:
            return ToolResult(
                content=(
                    f"Reloaded {module_name}.{class_name}, but '{skill_name}' still resolves "
                    f"to the old skill (the new class registers as "
                    f"'{getattr(new_instance, 'name', '?')}'): the reload did not take effect."
                ),
                is_error=True,
            )

        logger.info(
            "Reloaded skill '%s' from %s.%s", skill_name, module_name, class_name
        )
        return ToolResult(
            content=f"Reloaded '{skill_name}' from {module_name}.{class_name}",
            metadata={
                "skill_name": skill_name,
                "module": module_name,
                "class": class_name,
            },
        )
