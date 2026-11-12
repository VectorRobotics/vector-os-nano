# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""start_simulation / stop_simulation — launch or switch a sim by natural language.

Both are thin over :class:`vector_os_nano.sim.session.SimSession`: the ONE
in-process simulation (the same one ``--sim``/``--sim-go2``/``--embodiment``
start). The embodiment enum is generated from the manifest registry
(``embodiments/<id>/robot.yaml``); a different embodiment stops the running one
first, and a start that does not become ready is an error, never "Started".
"""
from __future__ import annotations

import logging
from typing import Any

from vector_os_nano.embodiments.registry import embodiment_ids
from vector_os_nano.vcli.tools.base import (
    ToolContext,
    ToolResult,
    tool,
)

logger = logging.getLogger(__name__)


def locate_mjpython(executable: str | None = None) -> str | None:
    """Locate the ``mjpython`` launcher for the running environment.

    mjpython lives next to the running interpreter (the venv's ``bin/``), so it is
    derived from ``sys.executable`` — deliberately NOT ``resolve()``-d: resolving
    follows the venv's ``python`` symlink to the base interpreter's ``bin``, where
    mjpython is absent. Falls back to ``shutil.which``. Returns an absolute path
    string, or ``None`` when mjpython is not installed.

    (Regression guard: a prior off-by-one ``parents[N]``-from-``__file__`` path
    resolved to ``$HOME``, so mjpython was never found and the viewer silently fell
    back to headless. Deriving from ``sys.executable`` is depth-independent.)
    """
    import os
    import shutil
    import sys
    from pathlib import Path

    exe = executable or sys.executable
    cand = Path(exe).parent / "mjpython"
    if cand.is_file() and os.access(str(cand), os.X_OK):
        return str(cand)
    return shutil.which("mjpython")


def _embodiment_menu() -> str:
    """'id' = display name, for every manifest (a broken one is listed as such)."""
    from vector_os_nano.embodiments.registry import load

    items = []
    for eid in embodiment_ids():
        try:
            items.append(f"'{eid}' = {load(eid).display_name}")
        except Exception:  # noqa: BLE001 — start_simulation reports the error
            items.append(f"'{eid}' (manifest error)")
    return "; ".join(items)


@tool(
    name="start_simulation",
    category="sim",
    description=(
        "Start, or switch to, an in-process robot simulation. The tool returns only "
        "once the sim is ready (physics stepping, camera live). A running sim of a "
        "different embodiment is stopped first."
    ),
    read_only=False,
    permission="allow",
)
class SimStartTool:
    """Start a SimSession for the requested embodiment and rebind the REPL to it."""

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "embodiment": {
                "type": "string",
                "enum": embodiment_ids(),
                "description": (
                    "Which body to simulate: " + _embodiment_menu() + ". Use the arm "
                    "variant (go2_piper) when the user asks for an arm / 机械臂 / "
                    "manipulation on the dog."
                ),
            },
            "gui": {
                "type": "boolean",
                "description": (
                    "Open the viewer window (default: true). When the user says "
                    "'headless' / '无窗口' / 'no window', pass gui=false."
                ),
                "default": True,
            },
        },
        "required": ["embodiment"],
    }

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        from vector_os_nano.sim.session import SimSession, SimSpec, SimStartError
        from vector_os_nano.vcli.runtime import bind_body

        app = context.app_state
        if app is None:
            return ToolResult(content="No app state available", is_error=True)
        gui = params.get("gui", True)
        if not isinstance(gui, bool):  # bool("false") is True: never guess a window
            return ToolResult(content=f"gui must be true or false, got {gui!r}", is_error=True)
        spec = SimSpec(embodiment=str(params.get("embodiment", "")), gui=gui)

        current = app.get("sim_session")
        if current is not None and current.spec == spec:
            return ToolResult(content=f"{spec.embodiment} simulation already running ({current.summary()}).")

        note = ""
        if spec.embodiment == "so101" and spec.gui and not self._viewer_available():
            # macOS: the viewer needs mjpython. The running REPL (and its conversation)
            # is never replaced: run headless and say how to get a window.
            spec = SimSpec(embodiment=spec.embodiment, gui=False)
            note = (" (headless: on macOS the viewer window needs mjpython; relaunch as "
                    "`mjpython -m vector_os_nano.vcli.cli`)")

        try:
            SimSession.validate(spec)  # before touching the running sim: a typo must not stop it
        except SimStartError as exc:
            return ToolResult(content=f"Cannot start {spec.embodiment!r}: {exc}", is_error=True)

        prefix = ""
        if current is not None:
            prefix = f"Stopped {current.spec.embodiment} ({current.stop()}). "
            bind_body(app, None)
        try:
            session = SimSession.start(spec)
        except SimStartError as exc:
            return ToolResult(
                content=f"{prefix}Failed to start {spec.embodiment} simulation: {exc}",
                is_error=True,
            )
        try:
            bind_body(app, session)
        except Exception as exc:  # noqa: BLE001 — a body the engine cannot bind must not run
            session.stop()
            return ToolResult(
                content=f"{prefix}Stopped the new {spec.embodiment} simulation: binding it failed: {exc}",
                is_error=True,
            )
        return ToolResult(content=f"{prefix}Started {spec.embodiment} simulation, {session.summary()}{note}")

    @staticmethod
    def _viewer_available() -> bool:
        """True unless this is macOS outside mjpython (where a viewer window needs it)."""
        import sys as _sys

        from vector_os_nano.hardware.sim.viewer_mode import running_under_mjpython

        return _sys.platform != "darwin" or running_under_mjpython()


@tool(
    name="stop_simulation",
    category="sim",
    description=(
        "Stop the running robot simulation: disconnects the sim, stops anything it "
        "spawned, and removes its skill tools. Call this when the user says "
        "'关闭仿真' / 'stop sim' / 'shutdown simulation' etc."
    ),
    read_only=False,
    permission="allow",
)
class SimStopTool:
    """Stop the running SimSession and rebind the REPL to the dev world."""

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
    }

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        from vector_os_nano.vcli.runtime import bind_body

        app = context.app_state
        if app is None:
            return ToolResult(content="No app state available", is_error=True)
        session = app.get("sim_session")
        if session is None:
            return ToolResult(content="No simulation is running.")
        summary = session.stop()
        bind_body(app, None)
        return ToolResult(content=f"Simulation stopped: {summary}.")
