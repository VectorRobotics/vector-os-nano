# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The ONE runtime builder: everything a face needs to run a turn.

``build_runtime`` is called by every face — the REPL, ``-p`` and the MCP server —
and creates, in order: the credentials, the LLM backend, the body (a sim when one
was asked for), the tool registry, the session and the engine bound to the body's
world (its verify namespace and capabilities). The prompt is built per round-trip
by the producer from that binding (``native_loop._native_system_prompt``).

``bind_body`` is the ONE path that points a runtime at a body: the builder uses it,
and so do ``start_simulation`` / ``stop_simulation`` when they switch bodies.
``run_turn`` is the ONE way a face runs a turn: the faces differ only in how they
read input and print.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable

from vector_os_nano.vcli.engine import VectorEngine
from vector_os_nano.vcli.permissions import PermissionContext
from vector_os_nano.vcli.session import Session, create_session, get_latest_session, load_session

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    """What a face holds: the engine, the session and the shared app state.

    ``app_state`` is the mutable binding the tools see (``agent``, ``registry``,
    ``engine``, ``sim_session``, ``scene_graph``, ``skill_registry``, ``model``,
    ``provider``, ``api_key``, ``base_url``); ``bind_body`` keeps it current.
    """

    engine: VectorEngine | None
    session: Session
    app_state: dict[str, Any]

    @property
    def agent(self) -> Any:
        return self.app_state.get("agent")

    @property
    def registry(self) -> Any:
        return self.app_state["registry"]


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def make_backend(*, provider: str, api_key: str, model: str, base_url: str | None) -> Any:
    """The LLM backend, with the deterministic test seam.

    ``VECTOR_FAKE_LLM_TOOLS=<json-path>`` replaces the network model with a script
    replay (``tests/harness/fake_backend.py``); every tool, verify, permission and
    verdict is still the product's own.
    """
    from vector_os_nano.vcli.backends import create_backend

    tools_path = os.environ.get("VECTOR_FAKE_LLM_TOOLS")
    if tools_path:
        # TEST-ONLY import (lazy, gated on the env var): production never reaches it.
        from tests.harness.fake_backend import FakeToolScriptBackend

        return FakeToolScriptBackend.from_json_file(tools_path)
    return create_backend(provider=provider, api_key=api_key, model=model, base_url=base_url)


def build_registry() -> Any:
    """The built-in tools, each under its category (the skill tools come with a body)."""
    from vector_os_nano.vcli.tools import CategorizedToolRegistry, discover_categorized_tools

    registry = CategorizedToolRegistry()
    tools, categories = discover_categorized_tools()
    for t in tools:
        registry.register(t, category=next((c for c, names in categories.items() if t.name in names), "default"))
    return registry


def start_sim(embodiment: str, *, gui: bool, notice: Callable[[str], None] = _stderr) -> Any:
    """Start the SimSession for *embodiment* (the one NL ``start_simulation`` uses).

    Raises ``SimStartError``: a requested sim that does not come up never degrades
    silently to the dev world.
    """
    from vector_os_nano.sim.session import SimSession, SimSpec

    notice(f"  Starting {embodiment} simulation...")
    # No vlm_api_key: the chat LLM's credential is never sent to the VLM provider.
    session = SimSession.start(SimSpec(embodiment=embodiment, gui=gui))
    notice(f"  {embodiment} {session.summary()}")
    return session


def bind_body(app_state: dict[str, Any], session: Any, agent: Any = None) -> None:
    """Point the runtime at *session*'s body (or at *agent*, or at none: the dev world).

    The engine is bound first: a bind failure raises before anything else changed.
    Then the app state and the skill tools follow the body.
    """
    from vector_os_nano.vcli.tools.skill_wrapper import wrap_skills
    from vector_os_nano.vcli.worlds import resolve_world

    if session is not None:
        agent = session.agent
    engine = app_state.get("engine")
    if engine is not None:
        engine.bind(agent, resolve_world(agent))
    app_state.update(sim_session=session, agent=agent,
                     scene_graph=getattr(agent, "_spatial_memory", None),
                     skill_registry=getattr(agent, "_skill_registry", None))
    registry = app_state.get("registry")
    if registry is not None:
        for name in list(registry.list_tools()):
            if getattr(registry.get(name), "_is_skill_wrapper", False):
                registry.unregister(name)
        if agent is not None:
            for t in wrap_skills(agent):
                registry.register(t, category="robot")


def _open_session(resume: str | None, metadata: dict[str, Any], notice: Callable[[str], None]) -> Session:
    if resume is None:
        return create_session(metadata=metadata)
    if resume == "latest":
        loaded = get_latest_session()
        if loaded is None:
            notice("No previous session, starting new.")
            return create_session(metadata=metadata)
        notice(f"Resumed: {loaded.session_id}")
        return loaded
    session = load_session(resume)
    notice(f"Resumed: {session.session_id}")
    return session


def build_runtime(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    embodiment: str | None = None,
    gui: bool = True,
    agent: Any = None,
    no_permission: bool = False,
    resume: str | None = None,
    face: str = "repl",
    notice: Callable[[str], None] = _stderr,
) -> Runtime:
    """Build everything a turn needs, for any face.

    Credentials -> the body (``embodiment`` starts a sim; ``agent`` is a body the
    caller already built) -> the tool registry -> the session -> the engine, bound
    to the body and its world. With no key there is no engine (a face says so). A
    sim start or bind failure raises; the caller reports it.
    """
    from vector_os_nano.vcli.config import resolve_credentials

    api_key, provider, model, base_url = resolve_credentials(
        cli_api_key=api_key, cli_base_url=base_url, cli_model=model,
    )
    if not api_key:
        notice("No API key configured.")
        notice("  /login claude     auto-detect Claude Code subscription")
        notice("  /login anthropic  enter Anthropic API key")
        notice("  /login openrouter enter OpenRouter key\n")

    sim_session = start_sim(embodiment, gui=gui, notice=notice) if embodiment else None
    registry = build_registry()
    session = _open_session(resume, {"model": model, "face": face}, notice)
    engine = None
    if api_key:
        engine = VectorEngine(
            backend=make_backend(provider=provider, api_key=api_key, model=model, base_url=base_url),
            registry=registry, permissions=PermissionContext(no_permission=no_permission),
        )
    app_state: dict[str, Any] = {
        "registry": registry, "engine": engine, "model": model, "provider": provider,
        "api_key": api_key, "base_url": base_url,
    }
    try:
        bind_body(app_state, sim_session, agent)
    except BaseException:
        if sim_session is not None:
            sim_session.stop()
        raise
    return Runtime(engine=engine, session=session, app_state=app_state)


def flush(rt: Runtime) -> None:
    """Persist the transcript and the scene graph NOW (between turns).

    A MuJoCo segfault kills the process without the exit-time saves, so this bounds
    the loss to the turn that crashed.
    """
    try:
        rt.session.save()
        sm = getattr(rt.agent, "_spatial_memory", None)
        if sm is not None and hasattr(sm, "save"):
            sm.save()
    except Exception as exc:  # noqa: BLE001 — persistence is best-effort, never a crash
        logger.debug("per-turn flush failed: %s", exc)


def run_turn(
    rt: Runtime,
    text: str,
    *,
    on_progress: Callable[[str], None] | None = None,
    ask_permission: Callable[[str, dict[str, Any]], str] | None = None,
) -> tuple[Any, str, list[dict[str, Any]]]:
    """Run ONE turn through the native producer on the runtime's real session.

    Returns ``(trace, answer, entries)``: the ExecutionTrace (the face grades it with
    ``VerdictReport.from_trace``), the model's last words and the turn's session
    entries. ``ask_permission`` None denies every ``ask`` (no one to ask).
    """
    session, engine = rt.session, rt.engine
    n0 = len(session._entries)
    trace = engine.run_turn_native(
        text, agent=getattr(engine, "_agent", None), session=session, app_state=rt.app_state,
        on_progress=on_progress, ask_permission=ask_permission,
    )
    entries = session._entries[n0:]
    answer = session.last_assistant_text(n0)
    if len(session._entries) > 50:
        session.compact(keep_recent=12)
    flush(rt)
    return trace, answer, entries
