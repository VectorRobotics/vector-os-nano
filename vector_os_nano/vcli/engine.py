# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""VectorEngine — the binding every native turn runs on.

The engine holds the LLM backend, the tool registry and the permission context,
and is bound (``bind``) to the body and world a turn drives: the agent, the world,
the world's routable capabilities and its verify namespace. A turn is produced by
the ONE producer, ``run_turn_native`` (``vcli/native_loop.py``); the engine also
owns the single-flight actuating run and the P0 stop.
"""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Callable

from vector_os_nano.vcli.backends import LLMBackend
from vector_os_nano.vcli.cognitive import abort as _abort
from vector_os_nano.vcli.cognitive.capabilities.registry import CapabilityRegistry
from vector_os_nano.vcli.permissions import PermissionContext
from vector_os_nano.vcli.session import Session
from vector_os_nano.vcli.tools.base import ToolRegistry

logger = logging.getLogger(__name__)

# How long a new actuating run waits for the cancelled previous one to let go.
_PREEMPT_WAIT_S = 30.0

# What a world must implement to be bound (vcli/worlds/base.py World).
_WORLD_MEMBERS: tuple[str, ...] = (
    "is_robot", "persona_blocks", "build_verify_namespace", "register_capabilities",
)


class VectorEngine:
    """The LLM backend, tools and permissions, bound to one body and world.

    Thread-safety: one engine per agent session; actuating runs are single-flight.
    """

    def __init__(
        self,
        backend: LLMBackend,
        registry: ToolRegistry | None = None,
        permissions: PermissionContext | None = None,
        max_tokens: int = 16384,
    ) -> None:
        self._backend: LLMBackend = backend
        self._registry: ToolRegistry = registry or ToolRegistry()
        self._permissions: PermissionContext = permissions or PermissionContext()
        self._max_tokens: int = max_tokens

        # Set by bind(): the body, its world and the world's routable capabilities.
        self._agent: Any = None
        self._world: Any = None
        self._capability_registry: CapabilityRegistry | None = None

        # Single-flight for runs that may move the robot (actuating_run).
        self._run_lock = threading.Lock()
        self._run_token: Any = None
        self._run_owner: int | None = None

    def bind(self, agent: Any, world: Any) -> None:
        """Bind the body (*agent*, None for the dev world) and the *world* turns run on.

        Builds the world's capability registry and checks its verify namespace into
        locals first and commits all three only when both succeed: a failure raises
        and leaves the previous binding whole. A *world* missing a World member fails
        loud with the members a world must implement.
        """
        missing = [m for m in _WORLD_MEMBERS if not callable(getattr(world, m, None))]
        if missing:
            raise TypeError(
                f"cannot bind {world!r}: it lacks {missing}; a world implements "
                f"{list(_WORLD_MEMBERS)}"
            )
        capabilities = CapabilityRegistry()
        world.register_capabilities(capabilities, agent, self._backend)
        world.build_verify_namespace(agent)
        self._agent, self._world, self._capability_registry = agent, world, capabilities

    def _build_verifier_namespace(self, agent: Any) -> dict[str, Any]:
        """The verify namespace for *agent*: the active world's predicates, nothing else.

        Only ``@oracle``-tagged callables of the active world's scope count as
        evidence (``trace_store.verify_oracle_names``). The kernel adds none: the
        agent's own pose reads are a self-report, never evidence, so a body without
        a ground-truth world (real hardware) verifies nothing GROUNDED. A verify
        naming an unavailable function fails closed (NameError -> False); a world
        whose namespace raises contributes nothing.
        """
        world = self._world
        if world is None:
            from vector_os_nano.vcli.worlds.registry import resolve_world

            world = resolve_world(agent)
        try:
            return dict(world.build_verify_namespace(agent) or {})
        except Exception as exc:  # noqa: BLE001 — fail closed: no world oracles
            logger.debug("world %r build_verify_namespace failed: %s", getattr(world, "name", world), exc)
            return {}

    def _has_live_viewer(self) -> bool:
        """True when a connected sim has an OPEN GUI viewer.

        Used (together with ``running_under_mjpython()``) to gate synchronous
        turn execution: only under mjpython/macOS is GLFW main-thread-only and the
        caller's thread the viewer owner. Duck-types the connected agent's arm/base
        for a live viewer handle (no world import, no embodiment branch).
        """
        agent = self._agent
        if agent is None:
            return False
        for attr in ("_arm", "_base"):
            hw = getattr(agent, attr, None)
            if hw is None:
                continue
            if getattr(hw, "_viewer", None) is not None or getattr(hw, "viewer", None) is not None:
                return True
        return False

    def _emergency_stop(
        self,
        user_message: str,
        session: Session | None,
        agent: Any = None,
        app_state: dict[str, Any] | None = None,
    ) -> None:
        """P0 stop bypass — cancel every run and execute StopSkill directly, no LLM call.

        ``session`` None (a stop typed while a turn runs on another thread) skips the
        session record, so the stop never races the running turn's session writes.
        """
        _agent = agent or self._agent
        # The motors first: no skill context to build, no run to wake (BaseProtocol.stop
        # is ungated and lets a moving body brake while its walk still holds the token).
        try:
            stop_base = getattr(getattr(_agent, "_base", None), "stop", None)
            if callable(stop_base):
                stop_base()
        except Exception as exc:  # noqa: BLE001 — the stop skill below stops it again
            logger.warning("P0 stop: base.stop failed: %s", exc)
        _abort.request_abort()  # every live run's CancelToken (background ones too)

        # The stop skill itself — never via agent.execute_skill, whose plan appends a
        # 'home' step (moving the arm and opening the gripper is not a stop).
        reg = getattr(_agent, "_skill_registry", None)
        stop = reg.get("stop") if reg is not None else None
        if stop is not None:
            try:
                stop.execute({}, _agent._build_context())
            except Exception as exc:  # noqa: BLE001 — the run tokens are cancelled already
                logger.warning("P0 stop: the stop skill failed: %s", exc)

        if session is not None:
            session.append_user(user_message)
            session.append_assistant("Stopped.", None)

    def stop_words(self) -> frozenset[str]:
        """The P0 stop phrases: the REGISTERED stop skill's name + aliases (lowercase).

        Single-sourced from the skill (never a second hand-written list); no stop
        skill registered (no robot) -> none.
        """
        reg = getattr(self._agent, "_skill_registry", None)
        stop = reg.get("stop") if reg is not None else None
        if stop is None:
            return frozenset()
        words = [getattr(stop, "name", "stop"), *getattr(stop, "__skill_aliases__", [])]
        return frozenset(str(w).strip().lower() for w in words if str(w).strip())

    def is_stop(self, text: str) -> bool:
        # '停！' / 'stop!' are the same stop (a panicked owner punctuates); '停?' is not.
        return text.strip().lower().rstrip("!！。.~～ ") in self.stop_words()

    @contextmanager
    def actuating_run(self):
        """Single-flight for a run that may move the robot; yields its CancelToken.

        Busy -> the previous run is CANCELLED, then this one waits for it to finish
        (bounded) before taking over. Re-entry from the run's own thread (a native
        turn whose tool starts a nested execute) shares the outer run's token.
        """
        tid = threading.get_ident()
        if self._run_owner == tid and self._run_token is not None:
            yield self._run_token
            return
        prev = self._run_token
        if prev is not None:
            prev.cancel()
        if not self._run_lock.acquire(timeout=_PREEMPT_WAIT_S):
            raise RuntimeError(
                f"the previous robot task did not stop within {_PREEMPT_WAIT_S:.0f}s"
            )
        token = _abort.new_run()
        self._run_token, self._run_owner = token, tid
        try:
            yield token
        finally:
            self._run_token, self._run_owner = None, None
            self._run_lock.release()

    @property
    def busy(self) -> bool:
        """True while an actuating run holds the robot."""
        return self._run_token is not None

    def run_turn_native(
        self,
        user_message: str,
        agent: Any = None,
        session: Session | None = None,
        app_state: dict[str, Any] | None = None,
        on_progress: Callable[[str], None] | None = None,
        ask_permission: Callable[[str, dict[str, Any]], str] | None = None,
    ) -> Any:
        """Run ONE turn through the native tool-use producer; return its ExecutionTrace.

        Runs as ONE actuating run (single-flight + its CancelToken). ``ask_permission``
        resolves ``ask``-level tool permissions; None denies (the headless default).
        The producer lives in ``vcli/native_loop.py``; each caller feeds the trace to
        ``VerdictReport.from_trace`` — this never computes ``verified``.
        """
        from vector_os_nano.vcli.native_loop import run_turn_native as _run

        with self.actuating_run() as token:
            return _run(
                self,
                user_message,
                agent=agent if agent is not None else self._agent,
                session=session,
                app_state=app_state,
                on_progress=on_progress,
                ask_permission=ask_permission,
                cancel=token,
            )
