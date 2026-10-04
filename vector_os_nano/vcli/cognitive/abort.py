# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Per-run cancellation: one ``CancelToken`` per run, behind the old module facade.

A run (a native turn, a VGG execute, a ReAct turn) owns ONE token, bound to the
thread that runs it. Blocking loops (VGGHarness, GoalExecutor, navigate, walk,
explore) poll ``is_abort_requested()`` / ``wait_or_abort()``, which read the
CALLING thread's token. A stop (``request_abort``) cancels every live token at
once — stop means stop everything. Starting a new run (``new_run`` /
``clear_abort``) binds a FRESH token to the calling thread and never un-cancels
another run's token, so a chat turn can no longer clear the abort a stop just set
on a background run.

A user turn (``turn()``) is the stop's scope: every run the turn starts shares the
turn's token, so a stop that lands between two runs of one turn (native's LLM call,
then a legacy plan) still stops the later one.

Thread-safe: a token wraps one ``threading.Event``; the live set is guarded.
"""
from __future__ import annotations

import threading
import weakref
from contextlib import contextmanager
from typing import Iterator


class CancelToken:
    """One run's cancel signal. Event-compatible (``is_set``/``set``/``wait``), so it
    drops into ``ToolContext.abort``."""

    def __init__(self) -> None:
        self._ev = threading.Event()

    def cancel(self) -> None:
        self._ev.set()

    set = cancel  # threading.Event shape

    @property
    def cancelled(self) -> bool:
        return self._ev.is_set()

    def is_set(self) -> bool:
        return self._ev.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Sleep up to *timeout*; True as soon as the token is cancelled."""
        return self._ev.wait(timeout)


_lock = threading.Lock()
_live: "weakref.WeakSet[CancelToken]" = weakref.WeakSet()
_bound = threading.local()
_latest = CancelToken()  # the newest run's token; read by threads bound to no run
_live.add(_latest)


def new_run() -> CancelToken:
    """Start a run on this thread: a fresh token, bound here, cancelled by a stop.

    Inside a ``turn()`` the run shares the turn's token (cancelled already if the
    turn was stopped): a stopped turn cannot start a fresh, uncancelled run.
    """
    global _latest
    token = getattr(_bound, "turn", None) or CancelToken()
    with _lock:
        _live.add(token)
        _latest = token
    _bound.token = token
    return token


@contextmanager
def turn() -> Iterator[CancelToken]:
    """One user turn on this thread: a stop anywhere in it stops every run it starts."""
    _bound.turn = None
    _bound.turn = new_run()
    try:
        yield _bound.turn
    finally:
        _bound.turn = None


def current_token() -> CancelToken:
    """The calling thread's run token.

    ponytail: a helper thread a run spawns (a thread pool, a timer) is bound to no
    run and reads the newest run's token; pass the token explicitly (ToolContext.abort)
    where that matters.
    """
    return getattr(_bound, "token", None) or _latest


def request_abort(*, spare_caller: bool = False) -> None:
    """Stop: cancel every live run's token.

    ``spare_caller`` keeps the run bound to the calling thread going: a stop STEP
    inside a run ("walk two steps, then stop") stops everything else, not the run
    that asked for it. The owner's P0 stop never spares anything.
    """
    own = getattr(_bound, "token", None) if spare_caller else None
    with _lock:
        tokens = [t for t in _live if t is not own]
    for t in tokens:
        t.cancel()


def clear_abort() -> None:
    """Facade for 'a new task starts here': bind a fresh token to the calling thread.

    Never un-cancels another run's token (a new turn cannot clear another run's abort).
    """
    new_run()


def is_abort_requested() -> bool:
    """True when the calling thread's run has been cancelled."""
    return current_token().cancelled


def wait_or_abort(seconds: float) -> bool:
    """Sleep up to *seconds*, returning early if this thread's run is cancelled.

    Returns:
        True if abort was requested (caller should stop).
        False if the full duration elapsed normally.
    """
    return current_token().wait(seconds)
