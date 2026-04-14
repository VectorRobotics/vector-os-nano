"""Global abort signal for VGG task interruption.

All blocking loops (VGGHarness, GoalExecutor, navigate, explore)
check this signal. StopSkill and new-task-override set it.

Thread-safe: uses a single threading.Event.
"""
from __future__ import annotations

import threading
import time

_abort = threading.Event()


def request_abort() -> None:
    """Signal all running tasks to stop."""
    _abort.set()


def clear_abort() -> None:
    """Reset the abort flag. Called at the start of each new task."""
    _abort.clear()


def is_abort_requested() -> bool:
    """Check whether abort has been requested."""
    return _abort.is_set()


def wait_or_abort(seconds: float) -> bool:
    """Sleep up to *seconds*, returning early if abort is requested.

    Returns:
        True if abort was requested (caller should stop).
        False if the full duration elapsed normally.
    """
    return _abort.wait(timeout=seconds)
