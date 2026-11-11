# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""REPL log-spam quieting (FIX-logspam).

On the live SO-101 arm REPL, a step failure (e.g. "no strategy matched",
"execution failed") is emitted as a cognitive-layer WARNING AND already rendered
in the rich step UI ("[FAIL] ..."). With ``basicConfig(level=WARNING)`` those
duplicate WARNINGs propagate to the root console handler and flood the screen
(2x per step across retries) — pure noise.

``cli._setup_logging`` quiets the cognitive package logger to ERROR on the
non-verbose REPL console only. This must:
  - quiet ``vector_os_nano.vcli.cognitive`` to >= ERROR when NOT --verbose,
  - leave it un-quieted under --verbose (full DEBUG preserved),
  - NOT silence ERROR/CRITICAL (only WARNING/INFO/DEBUG are dropped),
  - scope to the cognitive package only (never the root logger).

The fix lives on the CLI entry path; library code / the test suite never call it.
"""

from __future__ import annotations

import logging

import pytest

from vector_os_nano.vcli import cli

COGNITIVE = "vector_os_nano.vcli.cognitive"

# Additional namespaces that FIX 2 extends the quieting to.
QUIET_NAMESPACES = [
    "vector_os_nano.vcli.cognitive",
    "vector_os_nano.skills",
    "vector_os_nano.perception",
    "vector_os_nano.hardware",
]


@pytest.fixture(autouse=True)
def _restore_quiet_logger_levels():
    """Snapshot/restore all quietable logger levels so tests don't leak state."""
    saved = {name: logging.getLogger(name).level for name in QUIET_NAMESPACES}
    try:
        yield
    finally:
        for name, level in saved.items():
            logging.getLogger(name).setLevel(level)


def test_non_verbose_quiets_cognitive_logger():
    """Non-verbose REPL: cognitive-layer WARNINGs are quieted (effective >= ERROR)."""
    logging.getLogger(COGNITIVE).setLevel(logging.NOTSET)

    cli._setup_logging(verbose=False)

    log = logging.getLogger(COGNITIVE)
    assert log.getEffectiveLevel() >= logging.ERROR
    # The noisy levels are dropped, but real errors still surface.
    assert not log.isEnabledFor(logging.WARNING)
    assert log.isEnabledFor(logging.ERROR)
    assert log.isEnabledFor(logging.CRITICAL)


def test_verbose_does_not_quiet_cognitive_logger():
    """--verbose REPL: the cognitive layer is NOT quieted (no ERROR pin)."""
    # Simulate a prior non-verbose quieting to prove --verbose restores it.
    logging.getLogger(COGNITIVE).setLevel(logging.ERROR)

    cli._setup_logging(verbose=True)

    log = logging.getLogger(COGNITIVE)
    # --verbose must REMOVE any explicit ERROR pin (reset to NOTSET) so the
    # cognitive logger inherits the (DEBUG) root level instead of swallowing
    # WARNINGs. We assert on the logger's own level rather than the effective
    # level, since basicConfig only raises the root level when it installs the
    # root handler (already present under pytest) — the no-quieting contract is
    # what matters, not whether this process's root happens to be at DEBUG.
    assert log.level == logging.NOTSET
    assert log.getEffectiveLevel() < logging.ERROR
    assert log.isEnabledFor(logging.WARNING)


def test_quieting_is_scoped_to_cognitive_not_root():
    """The quieting touches ONLY the cognitive package logger, not the root."""
    root = logging.getLogger()
    saved_root = root.level
    logging.getLogger(COGNITIVE).setLevel(logging.NOTSET)
    try:
        cli._setup_logging(verbose=False)
        # Root is not raised to ERROR by the quieting — it stays at the
        # basicConfig WARNING (or lower if already configured); never >= ERROR
        # purely as a side effect of quieting the cognitive logger.
        assert root.level <= logging.WARNING
        # The explicit override lives on the cognitive logger itself.
        assert logging.getLogger(COGNITIVE).level == logging.ERROR
    finally:
        root.setLevel(saved_root)


def test_sibling_loggers_not_quieted():
    """A non-cognitive logger (e.g. the engine) is unaffected by the quieting."""
    engine_log = logging.getLogger("vector_os_nano.vcli.engine")
    saved = engine_log.level
    engine_log.setLevel(logging.NOTSET)
    logging.getLogger(COGNITIVE).setLevel(logging.NOTSET)
    try:
        cli._setup_logging(verbose=False)
        # Engine WARNINGs are NOT collateral-damaged by the cognitive quieting.
        assert engine_log.isEnabledFor(logging.WARNING)
    finally:
        engine_log.setLevel(saved)


def test_all_noisy_namespaces_quieted_non_verbose():
    """Non-verbose: skills, perception, hardware loggers are also quieted to ERROR."""
    for name in QUIET_NAMESPACES:
        logging.getLogger(name).setLevel(logging.NOTSET)
    cli._setup_logging(verbose=False)
    for name in QUIET_NAMESPACES:
        log = logging.getLogger(name)
        assert log.getEffectiveLevel() >= logging.ERROR, (
            f"{name} should be quieted to ERROR in non-verbose mode"
        )
        assert not log.isEnabledFor(logging.WARNING), (
            f"{name} should not emit WARNINGs in non-verbose mode"
        )


def test_explore_events_follow_the_console_stream(capsys):
    """cli-entry:explore-events-stdout-in-json: under ``-p --json`` the console is on
    stderr, so explore progress must be too — stdout carries only the verdict line."""
    from rich.console import Console

    from vector_os_nano.skills.go2 import explore

    saved = explore._on_event
    try:
        cli._setup_explore_events(Console(stderr=True))
        explore._emit("started", {"total_rooms": 8})
    finally:
        explore.set_event_callback(saved)
    out = capsys.readouterr()
    assert out.out == ""
    assert "Exploration started (8 rooms" in out.err


def test_all_noisy_namespaces_restored_verbose():
    """--verbose restores all quieted namespaces to NOTSET."""
    # Simulate prior non-verbose quieting on all namespaces.
    for name in QUIET_NAMESPACES:
        logging.getLogger(name).setLevel(logging.ERROR)
    cli._setup_logging(verbose=True)
    for name in QUIET_NAMESPACES:
        log = logging.getLogger(name)
        assert log.level == logging.NOTSET, (
            f"{name} should be NOTSET (not pinned) under --verbose"
        )
