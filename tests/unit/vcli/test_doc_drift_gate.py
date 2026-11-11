# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""S9 — CI doc-drift gate: the documented DEFAULT producer must match the LIVE default.

The #1 doc-rot failure this guards: ``docs/cli-tool-system.md`` once described the legacy
IntentRouter + ``VectorEngine.run_turn()`` keyword path as canonical while the live default had
moved to the native model-driven producer. This gate ties the doc's machine-readable claim
(``<!-- doc-drift-gate: default_producer=run_turn_native -->``) to the code's ACTUAL default —
so the documented producer can never silently diverge from the live one again (if S8, or anyone,
flips the default, this fails until the doc is updated).
"""
from __future__ import annotations

import re
from pathlib import Path

from vector_os_nano.vcli.engine import VectorEngine

_DOC = Path(__file__).resolve().parents[3] / "docs" / "cli-tool-system.md"


def test_documented_default_producer_matches_live_default(monkeypatch):
    text = _DOC.read_text(encoding="utf-8")
    m = re.search(r"doc-drift-gate:\s*default_producer=(\w+)", text)
    assert m, (
        "docs/cli-tool-system.md must carry the "
        "`<!-- doc-drift-gate: default_producer=... -->` marker (S9)"
    )
    documented = m.group(1)
    assert documented == "run_turn_native", (
        f"doc claims default_producer={documented!r}; this gate expects the native producer. "
        "If the default producer genuinely changed, update BOTH the code and this gate."
    )

    # The LIVE code must agree: the native loop is the ONLY producer (REPL, -p, MCP).
    assert callable(getattr(VectorEngine, "run_turn_native", None))
    assert not hasattr(VectorEngine, "run_turn"), "a second producer is back — doc drift"
