# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""sys.modules hygiene: fake ROS/mujoco/yaml modules never outlive their user.

A stub installed into ``sys.modules`` (at import time or in a test) used to stay
for the rest of the session, so every later file imported a fake ``rclpy`` (no
``__spec__``) or a product module bound to a MagicMock ``yaml``. tests/conftest.py
restores the watched entries around every test; ``stubbed_modules`` scopes a stub
install to one block (e.g. importing a module under test at collection time).
"""
from __future__ import annotations

import re
import sys
import types
from contextlib import contextmanager
from typing import Iterator

WATCHED = re.compile(r"^(rclpy|\w+_msgs|tf2_ros|unitree_\w+|yaml|mujoco|vector_os_nano)(\.|$)")


def is_stub(mod: object) -> bool:
    """A hand-made module (no import spec) or a mock standing in for one."""
    return not isinstance(mod, types.ModuleType) or getattr(mod, "__spec__", None) is None


def snapshot() -> dict[str, object]:
    return {k: v for k, v in sys.modules.items() if WATCHED.match(k)}


def restore(before: dict[str, object]) -> None:
    """Put replaced entries back and drop stubs added since *before*.

    A real module first imported since *before* is kept (a legit lazy import)
    unless a watched entry was tampered with meanwhile: then it may be bound to a
    stub, so it is dropped too and the next import gets a clean copy.
    """
    now = snapshot()
    added = {k: v for k, v in now.items() if k not in before}
    tampered = any(before[k] is not v for k, v in now.items() if k in before) or any(
        is_stub(v) for v in added.values()
    )
    for k, v in added.items():
        if tampered or is_stub(v):
            del sys.modules[k]
    for k, v in before.items():
        if sys.modules.get(k) is not v:
            sys.modules[k] = v


@contextmanager
def stubbed_modules(stubs: dict[str, types.ModuleType]) -> Iterator[None]:
    """Install *stubs* for this block only (never replacing an existing entry)."""
    before = snapshot()
    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)
    try:
        yield
    finally:
        restore(before)
