# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Every module of the package imports: no module imports one that was deleted.

``__main__`` modules are entry points (importing one runs it) and are left out. An
optional third-party dependency that is not installed skips its module; a missing
``vector_os_nano`` module never does.
"""
from __future__ import annotations

import importlib
import pkgutil

import pytest

import vector_os_nano

_MODULES = sorted(
    m.name for m in pkgutil.walk_packages(vector_os_nano.__path__, "vector_os_nano.")
    if not m.name.endswith(".__main__")
)


@pytest.mark.parametrize("name", _MODULES)
def test_module_imports(name: str) -> None:
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if (exc.name or "").startswith("vector_os_nano"):
            raise
        pytest.skip(f"optional dependency {exc.name} is not installed")
