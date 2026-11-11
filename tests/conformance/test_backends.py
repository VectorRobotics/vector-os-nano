# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The backend table and the import firewall (hermetic: no engine is started)."""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import vector_os_nano
from vector_os_nano.interfaces import SimBackend
from vector_os_nano.sim import backends

PKG = Path(vector_os_nano.__file__).parent


def test_table_rows_are_backends():
    assert set(backends.BACKENDS) == {"mujoco", "pybullet"}
    for name in backends.BACKENDS:
        b = backends.get(name)
        assert isinstance(b, SimBackend) and b.name == name


def test_unknown_backend_names_the_valid_ones():
    with pytest.raises(KeyError, match=r"\['mujoco', 'pybullet'\]"):
        backends.get("bullet")


def _robot(ref: str) -> SimpleNamespace:
    return SimpleNamespace(policy=SimpleNamespace(ref=ref))


def test_supports_gives_a_reason_or_none(monkeypatch):
    assert backends.get("mujoco").supports(_robot("motion.pt"), None) is None
    pb = backends.get("pybullet")
    real = importlib.util.find_spec
    # the policy verdicts, as on a machine with the extra (whether or not this one has it)
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: object() if n == "pybullet" else real(n, *a))
    for ref in ("sinusoidal", "position_actuators"):  # certified on PyBullet (charter section 7)
        assert pb.supports(_robot(ref), None) is None, ref
    assert "falls" in pb.supports(_robot("convex_mpc"), None)  # measured: falls within 2 s
    assert "not certified" in pb.supports(_robot("my_new_mpc"), None)  # an unknown policy is a no
    assert "TorchScript" in pb.supports(_robot("motion.pt"), None)
    assert "TorchScript" in pb.supports(_robot("torchscript"), None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n == "pybullet" else real(n, *a))
    assert "pip install" in pb.supports(_robot("sinusoidal"), None)
    with pytest.raises(ImportError, match="pip install"):
        pb.load(None, gui=False)


def _imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module.split(".")[0] if not node.module.startswith("vector_os_nano") else node.module)
    return out


def _files(sub: str) -> list[Path]:
    return sorted((PKG / sub).rglob("*.py"))


def test_pybullet_is_imported_only_by_its_backend():
    own = PKG / "sim" / "backends" / "pybullet"
    offenders = [str(p.relative_to(PKG)) for p in PKG.rglob("*.py")
                 if _imports(p) & {"pybullet", "pybullet_utils"} and own not in p.parents]
    assert offenders == []


def test_new_layers_import_what_the_charter_allows():
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    for p in _files("interfaces"):
        bad = {m for m in _imports(p) if m not in stdlib | {"numpy"} and not m.startswith(
            ("vector_os_nano.interfaces", "vector_os_nano.schemas"))}
        assert not bad, f"{p.name}: {bad}"
    for p in _files("sim/backends/pybullet") + [PKG / "sim" / "backends" / "__init__.py"]:
        assert "mujoco" not in _imports(p), p  # it may read the compiled model via sim/scene only
    for p in _files("sim/scene") + _files("sim/backends/mujoco"):
        assert not _imports(p) & {"pybullet", "pybullet_utils"}, p
