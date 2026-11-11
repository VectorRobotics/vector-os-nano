# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""One version source: vector_os_nano/version.py (pyproject reads it; vector-cli prints it)."""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import vector_os_nano
from vector_os_nano.vcli.cli import parse_args
from vector_os_nano.version import __version__

ROOT = Path(vector_os_nano.__file__).resolve().parents[1]


def test_vector_cli_prints_the_package_version(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        parse_args(["--version"])
    assert exc.value.code == 0 and capsys.readouterr().out.strip() == f"vector-cli {__version__}"


def test_pyproject_reads_the_version_from_version_py() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "version" not in project["project"] and "version" in project["project"]["dynamic"]
    assert project["tool"]["hatch"]["version"]["path"] == "vector_os_nano/version.py"
