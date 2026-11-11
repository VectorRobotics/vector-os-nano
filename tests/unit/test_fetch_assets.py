# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""C0 — scripts/fetch_assets.py against a local upstream git repository (no network).

The real fetch (unitree_rl_gym at the pinned commit) is proven once by hand; these
pin the contract: exactly the listed files, every sha256, the licence, idempotent,
--check, and a mismatch that names the file and deletes nothing.
"""
from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from vector_os_nano.embodiments import assets
from vector_os_nano.embodiments.config import AssetsSpec

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fetch_assets.py"
_FILES = {"model.xml": b"<mujoco/>", "meshes/a.STL": b"mesh-a", "meshes/b.STL": b"mesh-b"}


def _load_script():
    spec = importlib.util.spec_from_file_location("fetch_assets", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _upstream(root: Path, files: dict[str, bytes]) -> str:
    """A git repo laid out like an upstream: robot/<file> plus an unlisted extra and LICENSE."""
    for rel, data in {**{f"robot/{k}": v for k, v in files.items()},
                      "robot/meshes/unused.STL": b"x", "LICENSE": b"BSD 3-Clause"}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run([*git, "config", "uploadpack.allowFilter", "true"], check=True)
    subprocess.run([*git, "config", "uploadpack.allowAnySHA1InWant", "true"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "upstream"], check=True)
    return subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """The script pointed at a fake robot whose recipe fetches from a local repo."""
    up = tmp_path / "upstream"
    rev = _upstream(up, _FILES)
    spec = AssetsSpec(
        name="fake", git=f"file://{up}", rev=rev,
        files={"model.xml": "robot/model.xml", "meshes": "robot/meshes"},
        sha256={k: hashlib.sha256(v).hexdigest() for k, v in _FILES.items()},
        license={"spdx": "BSD-3-Clause", "file": "LICENSE"},
    )
    mod = _load_script()
    monkeypatch.setattr(mod, "load_embodiment_config", lambda _id: SimpleNamespace(assets=spec))
    monkeypatch.setattr(assets, "STATE_DIR", tmp_path / "state")
    return mod, spec, tmp_path / "state" / "assets" / "fake"


def test_fetch_copies_exactly_the_listed_files_verified_with_the_licence(fake, capsys) -> None:
    mod, spec, dest = fake
    assert mod.main(["fake"]) == 0
    got = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
    assert got == sorted([*_FILES, "LICENSE"])  # the unlisted mesh is not copied
    assert (dest / "LICENSE").read_bytes() == b"BSD 3-Clause"
    assert assets.problems(spec, dest) == {}
    assert "3 files verified" in capsys.readouterr().out


def test_fetch_is_idempotent_and_check_only_verifies(fake, capsys) -> None:
    mod, _spec, dest = fake
    assert mod.main(["fake", "--check"]) == 1
    err = capsys.readouterr().err
    assert "not verified" in err and "fetch them with: python scripts/fetch_assets.py fake" in err
    assert not dest.exists()  # --check wrote nothing
    assert mod.main(["fake"]) == 0
    stamp = {p: p.stat().st_mtime_ns for p in dest.rglob("*")}
    assert mod.main(["fake"]) == 0 and mod.main(["fake", "--check"]) == 0
    assert {p: p.stat().st_mtime_ns for p in dest.rglob("*")} == stamp  # untouched


def test_dest_overrides_the_state_dir(fake, tmp_path) -> None:
    mod, spec, default = fake
    assert mod.main(["fake", "--dest", str(tmp_path / "elsewhere")]) == 0
    assert assets.problems(spec, tmp_path / "elsewhere") == {} and not default.exists()


def test_an_upstream_mismatch_names_the_file_and_writes_nothing(fake, capsys) -> None:
    mod, spec, dest = fake
    spec.sha256["meshes/b.STL"] = "0" * 64  # the pin no longer matches upstream
    assert mod.main(["fake"]) == 1
    err = capsys.readouterr().err
    assert "robot/meshes/b.STL" in err and assets.MISMATCH in err and "nothing was written" in err
    assert not dest.exists()


def test_a_changed_local_file_is_named_and_never_deleted(fake, capsys) -> None:
    mod, _spec, dest = fake
    assert mod.main(["fake"]) == 0
    (dest / "meshes" / "a.STL").write_bytes(b"my edit")
    (dest / "model.xml").unlink()
    assert mod.main(["fake"]) == 1
    assert "meshes/a.STL" in capsys.readouterr().err
    assert (dest / "meshes" / "a.STL").read_bytes() == b"my edit"  # left as the user had it
    assert not (dest / "model.xml").exists()  # and nothing else was written either


def test_weights_prefetch_reads_the_package_table(monkeypatch, capsys) -> None:
    from vector_os_nano.perception import weights

    got = []
    monkeypatch.setattr(weights, "local_dir", lambda w: got.append(w) or f"/cache/{w.repo_id}")
    assert _load_script().main(["--weights"]) == 0
    assert tuple(got) == weights.default_stack()
    monkeypatch.setattr(weights, "local_dir", lambda w: (_ for _ in ()).throw(
        weights.WeightsUnavailable(f"{w} missing")))
    assert _load_script().main(["--weights"]) == 1
    assert "missing" in capsys.readouterr().err


def test_a_robot_that_ships_its_files_has_nothing_to_fetch(capsys) -> None:
    assert _load_script().main(["go2"]) == 0
    assert "nothing to fetch" in capsys.readouterr().out
