# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""C0 — robot assets by manifest: the g1 `assets:` block and the ONE resolver.

Hermetic: the state dir is pointed at a temp directory; no network, no MuJoCo.
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import pytest

from vector_os_nano.embodiments import assets
from vector_os_nano.schemas.robot import (
    EmbodimentConfigError,
    load_embodiment_config,
    parse_embodiment_config,
)

FETCH = "python scripts/fetch_assets.py g1"


@pytest.fixture
def state_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("VECTOR_HOME", str(tmp_path / "state"))
    return tmp_path / "state"


def _fake_assets(folder: Path, spec, *, corrupt: str | None = None) -> None:
    """Files whose sha256 we control: rewrite the spec digests to match them."""
    for rel in spec.sha256:
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_bytes(rel.encode())
        spec.sha256[rel] = hashlib.sha256(rel.encode()).hexdigest()
    if corrupt:
        (folder / corrupt).write_bytes(b"tampered")


def test_g1_manifest_declares_a_pinned_fetch_recipe() -> None:
    cfg = load_embodiment_config("g1")
    a = cfg.assets
    assert a is not None and a.name == "g1_gait" and len(a.rev) == 40
    assert a.git == "https://github.com/unitreerobotics/unitree_rl_gym.git"
    assert a.license == {"spdx": "BSD-3-Clause", "file": "LICENSE"}
    # model and policy are relative to the asset folder, and every one is pinned
    assert cfg.model.path == "g1_12dof.xml" and cfg.base.policy.params["file"] == "motion.pt"
    for rel in (cfg.model.path, cfg.base.policy.params["file"], "meshes/pelvis.STL"):
        assert len(a.sha256[rel]) == 64
    assert a.upstream("meshes/pelvis.STL") == "resources/robots/g1_description/meshes/pelvis.STL"
    assert a.upstream("motion.pt") == "deploy/pre_train/g1/motion.pt"


def test_every_mesh_the_g1_model_references_is_pinned(g1_assets) -> None:
    """The asset list is exactly what the robot loads, so a fetch can never miss a mesh."""
    import re

    pinned = {k for k in load_embodiment_config("g1").assets.sha256 if k.startswith("meshes/")}
    xml = (g1_assets / "g1_12dof.xml").read_text()
    assert {f"meshes/{m}" for m in re.findall(r'file="([^"]+)"', xml)} == pinned


def test_absent_assets_fail_with_one_sentence_naming_the_fetch_command(state_dir) -> None:
    with pytest.raises(assets.AssetsMissing) as exc:
        assets.asset_dir(load_embodiment_config("g1"))
    msg = str(exc.value)
    assert msg.endswith(f"fetch them with: {FETCH}") and msg.count(". ") == 0
    assert "motion.pt missing" in msg
    # the sentence reaches the LLM provider: it names no machine path
    assert str(state_dir) not in msg and str(Path.home()) not in msg
    assert not [w for w in msg.split() if w.lstrip("(").startswith("/")], msg


def test_provisioned_assets_resolve_and_a_tampered_file_is_refused(state_dir) -> None:
    cfg = load_embodiment_config("g1")
    folder = state_dir / "assets" / "g1_gait"
    _fake_assets(folder, cfg.assets)
    assert assets.asset_dir(cfg) == folder
    (folder / "meshes" / "pelvis.STL").write_bytes(b"tampered")
    with pytest.raises(assets.AssetsMissing, match="meshes/pelvis.STL does not match its pinned sha256"):
        assets.asset_dir(cfg)


def test_files_shipped_in_the_robot_folder_win(tmp_path, state_dir, monkeypatch) -> None:
    cfg = load_embodiment_config("g1")
    own = tmp_path / "embodiments" / "g1"
    _fake_assets(own, cfg.assets)
    monkeypatch.setattr(assets, "_EMBODIMENTS_DIR", own.parent)
    assert assets.asset_dir(cfg) == own


def test_a_robot_without_an_assets_block_resolves_to_its_own_folder(state_dir) -> None:
    assert assets.asset_dir(load_embodiment_config("go2")).name == "go2"


@pytest.mark.parametrize("patch, error", [
    ({"rev": "276801e"}, "full 40-hex commit"),
    ({"sha256": {"other.pt": "0" * 64}}, "'other.pt' lies under no files entry"),
    ({"files": {"motion.pt": "a/motion.pt", "extra": "b"}}, "'extra' covers no file listed in sha256"),
    ({"license": None}, "license: required"),
])
def test_a_malformed_assets_block_fails_loud(patch, error) -> None:
    block = {"name": "x", "git": "https://example.invalid/x.git", "rev": "a" * 40,
             "files": {"motion.pt": "a/motion.pt"}, "sha256": {"motion.pt": "0" * 64},
             "license": {"spdx": "MIT", "file": "LICENSE"}, **patch}
    raw = {"id": "x", "display_name": "x", "model": {"path": "m.xml", "root_body": "b"},
           "assets": block}
    with pytest.raises(EmbodimentConfigError, match=error):
        parse_embodiment_config(raw)


def test_start_simulation_tells_the_model_the_fetch_command(state_dir) -> None:
    """validate() refuses before anything starts (or anything running is stopped)."""
    from vector_os_nano.sim.session import SimSession, SimSpec, SimStartError
    from vector_os_nano.vcli.tools.base import ToolContext
    from vector_os_nano.vcli.tools.sim_tool import SimStartTool

    with pytest.raises(SimStartError, match="assets not provisioned") as exc:
        SimSession.validate(SimSpec("g1", gui=False))
    assert FETCH in str(exc.value)
    ctx = ToolContext(agent=None, cwd=Path.cwd(), session=None, permissions=None,
                      abort=threading.Event(), app_state={})
    res = SimStartTool().execute({"embodiment": "g1", "gui": False}, ctx)
    assert res.is_error and FETCH in res.content


def test_the_driver_refuses_without_assets(state_dir) -> None:
    pytest.importorskip("mujoco")
    from vector_os_nano.sim.scene.compose import compose
    from vector_os_nano.schemas.world import load_world

    with pytest.raises(assets.AssetsMissing, match=FETCH):
        compose(load_embodiment_config("g1"), load_world("apartment"))
