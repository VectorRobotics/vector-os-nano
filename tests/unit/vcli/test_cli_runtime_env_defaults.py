# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Rule 11: bare `vector-cli` self-sufficiency — the CLI fills the sim/runtime env
the user would otherwise have to export, so they only ever type `vector-cli`.

The defaults are applied in main() (never in the runtimes), so the test suite —
which does not run through main() — is unaffected. Here we unit-test the pure
decision helper directly.
"""
from __future__ import annotations

from vector_os_nano.vcli.cli import _apply_runtime_env_defaults


def test_photoreal_on_when_blender_present_and_unset():
    env: dict = {}
    _apply_runtime_env_defaults(env, blender_present=True, has_display=True)
    assert env["VECTOR_G1_PHOTOREAL"] == "1"
    assert env["VECTOR_GO2_PHOTOREAL"] == "1"
    assert "MUJOCO_GL" not in env          # display present -> leave GL default


def test_headless_sets_egl():
    env: dict = {}
    _apply_runtime_env_defaults(env, blender_present=True, has_display=False)
    assert env["MUJOCO_GL"] == "egl"       # offscreen backend needed when headless


def test_explicit_photoreal_off_is_respected():
    env: dict = {"VECTOR_G1_PHOTOREAL": "0"}
    _apply_runtime_env_defaults(env, blender_present=True, has_display=True)
    assert env["VECTOR_G1_PHOTOREAL"] == "0"     # user disable not overridden
    assert "VECTOR_GO2_PHOTOREAL" not in env      # and not force-set on the other


def test_explicit_photoreal_on_left_as_is():
    env: dict = {"VECTOR_GO2_PHOTOREAL": "1"}
    _apply_runtime_env_defaults(env, blender_present=True, has_display=True)
    assert env["VECTOR_GO2_PHOTOREAL"] == "1"
    assert "VECTOR_G1_PHOTOREAL" not in env       # one flag set -> don't touch


def test_no_blender_no_photoreal():
    env: dict = {}
    _apply_runtime_env_defaults(env, blender_present=False, has_display=True)
    assert "VECTOR_G1_PHOTOREAL" not in env
    assert "VECTOR_GO2_PHOTOREAL" not in env


def test_existing_mujoco_gl_untouched():
    env: dict = {"MUJOCO_GL": "glfw"}
    _apply_runtime_env_defaults(env, blender_present=False, has_display=False)
    assert env["MUJOCO_GL"] == "glfw"            # never override an explicit choice
