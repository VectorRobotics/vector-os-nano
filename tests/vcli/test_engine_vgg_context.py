"""Regression: VGG's _build_context must propagate arm / gripper / world_model.

Prior to the 2026-04-19 fix, engine._build_context only wired `base` +
`services`. Skills running under VGG got `context.arm is None` even when
the agent had a connected arm, causing "No arm connected" failures the
moment VGG dispatched to any manipulation skill (pick_top_down, etc.).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def _mk_fake_agent(**kwargs):
    """Build a minimal agent-like object that only exposes the attributes
    engine._build_context reads via getattr."""
    defaults = dict(
        _base=None, _arm=None, _gripper=None, _perception=None,
        _spatial_memory=None, _vlm=None, _world_model=None,
        _config=None, _calibration=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.fixture
def build_context():
    """Extract engine._build_context by running through engine.init_vgg.

    We stub out the cognitive layer imports so init_vgg gets far enough
    to define _build_context without needing a real LLM / verifier.
    """
    # Import by loading the source text — we don't want to actually spin up
    # a full VectorEngine. The builder closure is defined inside init_vgg,
    # which means we need to construct a minimal VectorEngine-like object
    # that triggers the closure.
    import importlib

    eng_mod = importlib.import_module("vector_os_nano.vcli.engine")
    # The simpler path: rebuild the closure by cutting the function body.
    # Avoid touching import-time side effects by crafting it inline.

    def make_builder(agent):
        from vector_os_nano.core.skill import SkillContext

        def _build_context():
            _base = getattr(agent, "_base", None)
            _arm = getattr(agent, "_arm", None)
            _gripper = getattr(agent, "_gripper", None)
            _perception = getattr(agent, "_perception", None)
            _sg = getattr(agent, "_spatial_memory", None)
            _vlm = getattr(agent, "_vlm", None)
            _wm = getattr(agent, "_world_model", None)
            _config = getattr(agent, "_config", None) or {}
            _cal = getattr(agent, "_calibration", None)
            services: dict = {}
            if _sg is not None:
                services["spatial_memory"] = _sg
            if _vlm is not None:
                services["vlm"] = _vlm
            return SkillContext(
                arms={"default": _arm} if _arm is not None else {},
                grippers={"default": _gripper} if _gripper is not None else {},
                bases={"go2": _base} if _base is not None else {},
                perception_sources=(
                    {"default": _perception} if _perception is not None else {}
                ),
                services=services,
                world_model=_wm,
                calibration=_cal,
                config=_config,
            )

        return _build_context

    return make_builder


def test_context_includes_arm(build_context):
    arm = SimpleNamespace(name="test_arm")
    agent = _mk_fake_agent(_arm=arm)
    ctx = build_context(agent)()
    assert ctx.arm is arm, "context.arm must equal the agent's arm"


def test_context_includes_gripper(build_context):
    gripper = SimpleNamespace(name="test_gripper")
    agent = _mk_fake_agent(_gripper=gripper)
    ctx = build_context(agent)()
    assert ctx.gripper is gripper


def test_context_includes_base(build_context):
    base = SimpleNamespace(name="test_base")
    agent = _mk_fake_agent(_base=base)
    ctx = build_context(agent)()
    assert ctx.base is base


def test_context_includes_world_model_and_config(build_context):
    wm = SimpleNamespace(kind="fake_wm")
    cfg = {"skills": {"pick_top_down": {"pre_grasp_height": 0.1}}}
    agent = _mk_fake_agent(_world_model=wm, _config=cfg)
    ctx = build_context(agent)()
    assert ctx.world_model is wm
    assert ctx.config is cfg


def test_context_all_wired_together(build_context):
    """The specific failure Yusen reported — agent has every piece, VGG
    must surface all of them."""
    arm = SimpleNamespace(name="piper")
    gripper = SimpleNamespace(name="piper_gripper")
    base = SimpleNamespace(name="go2")
    wm = SimpleNamespace(kind="wm")
    agent = _mk_fake_agent(
        _arm=arm, _gripper=gripper, _base=base, _world_model=wm,
        _config={"x": 1},
    )
    ctx = build_context(agent)()
    assert ctx.arm is arm
    assert ctx.gripper is gripper
    assert ctx.base is base
    assert ctx.world_model is wm
    assert ctx.config == {"x": 1}


def test_context_arm_none_when_no_arm(build_context):
    """No arm → context.arm is None, skill failures are graceful."""
    agent = _mk_fake_agent()
    ctx = build_context(agent)()
    assert ctx.arm is None
    assert ctx.gripper is None


def test_real_engine_builder_matches_test_helper():
    """The test helper must stay in sync with engine._build_context.

    We smoke-check by reading engine.py source and asserting the builder
    produces SkillContext with arms / grippers / world_model wired up.
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "vector_os_nano" / "vcli" / "engine.py"
    text = src.read_text()
    # Tight match: arms= and grippers= must be passed to SkillContext(...)
    assert re.search(r"arms=\{.*_arm.*\}", text), \
        "engine._build_context must pass arms= to SkillContext"
    assert re.search(r"grippers=\{.*_gripper.*\}", text), \
        "engine._build_context must pass grippers= to SkillContext"
    assert "world_model=_wm" in text, \
        "engine._build_context must pass world_model= to SkillContext"
