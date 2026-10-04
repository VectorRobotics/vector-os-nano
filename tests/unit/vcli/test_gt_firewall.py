# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""The GT firewall: the actor (skills, perception) never reads ground truth.

Ground truth (``sim.ground_truth``) and the oracles over it (``vcli/worlds/
sim_oracles``) are for grading only. A skill or a perception module that imported
them could read the answer it is graded on; a SkillContext that carried GT would
hand it to every skill.
"""
from __future__ import annotations

import ast
from pathlib import Path

_PKG = Path(__file__).resolve().parents[3] / "vector_os_nano"
_FORBIDDEN = ("vector_os_nano.sim.ground_truth", "vector_os_nano.vcli.worlds.sim_oracles")


def _imports(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
            out |= {f"{node.module}.{a.name}" for a in node.names}
        elif isinstance(node, ast.Name) and node.id == "GroundTruth":
            out.add("GroundTruth")
    return out


def test_skills_and_perception_never_import_ground_truth() -> None:
    offenders = []
    for top in ("skills", "perception"):
        for path in sorted((_PKG / top).rglob("*.py")):
            bad = [m for m in _imports(path) if m == "GroundTruth" or m.startswith(_FORBIDDEN)]
            if bad:
                offenders.append(f"{path.relative_to(_PKG)}: {sorted(bad)}")
    assert not offenders, offenders


def test_skill_context_has_no_ground_truth() -> None:
    from types import SimpleNamespace

    from vector_os_nano.core.agent import Agent
    from vector_os_nano.core.skill import SkillContext

    ctx = SkillContext()
    assert not [k for k in vars(ctx) if "gt" in k.lower().split("_") or "ground" in k.lower()]

    agent = Agent(base=SimpleNamespace(name="base"), config={"world": "apartment"})
    built = agent._build_context()
    assert not any("ground" in k.lower() or k == "gt" for k in built.services)
    for value in [*vars(built).values(), *built.services.values()]:
        assert type(value).__name__ not in ("GroundTruth", "SimSession"), type(value)


def test_oracles_never_run_the_actors_models() -> None:
    """g1-oracle-reruns-detector: an oracle that re-ran the detector at verify time
    graded a fresh detection, not the step's output. The GT oracles import no
    perception / detector module at all (in_view is a segmentation render)."""
    for mod in ("vcli/worlds/sim_oracles.py", "sim/ground_truth.py"):
        bad = [m for m in _imports(_PKG / mod)
               if m.startswith(("vector_os_nano.perception", "vector_os_nano.skills"))]
        assert not bad, (mod, bad)
