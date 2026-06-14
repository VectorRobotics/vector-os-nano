# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Scene-reference resolution for habitat scenarios (ADR-009).

Scene assets are EXTERNAL runtime data (never vendored into git — ADR-008).
A scenario carries a backend-generic ``scene_ref``; this module resolves it
to a real file under the data root:

- absolute refs pass through (power users / tests),
- relative refs resolve against ``VECTOR_HABITAT_DATA`` (default: the spike
  download location), fail-loud with the searched path when missing.

N0 (campaign #2) adds DATASET scenes: when a scenario carries a
``scene_dataset_config``, its ``scene_ref`` is a scene-instance NAME (e.g.
``"apt_1"``) habitat resolves through the dataset config (stage + furniture
+ lighting composed), not a file path. Bare ``habitat_sim`` does NOT consume
the dataset's ``navmesh_instances`` mapping (habitat-lab machinery;
spike-verified on ReplicaCAD — the sim boots with no navmesh), so
``dataset_navmesh_path`` resolves the authored navmesh here and the server
gets it as an explicit ``--navmesh``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_DEFAULT_DATA_ROOT = (
    Path.home() / "sandbox" / "habitat-spike" / "data" / "scene_datasets"
)


def habitat_data_root() -> Path:
    """Directory habitat scene refs resolve against (env-overridable)."""
    return Path(os.environ.get("VECTOR_HABITAT_DATA", str(_DEFAULT_DATA_ROOT)))


def resolve_scene_ref(scene_ref: str) -> str:
    """Resolve a scenario ``scene_ref`` to an existing scene file, fail-loud."""
    if not scene_ref:
        raise FileNotFoundError("scenario has an empty scene_ref")
    p = Path(scene_ref)
    if not p.is_absolute():
        p = habitat_data_root() / p
    if not p.exists():
        raise FileNotFoundError(
            f"habitat scene not found: {p} — set VECTOR_HABITAT_DATA or download "
            f"the dataset (license-free test scenes: habitat_test_scenes)"
        )
    return str(p)


def resolve_scene_dataset_config(ref: str) -> str:
    """Resolve a scenario ``scene_dataset_config`` to an existing file, fail-loud."""
    if not ref:
        raise FileNotFoundError("scenario has an empty scene_dataset_config")
    p = Path(ref)
    if not p.is_absolute():
        p = habitat_data_root() / p
    if not p.exists():
        raise FileNotFoundError(
            f"habitat scene dataset config not found: {p} — set "
            f"VECTOR_HABITAT_DATA or download the dataset (ReplicaCAD, "
            f"CC-BY-4.0: python -m habitat_sim.utils.datasets_download "
            f"--uids replica_cad_dataset)"
        )
    return str(p)


def preflight_scene_instance(dataset_config: str, scene_name: str) -> str:
    """Check ``scene_name`` exists in the dataset's conventional scene dir.

    Datasets following the habitat convention keep scene instances under
    ``configs/scenes/<name>.scene_instance.json`` next to the dataset config
    (ReplicaCAD does). When that directory exists we fail loud with the
    available set — a typo'd scene name would otherwise die INSIDE the conda
    subprocess before the PORT handshake (whose stderr is swallowed).
    Datasets with a different layout skip the check (the server judges).
    """
    if not scene_name:
        raise FileNotFoundError("scenario has an empty scene_ref (dataset scene name)")
    scenes_dir = Path(dataset_config).parent / "configs" / "scenes"
    if not scenes_dir.is_dir():
        return scene_name
    if (scenes_dir / f"{scene_name}.scene_instance.json").exists():
        return scene_name
    available = sorted(
        p.name.split(".")[0] for p in scenes_dir.glob("*.scene_instance.json")
    )
    shown = ", ".join(available[:12]) + (" ..." if len(available) > 12 else "")
    raise FileNotFoundError(
        f"scene instance {scene_name!r} not in dataset {dataset_config}; "
        f"available: {shown or '<none>'}"
    )


def dataset_navmesh_path(dataset_config: str, scene_name: str) -> str:
    """Authored navmesh for ``scene_name`` from the dataset config, or ``""``.

    Resolved repo-side because bare ``habitat_sim`` ignores the dataset's
    ``navmesh_instances`` mapping (spike-verified). Missing/unmapped returns
    ``""`` — the server then falls back to ``recompute_navmesh`` (an authored
    navmesh is better: it was baked with the furniture as static obstacles).
    """
    try:
        data = json.loads(Path(dataset_config).read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    rel = (data.get("navmesh_instances") or {}).get(scene_name, "")
    if not rel:
        return ""
    p = Path(dataset_config).parent / rel
    return str(p) if p.exists() else ""
