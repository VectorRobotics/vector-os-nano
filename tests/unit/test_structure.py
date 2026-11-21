# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The structure ratchet: acceptance A11 of docs/plan-release-1.0.md (a concise code base).

What is wrong today is one row of structure_allowlist.txt (``rule | path``). Anything
not listed is red. The list only shrinks: ``python tests/unit/test_structure.py --prune``
drops the rows that are fixed (it can never add one), the coordinator runs that at every
merge, and release 1.0 ships the list empty.

  over-800     a package source file longer than 800 lines
  unreachable  a package module no entry point reaches through imports
  target-tree  a tracked path the charter's target tree does not have
"""
from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

import vector_os_nano

PKG = Path(vector_os_nano.__file__).resolve().parent
ROOT = PKG.parent
ALLOW = Path(__file__).with_name("structure_allowlist.txt")
P = "vector_os_nano"

# The whole package, vendored code included, must end below what the campaign started with
# (53343 lines at 3862003; the second backend and the scene compiler added 1.5k before the
# legacy producer goes). The coordinator sets this at every merge to the merged count plus 500:
# a work package may grow the package by that much while it replaces something, never more.
LINE_CEILING = 37_107 + 500

# Charter section 6, the target tree.
PACKAGE_TREE = {"__init__.py", "version.py", "interfaces", "schemas", "core", "sim", "embodiments",
                "worlds", "skills", "perception", "vcli", "mcp", "hardware", "third_party"}
HARDWARE_TREE = {"__init__.py", "so101"}
REPO_TREE = {".github", ".gitignore", ".env.example", "CLAUDE.md", "CHANGELOG.md", "LICENSE", "NOTICE",
             "README.md", "conftest.py", "docs", "examples", "images", "pyproject.toml", "scripts", "tests", P}


def _sources() -> dict[str, Path]:
    """dotted module name -> file, for every package module (vendored code excluded)."""
    out = {}
    for path in PKG.rglob("*.py"):
        rel = path.relative_to(ROOT).with_suffix("").parts
        if "third_party" in rel:
            continue
        out[".".join(rel[:-1] if rel[-1] == "__init__" else rel)] = path
    return out


def _edges(path: Path, name: str, modules: dict[str, Path]) -> set[str]:
    """The package modules *name* reaches: imports (lazy ones too, relative ones resolved)
    and string constants that spell a module (tables that load by name)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = name.split(".") if path.name == "__init__.py" else name.split(".")[:-1]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = ".".join(package[: len(package) - node.level + 1] + ([base] if base else []))
            found.add(base)
            found.update(f"{base}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(P + "."):
            found.add(node.value.split(":")[0])
    out = set()
    for m in found:
        parts = m.split(".")
        out.update(".".join(parts[:i]) for i in range(1, len(parts) + 1))  # a module's parents import too
    return out & modules.keys()


def _unreachable() -> set[str]:
    modules = _sources()
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"].get("scripts", {})
    todo = [P] + [s.split(":")[0] for s in scripts.values()] + [m for m in modules if m.endswith(".__main__")]
    seen: set[str] = set()
    while todo:
        m = todo.pop()
        if m in seen or m not in modules:
            continue
        seen.add(m)
        todo.extend(_edges(modules[m], m, modules))
    return {modules[m].relative_to(ROOT).as_posix() for m in modules.keys() - seen}


def _tracked_top(prefix: str = "") -> set[str]:
    done = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--", prefix or "."],
                          capture_output=True, text=True, timeout=60)
    if done.returncode != 0:
        return set()  # not a git checkout (an installed wheel): nothing to say about the tree
    return {ln[len(prefix):].split("/")[0] for ln in done.stdout.splitlines() if ln.startswith(prefix)}


def violations() -> set[str]:
    out = {f"unreachable | {p}" for p in _unreachable()}
    for path in PKG.rglob("*.py"):
        if "third_party" not in path.parts and len(path.read_text(encoding="utf-8").splitlines()) > 800:
            out.add(f"over-800 | {path.relative_to(ROOT).as_posix()}")
    out |= {f"target-tree | {name}" for name in _tracked_top() - REPO_TREE}
    out |= {f"target-tree | {P}/{name}" for name in _tracked_top(P + "/") - PACKAGE_TREE}
    out |= {f"target-tree | {P}/hardware/{name}" for name in _tracked_top(P + "/hardware/") - HARDWARE_TREE}
    return out


def _listed() -> set[str]:
    if not ALLOW.exists():
        return set()
    return {ln.strip() for ln in ALLOW.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")}


def test_no_new_structure_violation():
    new = sorted(violations() - _listed())
    assert not new, "new structure violations (acceptance A11); fix them, never list them:\n" + "\n".join(new)


def test_the_package_does_not_grow():
    lines = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in PKG.rglob("*.py"))
    assert lines <= LINE_CEILING, f"vector_os_nano/ has {lines} Python lines; the ceiling is {LINE_CEILING}"


if __name__ == "__main__" and sys.argv[1:] == ["--prune"]:
    keep = sorted(_listed() & violations())
    head = [ln for ln in ALLOW.read_text(encoding="utf-8").splitlines() if ln.startswith("#")]
    ALLOW.write_text("\n".join(head + keep) + "\n", encoding="utf-8")
    print(f"{len(keep)} rows left in {ALLOW.name}")
