# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The layering ratchet: interface charter section 6 (docs/plan-release-1.0-interfaces.md).

AST-based. Every violation that exists today is one row of layering_allowlist.txt
(``rule | file | what``). A violation that is not listed is red. The list only shrinks:
``python tests/unit/test_layering.py --prune`` rewrites it as the listed rows that are
still violations (it can never add one), the coordinator runs that at every merge, and
release 1.0 ships the list empty.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import vector_os_nano

PKG = Path(vector_os_nano.__file__).resolve().parent
ALLOW = Path(__file__).with_name("layering_allowlist.txt")
P = "vector_os_nano."

ACTOR = ("skills/", "perception/", "sim/drivers/", "sim/policies/")
KERNEL = ("vcli/engine.py", "vcli/native_loop.py", "vcli/native_tools.py", "vcli/cognitive/")
# Rule 6: a base install has none of these; the package and the CLI must import without them.
HEAVY = ("torch", "transformers", "pybullet", "casadi", "pinocchio", "mcp", "rclpy")


def _imports(tree: ast.AST, rel: str) -> set[str]:
    """Every module a file imports, lazy imports included, relative ones resolved.
    ``from a.b import c`` yields a.b and a.b.c (c may be a submodule)."""
    package = (P + rel[:-3].replace("/", ".")).split(".")[:-1]
    if rel.endswith("/__init__.py"):
        package = (P + rel[: -len("/__init__.py")].replace("/", ".")).split(".")
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = ".".join(package[: len(package) - node.level + 1] + ([base] if base else []))
            out.add(base)
            out.update(f"{base}.{a.name}" for a in node.names)
    return out


def _under(module: str, *roots: str) -> bool:
    return any(module == r or module.startswith(r + ".") for r in roots)


def violations() -> set[str]:
    out: set[str] = set()
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(PKG).as_posix()
        if rel.startswith("third_party/"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        mods = _imports(tree, rel)
        ours = {m[len(P):] for m in mods if m.startswith(P)}
        top = {m.split(".")[0] for m in mods}

        def bad(rule: str, what: str) -> None:
            out.add(f"{rule} | {rel} | {what}")

        if rel.startswith("schemas/"):
            for m in ours - {m for m in ours if _under(m, "schemas")}:
                bad("1 schemas imports nothing from the package", m)
        if rel.startswith("interfaces/"):
            for m in ours - {m for m in ours if _under(m, "schemas", "interfaces")}:
                bad("1 interfaces imports only schemas", m)
        if top & {"pybullet", "pybullet_data"} and not rel.startswith("sim/backends/pybullet/"):
            bad("2 a simulator is imported only by its backend", "pybullet")
        if "mujoco" in top and not rel.startswith(("sim/backends/mujoco/", "sim/scene/")):
            bad("2 a simulator is imported only by its backend", "mujoco")
        if rel.startswith(ACTOR):
            for m in ours:
                if _under(m, "sim.ground_truth", "sim.grasp", "sim.backends", "vcli"):
                    bad("3 actor code imports neither the world side nor the kernel", m)
            for node in ast.walk(tree):
                # ponytail: `reset` is too common a method name for a name check; an actor
                # that cannot reach `world` / `runtime` (below) cannot call their reset.
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ("attach", "detach")):
                    bad("3 actor code calls no attach / detach", node.func.attr)
                if (rel.startswith(("skills/", "perception/")) and isinstance(node, ast.Attribute)
                        and node.attr in ("runtime", "world", "gt", "get_object_positions")):
                    bad("3 skills and perception touch no runtime / world / gt", node.attr)
        if rel.startswith(KERNEL):
            for m in ours:
                concrete_world = _under(m, "vcli.worlds") and m not in ("vcli.worlds", "vcli.worlds.base")
                if concrete_world or _under(m, "embodiments", "hardware", "perception", "sim") or (
                        _under(m, "skills") and m != "skills"):
                    bad("4 the kernel imports no concrete world, body, backend or skill pack", m)
        if rel.startswith("core/"):
            for m in ours - {m for m in ours if _under(m, "core", "interfaces", "schemas")}:
                bad("5 core imports only interfaces and schemas", m)
    return out


def _listed() -> set[str]:
    if not ALLOW.exists():
        return set()
    return {ln.strip() for ln in ALLOW.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")}


def test_no_new_layering_violation():
    new = sorted(violations() - _listed())
    assert not new, "new layering violations (charter section 6); fix them, never list them:\n" + "\n".join(new)


def test_package_and_cli_import_on_a_base_install():
    """Rule 6: with every optional heavy dependency blocked, the package and the CLI import."""
    code = (
        "import sys\n"
        f"HEAVY = {HEAVY!r}\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in HEAVY:\n"
        "            raise ImportError(f'blocked optional dependency: {name}')\n"
        "sys.meta_path.insert(0, Block())\n"
        "import vector_os_nano\n"
        "import vector_os_nano.vcli.cli as cli\n"
        "cli.parse_args([])\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]


if __name__ == "__main__" and sys.argv[1:] == ["--prune"]:
    keep = sorted(_listed() & violations())
    head = [ln for ln in ALLOW.read_text(encoding="utf-8").splitlines() if ln.startswith("#")]
    ALLOW.write_text("\n".join(head + keep) + "\n", encoding="utf-8")
    print(f"{len(keep)} rows left in {ALLOW.name}")
