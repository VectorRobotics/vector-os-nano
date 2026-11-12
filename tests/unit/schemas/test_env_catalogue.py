# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""C1a — schemas/env.py catalogues every environment variable the package reads
(charter section 4, Environment), in both directions: every read and every ``VECTOR_*``
literal is a row, and every row is read somewhere.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import vector_os_nano
from vector_os_nano.schemas.env import CATALOGUE
from vector_os_nano.schemas.user import PROVIDERS

PKG = Path(vector_os_nano.__file__).resolve().parent
KINDS = {"user", "test_seam", "experimental", "internal", "third_party"}
# A VECTOR_* word that is not a variable.
NOT_ENV = {"VECTOR_VERDICT": "the stdout sentinel of the verdict line (charter section 5 output)"}
# Reads whose name is computed, each with where its possible names come from.
DYNAMIC = {
    ("vcli/config.py", "key_env"): "a provider's key variable (schemas/user.py PROVIDERS) or llm.api_key_env",
    ("schemas/env.py", "name"): "env.read itself: its callers are checked like any read",
}
_WORD = re.compile(r"\bVECTOR_[A-Z0-9_]+\b")


def _is_environ(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "environ"


def _name_of(arg: ast.AST, consts: dict[str, str]) -> str | None:
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    if isinstance(arg, ast.Name):
        return consts.get(arg.id)
    return None


def scan(tree: ast.AST) -> tuple[set[str], set[str], set[str]]:
    """(names read, dynamic read expressions, VECTOR_* words in string literals)."""
    consts = {t.id: n.value.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
              and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
              for t in n.targets if isinstance(t, ast.Name)}
    reads: set[str] = set()
    dynamic: set[str] = set()
    for node in ast.walk(tree):
        arg = None
        if isinstance(node, ast.Call) and node.args:
            f = node.func
            if isinstance(f, ast.Attribute) and (
                    (f.attr in ("get", "setdefault", "pop") and _is_environ(f.value))
                    or f.attr == "getenv"
                    or (f.attr == "read" and isinstance(f.value, ast.Name) and f.value.id == "env")):
                arg = node.args[0]
        elif isinstance(node, ast.Subscript) and _is_environ(node.value) and isinstance(node.ctx, ast.Load):
            arg = node.slice
        elif isinstance(node, ast.Compare) and any(_is_environ(c) for c in node.comparators):
            arg = node.left
        if arg is not None:
            name = _name_of(arg, consts)
            if name is None:
                dynamic.add(ast.unparse(arg))
            else:
                reads.add(name)
    words = {w for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
             for w in _WORD.findall(n.value)}
    return reads, dynamic, words


def _package() -> tuple[set[str], set[tuple[str, str]], dict[str, set[str]]]:
    reads: set[str] = set()
    dynamic: set[tuple[str, str]] = set()
    words: dict[str, set[str]] = {}
    for path in PKG.rglob("*.py"):
        if "third_party" in path.parts:
            continue
        rel = path.relative_to(PKG).as_posix()
        r, d, w = scan(ast.parse(path.read_text(encoding="utf-8")))
        reads |= r
        dynamic |= {(rel, e) for e in d}
        for word in w:
            words.setdefault(word, set()).add(rel)
    return reads, dynamic, words


def test_every_row_is_well_formed() -> None:
    for name, var in CATALOGUE.items():
        assert var.kind in KINDS and var.meaning, name
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*", name), name


def test_every_read_and_every_vector_word_is_catalogued() -> None:
    reads, dynamic, words = _package()
    assert not reads - CATALOGUE.keys(), f"read but not in schemas/env.py: {sorted(reads - CATALOGUE.keys())}"
    unlisted = {w: sorted(where) for w, where in words.items() if w not in CATALOGUE and w not in NOT_ENV}
    assert not unlisted, f"VECTOR_* names not in schemas/env.py: {unlisted}"
    assert dynamic <= DYNAMIC.keys(), f"computed reads nobody accounted for: {sorted(dynamic - DYNAMIC.keys())}"
    assert {p.key_env for p in PROVIDERS.values()} <= CATALOGUE.keys()


def test_every_row_is_read_somewhere() -> None:
    reads, dynamic, _words = _package()
    read = reads | {p.key_env for p in PROVIDERS.values()}
    assert not CATALOGUE.keys() - read, f"stale rows (nothing reads them): {sorted(CATALOGUE.keys() - read)}"
    assert dynamic == DYNAMIC.keys(), "a DYNAMIC row whose read is gone is stale"
    assert not NOT_ENV.keys() & CATALOGUE.keys()


@pytest.mark.parametrize(("src", "read", "computed"), [
    ("import os\nos.environ.get('VECTOR_X')", {"VECTOR_X"}, set()),
    ("import os\nos.getenv('MY_VAR', '')", {"MY_VAR"}, set()),
    ("import os\nos.environ['A_B']", {"A_B"}, set()),
    ("import os\n'TOKEN' in os.environ", {"TOKEN"}, set()),
    ("import os\nK = 'VECTOR_K'\nos.environ.get(K)", {"VECTOR_K"}, set()),
    ("import os\nos.environ.get(name)", set(), {"name"}),
    ("import os\nos.environ.setdefault('S', 'v')", {"S"}, set()),
    ("from vector_os_nano.schemas import env\nenv.read('VECTOR_HOME')", {"VECTOR_HOME"}, set()),
])
def test_the_scan_sees_every_spelling_of_a_read(src: str, read: set[str], computed: set[str]) -> None:
    """The negative cases: an uncatalogued read in any of these spellings is found."""
    reads, dynamic, _ = scan(ast.parse(src))
    assert (reads, dynamic) == (read, computed)
    assert _WORD.findall("set VECTOR_NEW_THING=1 to ...") == ["VECTOR_NEW_THING"]
