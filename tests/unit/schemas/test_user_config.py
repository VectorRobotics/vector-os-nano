# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""C1a — ``$VECTOR_HOME/config.yaml`` (charter section 4): a strict, fail-loud loader.

Every error names the file and the key; a secret-looking key is refused with the
variable to set instead (its value never echoed); every typed field has a reader.
"""
from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

import vector_os_nano
from vector_os_nano.schemas import user

PKG = Path(vector_os_nano.__file__).resolve().parent


def _file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_a_valid_file_and_an_absent_one(tmp_path) -> None:
    assert user.load(tmp_path / "absent.yaml") == user.UserConfig()
    got = user.load(_file(tmp_path, "schema: 1\nllm:\n  provider: openai_compat\n  model: qwen\n"
                                    "  base_url: http://127.0.0.1:8000/v1\n  api_key_env: MY_KEY\n"))
    assert got.llm == user.Llm("openai_compat", "qwen", "http://127.0.0.1:8000/v1", "MY_KEY")
    assert user.load(_file(tmp_path, "schema: 1\n")) == user.UserConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        got.llm.model = "x"  # type: ignore[misc]


@pytest.mark.parametrize(("text", "key", "says"), [
    ("schema: 1\nperception: {detector: owlv2}\n", "perception", "valid: schema, llm"),
    ("schema: 1\nllm:\n  provider: deepseek\n  temperature: '0.2'\n", "llm.temperature",
     "valid: provider, model, base_url, api_key_env"),
    ("schema: 2\nllm: {}\n", "schema 2", "upgrade vector_os_nano"),
    ("schema: 0\n", "schema 0", "use schema: 1"),
    ("llm:\n  provider: deepseek\n", "schema: 1 is required", "None"),
    ("schema: '1'\n", "schema: 1 is required", "'1'"),
    ("schema: 1\nllm:\n  model: 7\n", "llm.model", "must be a string (got int)"),
    ("schema: 1\nllm: [deepseek]\n", "llm must be a mapping", "list"),
    ("schema: 1\nllm:\n  provider: claude\n", "llm.provider 'claude'", "deepseek, openrouter, anthropic, openai_compat"),
    ("- schema: 1\n", "a mapping with schema: 1", ""),
    ("schema: 1\nllm: {provider: [\n", "unreadable", "Error"),
])
def test_every_error_names_the_file_and_the_key(tmp_path, text, key, says) -> None:
    path = _file(tmp_path, text)
    with pytest.raises(user.UserConfigError) as err:
        user.load(path)
    msg = str(err.value)
    assert msg.startswith(f"{path}: ") and key in msg and says in msg, msg


@pytest.mark.parametrize(("text", "var"), [
    ("schema: 1\nanthropic_api_key: {s}\n", "ANTHROPIC_API_KEY"),
    ("schema: 1\nllm:\n  api_key: {s}\n", "DEEPSEEK_API_KEY"),  # the default provider's
    ("schema: 1\nllm:\n  provider: openrouter\n  api_key: {s}\n", "OPENROUTER_API_KEY"),
    ("schema: 1\nllm:\n  api_key_env: MY_KEY\n  token: {s}\n", "MY_KEY"),
    ("schema: 1\nllm:\n  provider: anthropic\n  password: {s}\n", "ANTHROPIC_API_KEY"),
    ("schema: 1\nhf_token: {s}\n", "HF_TOKEN"),
    ("llm:\n  api_key: {s}\n", "DEEPSEEK_API_KEY"),  # refused before the schema is even checked
])
def test_a_secret_looking_key_is_refused_naming_the_variable(tmp_path, text, var) -> None:
    secret = "sk-SECRET-VALUE-123"
    path = _file(tmp_path, text.format(s=secret))
    with pytest.raises(user.UserConfigError, match=f"looks like a secret.*set {var} in the environment") as err:
        user.load(path)
    assert str(path) in str(err.value) and secret not in str(err.value)


@pytest.mark.parametrize(("text", "key"), [
    ("schema: 1\nschema: 1\n", "schema"),
    ("schema: 2\nschema: 1\n", "schema"),
    ("schema: 1\nllm:\n  provider: anthropic\n  provider: deepseek\n", "provider"),
    ("schema: 1\nllm:\n  provider: anthropic\n  api_key: sk-SHADOWED\nllm:\n  provider: anthropic\n", "llm"),
])
def test_a_repeated_key_is_refused_never_silently_shadowed(tmp_path, text, key) -> None:
    """C1a2-V-D1: yaml keeps the LAST of two equal keys, so a block shadowed by a later one
    (and a secret inside it) was never seen by the checks."""
    path = _file(tmp_path, text)
    with pytest.raises(user.UserConfigError) as err:
        user.load(path)
    msg = str(err.value)
    assert msg.startswith(f"{path}: ") and f"duplicate key {key}" in msg and "sk-SHADOWED" not in msg, msg


@pytest.mark.parametrize("key", ["api_key_env", "model", "base_url", "provider"])
def test_a_key_that_names_a_variable_is_not_a_secret(key) -> None:
    assert not user._SECRET.search(key)


def test_every_typed_field_has_a_reader_outside_schemas() -> None:
    """No decorative keys (charter section 4): each field of the schema is read as an
    attribute by a module that imports schemas.user."""
    read: set[str] = set()
    for path in PKG.rglob("*.py"):
        if "schemas" in path.parts or "third_party" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        if not any(isinstance(n, ast.ImportFrom) and n.module == "vector_os_nano.schemas"
                   and any(a.name == "user" for a in n.names) for n in ast.walk(tree)):
            continue
        read |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for cls in (user.UserConfig, user.Llm):
        for f in dataclasses.fields(cls):
            assert f.name in read, f"{cls.__name__}.{f.name} has no reader outside schemas/"


@pytest.mark.parametrize('value', ['[]', 'null', '{gui: 1}', '{gui: "false"}', '{backend: ""}', '{world: false}', '{embodiment: 2}', '{unknown: x}'])
def test_invalid_sim_defaults_are_rejected(tmp_path, value):
    with pytest.raises(user.UserConfigError, match='defaults'):
        user.load(_file(tmp_path, f'schema: 1\ndefaults: {value}\n'))


def test_sim_defaults_have_explicit_types(tmp_path):
    cfg = user.load(_file(tmp_path, 'schema: 1\ndefaults: {embodiment: piper, backend: pybullet, world: workbench, gui: false}\n'))
    assert cfg.defaults == user.Defaults('piper', 'pybullet', 'workbench', False)
