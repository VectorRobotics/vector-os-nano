# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Every environment variable the package reads (charter section 4, Environment).

kind: ``user`` (a setting a user may export), ``test_seam`` (the suite's replay and
grading hooks), ``experimental`` (opt-in, not part of the 1.0 contract), ``internal``
(the package sets it for its own child process), ``third_party`` (owned by another
tool; read, never set). tests/unit/test_env_catalogue.py holds the package to this
table in both directions: every read is listed and every row is read.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvVar:
    kind: str
    default: str
    meaning: str


CATALOGUE: dict[str, EnvVar] = {
    "VECTOR_HOME": EnvVar("user", "~/.vector", "the state directory (config.yaml, sessions, memory, assets, ...)"),
    "DEEPSEEK_API_KEY": EnvVar("user", "", "key of the deepseek LLM provider"),
    "OPENROUTER_API_KEY": EnvVar("user", "", "key of the openrouter LLM provider, the scene VLM and the LLM fallback"),
    "ANTHROPIC_API_KEY": EnvVar("user", "", "key of the anthropic LLM provider"),
    "OPENAI_API_KEY": EnvVar("user", "", "key of the openai_compat LLM provider"),
    "MOONDREAM_API_KEY": EnvVar("user", "", "key of the Moondream cloud VLM"),
    "MOONDREAM_MODEL": EnvVar("user", "vikhyatk/moondream2", "the Moondream model id"),
    "VECTOR_VLM_URL": EnvVar("user", "", "a local OpenAI-compatible server (e.g. Ollama) for the scene VLM"),
    "VECTOR_VLM_MODEL": EnvVar("user", "gemma4:e4b", "the model that local VLM server serves"),
    "VECTOR_RYNNBRAIN_URL": EnvVar("user", "http://127.0.0.1:8786", "the RynnBrain detector server"),
    "VECTOR_NAV_STACK": EnvVar("user", "", "set: go2 explore launches the external TARE/FAR nav stack"),
    "VECTOR_RUN_DIR": EnvVar("user", "/tmp", "where the cross-process flag files and nav logs of one sim live"),
    "VECTOR_LLM_FALLBACK": EnvVar("experimental", "", "'openrouter': a failing LLM call is retried there"),
    "VECTOR_LLM_FALLBACK_MODEL": EnvVar("experimental", "anthropic/claude-sonnet-4-6", "the fallback's model"),
    "VECTOR_SIM_LOCKSTEP": EnvVar("test_seam", "", "'1': physics steps only when the caller steps it"),
    "VECTOR_GT_DUMP": EnvVar("test_seam", "", "a file: one ground-truth line per turn for scripts/gt_grade.py"),
    "VECTOR_ROLLOUT_DIR": EnvVar("experimental", "", "a directory: MuJoCo records every actual step, solver contact loads and requested sensor frames"),
    "VECTOR_FAKE_LLM_TOOLS": EnvVar("test_seam", "", "a JSON script that replaces the network LLM"),
    "VECTOR_FAKE_LLM_LOG": EnvVar("test_seam", "", "a file the script LLM logs every call to"),
    "VECTOR_REEXEC": EnvVar("internal", "", "'1' on the mjpython re-exec (macOS), so it happens once"),
    "HF_HUB_OFFLINE": EnvVar("third_party", "", "Hugging Face: '1' forbids fetching model weights"),
    "CUDA_VISIBLE_DEVICES": EnvVar("third_party", "", "the GPUs torch sees (the perception note names the device)"),
    "CUDA_CACHE_PATH": EnvVar("user", "", "CUDA JIT cache; empty uses VECTOR_HOME/cache/cuda before simulator startup"),
    "MUJOCO_GL": EnvVar("third_party", "", "MuJoCo's GL backend ('egl': headless)"),
    "PYTEST_CURRENT_TEST": EnvVar("third_party", "", "set by pytest: the CLI never re-execs under it"),
}


def read(name: str) -> str:
    """The variable's value, else its catalogued default (KeyError: not catalogued)."""
    return os.environ.get(name) or CATALOGUE[name].default
