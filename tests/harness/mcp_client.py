# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Drive the REAL ``vector-os-mcp`` over stdio with the MCP SDK's own client.

The server is spawned as a subprocess (``python -m vector_os_nano.mcp``) with the
scripted LLM seam (``VECTOR_FAKE_LLM_TOOLS``); only the network model is replaced,
every tool, permission, verify and verdict is the product's own.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

import anyio

_REPO_ROOT = Path(__file__).resolve().parents[2]


def drive_mcp(
    tmp: Path,
    script: dict[str, Any],
    calls: Callable[[Any], Awaitable[Any]],
    *,
    args: tuple[str, ...] = (),
    cwd: Path | None = None,
) -> Any:
    """Start the server, await ``calls(client)`` and return its value."""
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    path = tmp / "mcp_script.json"
    path.write_text(json.dumps(script), encoding="utf-8")
    env = dict(os.environ, VECTOR_FAKE_LLM_TOOLS=str(path),
               PYTHONPATH=os.pathsep.join([str(_REPO_ROOT), os.environ.get("PYTHONPATH", "")]))
    params = StdioServerParameters(command=sys.executable, args=["-m", "vector_os_nano.mcp", *args],
                                   env=env, cwd=str(cwd or tmp))

    async def main() -> Any:
        async with Client(params) as client:
            return await calls(client)

    return anyio.run(main)


async def run(client: Any, text: str) -> dict[str, Any]:
    """One ``run`` call: its structured result (answer, verdict, refused)."""
    result = await client.call_tool("run", {"text": text})
    assert not result.is_error, result
    return result.structured_content
