# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""``vector-os-mcp``: the MCP face of the one agent.

The runtime is ``vcli.runtime.build_runtime``, the same world, tools, verify
namespace, permission policy and session as ``vector-cli -p``; a body starts through
the simulator session (the model's ``start_simulation``, or ``--embodiment``).

    run(text)        ONE natural-language turn through the native producer: the
                     model's words, the verdict (the VECTOR_VERDICT report) and what
                     the permission gate refused
    stop()           the P0 stop: every run cancelled, the stop skill, no LLM call
    status()         the body, the world, the model, what is running
    camera://frame   the running body's camera image (PNG), never ground truth

Transport: stdio by default. A network transport binds loopback only: 1.0 ships no
authentication, and this agent drives a robot and has file tools.

Permissions: nobody can be asked over MCP. What the CLI allows by itself is allowed;
what it would ask about is refused, with the reason in the result, unless the server
was started with ``--allow`` naming that tool.
"""
from __future__ import annotations

import argparse
import io
import os
import signal
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

LOOPBACK = ("127.0.0.1", "localhost", "::1")
ALLOW_NOTE = ("nobody can be asked over MCP: a tool the CLI would ask about is refused "
              "unless vector-os-mcp was started with --allow naming it")


def bind_refusal(transport: str, host: str) -> str:
    """Why *host* may not be bound for *transport* ('' when it may)."""
    if transport == "stdio" or host in LOOPBACK:
        return ""
    return (f"refusing to listen on {host!r}: a robot-driving agent with file tools never "
            f"listens on a network without authentication, and 1.0 ships none. "
            f"Bind one of {', '.join(LOOPBACK)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from vector_os_nano.embodiments.registry import embodiment_ids
    from vector_os_nano.vcli.config import PROVIDERS

    p = argparse.ArgumentParser(prog="vector-os-mcp", description="Vector OS Nano MCP server")
    p.add_argument("--embodiment", choices=embodiment_ids(), metavar="ID",
                   help=f"start this simulation first (else the model starts one): {', '.join(embodiment_ids())}")
    p.add_argument("--headless", action="store_true", help="no MuJoCo viewer window for --embodiment")
    p.add_argument("--provider", choices=sorted(PROVIDERS), help="LLM provider (its key comes from the environment)")
    p.add_argument("--model", help="LLM model")
    p.add_argument("--base-url", help="LLM API base URL")
    p.add_argument("--allow", action="append", default=[], metavar="TOOL[,TOOL]",
                   help="allow these tools the CLI would ask about (nobody can be asked over MCP)")
    p.add_argument("--transport", choices=("stdio", "sse", "streamable-http"), default="stdio")
    p.add_argument("--host", default="127.0.0.1", help=f"network transports only; one of {', '.join(LOOPBACK)}")
    p.add_argument("--port", type=int, default=8100, help="network transports only")
    args = p.parse_args(argv)
    why = bind_refusal(args.transport, args.host)
    if why:
        p.error(why)
    args.allow = frozenset(t.strip() for a in args.allow for t in a.split(",") if t.strip())
    return args


@contextmanager
def _stdout_to_stderr() -> Iterator[None]:
    """fd 1 is the stdio wire: what the runtime prints while it starts (and every child
    it spawns meanwhile) goes to stderr. While serving, the SDK diverts fd 1 itself."""
    sys.stdout.flush()
    saved = os.dup(1)
    os.dup2(2, 1)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def _camera_png(agent: Any) -> bytes:
    """The body's camera image as PNG, read through its own perception (a render of
    the sensor, never ground truth)."""
    import numpy as np
    from PIL import Image
    from mcp.server.mcpserver.exceptions import ResourceError

    perception = getattr(agent, "_perception", None)
    if callable(getattr(perception, "capture", None)):
        rgb = perception.capture().rgb
    elif callable(getattr(perception, "get_color_frame", None)):
        rgb = perception.get_color_frame()
    else:
        raise ResourceError("no camera: the current body has none (start a simulation first)")
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="PNG")
    return buf.getvalue()


def build_server(rt: Any, allow: frozenset[str]) -> Any:
    """The MCP server over runtime *rt*; *allow* names the tools allowed instead of asked."""
    import anyio
    from mcp.server import MCPServer

    from vector_os_nano.vcli.cognitive import abort
    from vector_os_nano.vcli.repl import _stop_report
    from vector_os_nano.vcli.runtime import graded_turn
    from vector_os_nano.vcli.verdict import VerdictReport

    server = MCPServer("vector-os-nano", instructions=__doc__)
    turn_lock = anyio.Lock()  # one turn at a time; stop() never waits for it

    def ask(name: str, _params: dict[str, Any]) -> str:
        return "y" if name in allow else "n"

    def one_turn(text: str) -> tuple[str, Any, Any]:
        with abort.turn():  # a stop anywhere in the turn stops every run it starts
            return graded_turn(rt, text, ask_permission=ask)

    def p0_stop() -> str:
        t0 = time.monotonic()
        rt.engine._emergency_stop("stop", None, rt.agent, rt.app_state)
        line = f"stopped ({(time.monotonic() - t0) * 1000:.0f} ms, no LLM call)"
        report = _stop_report(rt.agent)
        return f"{line}; {report}" if report else line

    @server.tool()
    async def stop() -> str:
        """Stop the robot NOW: cancel every running task and run the stop skill (no LLM call)."""
        return await anyio.to_thread.run_sync(p0_stop)

    @server.tool()
    async def run(text: str) -> dict[str, Any]:
        """Run ONE natural-language turn (any language) through the agent: it may start or
        switch a simulation, plan, act and verify. Returns the agent's answer, the verdict
        (verified, evidence, goal_check, contract, per-step rows, reject_reason) and the
        tool calls the permission gate refused."""
        if rt.engine.is_stop(text):
            answer = await stop()
            report = VerdictReport.no_trace(goal=text, error="stopped (P0, no LLM call)")
            return {"answer": answer, "verdict": report.to_dict(), "refused": []}
        async with turn_lock:
            answer, report, trace = await anyio.to_thread.run_sync(one_turn, text)
        refused = list(getattr(trace, "refused", ()) or ())
        result = {"answer": answer,
                  "verdict": (report or VerdictReport.no_trace(goal=text)).to_dict(),
                  "refused": refused}
        if refused:
            result["note"] = ALLOW_NOTE
        return result

    @server.tool()
    async def status() -> dict[str, Any]:
        """The body, the world, the LLM, the simulation and what is running."""
        sim, creds = rt.app_state.get("sim_session"), rt.app_state["credentials"]
        agent = rt.agent
        return {
            "body": sim.spec.embodiment if sim else (type(agent).__name__ if agent else "none (dev world)"),
            "world": getattr(rt.engine._world, "name", ""),
            "llm": (f"{creds.provider}/{creds.model}" if rt.engine._backend is not None
                    else f"none: set {creds.key_env}"),
            "sim": sim.summary() if sim else "",
            "turn_running": turn_lock.locked(),
            "robot_busy": rt.engine.busy,
            "allowed": sorted(allow),
        }

    @server.resource("camera://frame", name="camera", mime_type="image/png",
                     description="The running body's camera image (PNG)")
    async def camera() -> bytes:
        return await anyio.to_thread.run_sync(_camera_png, rt.agent)

    return server


def main(argv: list[str] | None = None) -> None:
    """The ``vector-os-mcp`` console script."""
    args = parse_args(argv)
    try:
        import mcp.server  # noqa: F401
    except ImportError:
        sys.exit("vector-os-mcp needs the MCP SDK: pip install 'vector-os-nano[mcp]'")
    from vector_os_nano.sim.session import stop_all
    from vector_os_nano.vcli.runtime import build_runtime

    # SIGTERM / SIGHUP take the normal exit path, so every sim this process started stops.
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda signum, _frame: sys.exit(128 + signum))
    try:
        with _stdout_to_stderr():
            rt = build_runtime(provider=args.provider, model=args.model, base_url=args.base_url,
                               embodiment=args.embodiment, gui=not args.headless, face="mcp")
    except Exception as exc:  # noqa: BLE001 — a sim that did not start, a body that cannot bind
        sys.exit(f"vector-os-mcp: setup error: {exc}")
    try:
        server = build_server(rt, args.allow)
        if args.transport == "stdio":
            server.run("stdio")
        else:
            server.run(args.transport, host=args.host, port=args.port)
    finally:
        stop_all()
