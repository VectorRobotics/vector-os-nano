# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Batch 1 (campaign #4) — habitat server main-channel protocol hardening (#20).

A malformed FIRST line on the main channel left ``req`` unbound, so the
post-response ``req.get("op") == "shutdown"`` check raised NameError and
killed the whole server (stderr swallowed by DEVNULL — the owner only saw
"connection closed"). The serve loop is now an importable pure-stdlib
function (``_serve_main``) with the op localized per iteration, matching
the stream handler's already-correct shape.

server.py imports habitat_sim at module top (it runs in the pinned conda
env); the tests stub that import — none of the exercised code touches it.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_SERVER = (Path(__file__).resolve().parents[3] / "vector_os_nano"
           / "playground" / "habitat" / "server.py")


@pytest.fixture(scope="module")
def server_mod():
    sys.modules.setdefault("habitat_sim", MagicMock())
    spec = importlib.util.spec_from_file_location("_hab_server_test", _SERVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeServer:
    def __init__(self):
        self.handled: list = []

    def handle(self, req: dict) -> dict:
        self.handled.append(req)
        return {"ok": True, "echo": req.get("op", "")}


def _run(server_mod, lines: str):
    srv = _FakeServer()
    wfile = io.StringIO()
    server_mod._serve_main(io.StringIO(lines), wfile, srv)
    out = [json.loads(line) for line in wfile.getvalue().splitlines()]
    return srv, out


class TestMalformedJson:
    def test_malformed_first_line_does_not_kill_server(self, server_mod):
        # The #20 shape: bad json on line 1 → must answer the error and KEEP
        # SERVING (the old code died on NameError before the next read).
        srv, out = _run(
            server_mod,
            'not json\n{"op": "state"}\n{"op": "shutdown"}\n')
        assert out[0]["ok"] is False
        assert "bad json" in out[0]["error"]
        assert [r["op"] for r in srv.handled] == ["state", "shutdown"]

    def test_handler_exception_reported_not_fatal(self, server_mod):
        class _Boom(_FakeServer):
            def handle(self, req):
                if req.get("op") == "explode":
                    raise RuntimeError("kaput")
                return super().handle(req)

        srv = _Boom()
        wfile = io.StringIO()
        server_mod._serve_main(
            io.StringIO('{"op": "explode"}\n{"op": "shutdown"}\n'),
            wfile, srv)
        out = [json.loads(line) for line in wfile.getvalue().splitlines()]
        assert out[0]["ok"] is False and "RuntimeError" in out[0]["error"]
        assert out[1]["ok"] is True


class TestShutdown:
    def test_shutdown_breaks_loop(self, server_mod):
        srv, out = _run(
            server_mod,
            '{"op": "shutdown"}\n{"op": "after"}\n')
        # the line after shutdown is never handled
        assert [r["op"] for r in srv.handled] == ["shutdown"]
        assert len(out) == 1

    def test_blank_lines_ignored(self, server_mod):
        srv, out = _run(server_mod, '\n\n{"op": "shutdown"}\n')
        assert [r["op"] for r in srv.handled] == ["shutdown"]
