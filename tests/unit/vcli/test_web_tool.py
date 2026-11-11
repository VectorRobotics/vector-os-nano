# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Adversarial tests for web_fetch (R3): permission, SSRF on every hop, caps.

Hermetic: a local http.server on 127.0.0.1 stands in for "the internet". Loopback
is (correctly) refused by the real guard, so tests that need the first hop to go
through treat exactly that server's address as public; every other address keeps
the real predicate, so a redirect to a private address is judged for real.
"""
from __future__ import annotations

import http.server
import ipaddress
import socket
import threading
from pathlib import Path
from typing import Any, Iterator

import pytest

from vector_os_nano.vcli.permissions import PermissionContext
from vector_os_nano.vcli.tools import web_tool
from vector_os_nano.vcli.tools.base import ToolContext
from vector_os_nano.vcli.tools.web_tool import WebFetchTool

_LOOP = ipaddress.ip_address("127.0.0.1")


class _Handler(http.server.BaseHTTPRequestHandler):
    hits: list[str] = []

    def log_message(self, *a: Any) -> None:  # keep the test output quiet
        pass

    def do_GET(self) -> None:  # noqa: N802
        _Handler.hits.append(self.path)
        port = self.server.server_address[1]
        routes = {
            "/redir-private": "http://10.0.0.1/secret",
            "/redir-mapped": "http://[::ffff:127.0.0.1]:%d/ok" % port,
            "/redir-metadata": "http://169.254.169.254/latest/meta-data/",
            "/redir-file": "file:///etc/passwd",
            "/redir-ok": "/ok",
        }
        if self.path in routes:
            self.send_response(302)
            self.send_header("Location", routes[self.path])
            self.end_headers()
            return
        if self.path.startswith("/loop"):
            n = int(self.path[5:] or 0)
            self.send_response(302)
            self.send_header("Location", f"/loop{n + 1}")
            self.end_headers()
            return
        if self.path == "/big":
            self.send_response(200)
            self.end_headers()
            chunk = b"A" * 65536
            try:
                for _ in range(200):  # ~13 MB, far over the cap
                    self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        if self.path == "/slow":
            self.send_response(200)
            self.end_headers()
            try:
                for _ in range(100):
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    threading.Event().wait(0.2)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<html><body><p>hello   world</p></body></html>")


@pytest.fixture()
def server() -> Iterator[str]:
    _Handler.hits = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture()
def server_is_public(monkeypatch: pytest.MonkeyPatch) -> None:
    real = web_tool._is_public
    monkeypatch.setattr(web_tool, "_is_public", lambda a: a == _LOOP or real(a))


def _fetch(url: str, **kw: Any) -> Any:
    ctx = ToolContext(agent=None, cwd=Path("."), session=None, permissions=None,
                      abort=threading.Event())
    return WebFetchTool().execute({"url": url, **kw}, ctx)


# ---------------------------------------------------------------------------
# Permission
# ---------------------------------------------------------------------------

def test_web_fetch_asks_by_default() -> None:
    """A fetch can carry data out: it is never auto-allowed (nor by read_only)."""
    assert PermissionContext().check(WebFetchTool(), {"url": "https://example.com"}).behavior == "ask"


def test_web_fetch_never_joins_a_parallel_batch() -> None:
    """It always asks, and prompts from a thread pool would interleave."""
    assert WebFetchTool().is_concurrency_safe({"url": "https://example.com"}) is False


# ---------------------------------------------------------------------------
# The address predicate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("addr", [
    "127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.1.1", "169.254.169.254", "0.0.0.0",
    "100.64.0.1", "224.0.0.1", "::1", "fe80::1", "fd00::1", "::ffff:127.0.0.1",
    "::ffff:10.0.0.1", "64:ff9b::7f00:1", "2002:7f00:1::1", "::",
])
def test_non_public_addresses_are_refused(addr: str) -> None:
    assert web_tool._is_public(ipaddress.ip_address(addr)) is False


@pytest.mark.parametrize("addr", ["93.184.216.34", "8.8.8.8", "2606:4700:4700::1111"])
def test_public_addresses_pass(addr: str) -> None:
    assert web_tool._is_public(ipaddress.ip_address(addr)) is True


def test_one_private_answer_among_public_ones_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """DNS that returns a public AND a loopback address must not be fetched."""
    infos = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80)),
             (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]
    monkeypatch.setattr(web_tool.socket, "getaddrinfo", lambda *a, **k: infos)
    res = _fetch("http://rebind.test/")
    assert res.is_error and "private or loopback" in res.content


# ---------------------------------------------------------------------------
# Hops
# ---------------------------------------------------------------------------

def test_loopback_url_is_refused_without_connecting(server: str) -> None:
    res = _fetch(server + "/ok")
    assert res.is_error and "Access denied" in res.content
    assert _Handler.hits == []


def test_ordinary_fetch_and_a_same_host_redirect_work(server: str, server_is_public: None) -> None:
    res = _fetch(server + "/redir-ok")
    assert not res.is_error, res.content
    assert res.content == "hello world"


@pytest.mark.parametrize("route", ["/redir-private", "/redir-metadata", "/redir-mapped", "/redir-file"])
def test_a_redirect_to_a_private_address_is_refused(server: str, server_is_public: None, route: str) -> None:
    res = _fetch(server + route)
    assert res.is_error and "Access denied" in res.content, res.content
    assert _Handler.hits == [route]  # the redirect target was never fetched


def test_redirect_chain_is_capped(server: str, server_is_public: None) -> None:
    res = _fetch(server + "/loop0")
    assert res.is_error and "too many redirects" in res.content
    assert len(_Handler.hits) == web_tool._MAX_REDIRECTS + 1


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------

def test_body_is_capped(server: str, server_is_public: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_tool, "_MAX_BODY_BYTES", 100_000)
    seen: list[int] = []
    real = web_tool._html_to_text
    monkeypatch.setattr(web_tool, "_html_to_text", lambda h: (seen.append(len(h)), real(h))[1])
    res = _fetch(server + "/big", max_chars=10_000_000)
    assert not res.is_error, res.content
    assert seen == [100_000]
    assert "body cut at 100000 bytes" in res.content


def test_total_time_is_capped(server: str, server_is_public: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A server that drips one byte at a time cannot hold the call past the cap."""
    import time

    monkeypatch.setattr(web_tool, "_TOTAL_TIMEOUT_S", 1.0)
    t0 = time.monotonic()
    res = _fetch(server + "/slow")
    assert res.is_error and "timed out" in res.content
    assert time.monotonic() - t0 < 5.0
