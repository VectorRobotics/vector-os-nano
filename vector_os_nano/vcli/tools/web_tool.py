# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""WebFetchTool — fetch a URL and return its text content.

Read-only; asks before every fetch, so never run in a parallel batch. Stdlib only (http.client + re).

Security (the fetched page is attacker-controlled, and so may be the URL):
- permission "ask": a fetch is an outward-facing action that can carry data out;
- every hop (the URL and each redirect, at most _MAX_REDIRECTS) is checked: http(s)
  only, and EVERY resolved address must be public (no loopback / private /
  link-local / unique-local / CGNAT / v4-mapped forms of those);
- the connection goes to the address that was checked (no DNS-rebinding window);
- the body is capped at _MAX_BODY_BYTES and the whole fetch at _TOTAL_TIMEOUT_S.
"""
from __future__ import annotations

import http.client
import ipaddress
import re
import socket
import ssl
import threading
import time
import urllib.parse
from typing import Any

from vector_os_nano.vcli.tools.base import ToolContext, ToolResult, tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_CHARS: int = 10_000
_TOTAL_TIMEOUT_S: float = 20.0
_MAX_REDIRECTS: int = 5
_MAX_BODY_BYTES: int = 2_000_000
_REDIRECT_CODES: frozenset[int] = frozenset({301, 302, 303, 307, 308})

# Single-pass tag stripper: matches any HTML/XML tag
_TAG_RE: re.Pattern[str] = re.compile(r"<[^>]+>")

# Collapse runs of whitespace (spaces, tabs, newlines) into a single space
_WHITESPACE_RE: re.Pattern[str] = re.compile(r"\s+")


class _Refused(Exception):
    """A hop the SSRF guard will not fetch."""


# ---------------------------------------------------------------------------
# SSRF guard helpers
# ---------------------------------------------------------------------------


_NAT64 = ipaddress.ip_network("64:ff9b::/96")


def _is_public(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True only for a globally routable unicast address. An IPv6 address that
    embeds an IPv4 one (v4-mapped, 6to4, NAT64) must be public on both readings."""
    if not addr.is_global or addr.is_multicast:
        return False
    if isinstance(addr, ipaddress.IPv6Address):
        inner = addr.ipv4_mapped or addr.sixtofour or (
            ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF) if addr in _NAT64 else None
        )
        if inner is not None:
            return _is_public(inner)
    return True


def _public_address(host: str, port: int) -> str:
    """Resolve *host*; return an address to connect to, or raise _Refused when ANY
    resolved address is not public (one private answer poisons the whole set)."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as exc:
        raise _Refused(f"cannot resolve {host!r}: {exc}") from exc
    if not infos:
        raise _Refused(f"cannot resolve {host!r}")
    for info in infos:
        addr = ipaddress.ip_address(str(info[4][0]).split("%", 1)[0])
        if not _is_public(addr):
            raise _Refused(f"{host!r} resolves to a private or loopback address ({addr})")
    return str(infos[0][4][0])


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """Connects to the pre-checked address while keeping the URL's Host header."""

    def __init__(self, host: str, port: int, ip: str, timeout: float) -> None:
        super().__init__(host, port, timeout=timeout)
        self._ip = ip

    def connect(self) -> None:
        self.sock = self.raw_sock = socket.create_connection((self._ip, self.port), self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, port: int, ip: str, timeout: float) -> None:
        super().__init__(host, port, timeout=timeout, context=ssl.create_default_context())
        self._ip = ip

    def connect(self) -> None:
        sock = socket.create_connection((self._ip, self.port), self.timeout)
        self.sock = self.raw_sock = self._context.wrap_socket(sock, server_hostname=self.host)


def _fetch(url: str) -> tuple[bytes, bool]:
    """GET *url*, checking every hop; return (body, truncated). Raises _Refused / OSError."""
    deadline = time.monotonic() + _TOTAL_TIMEOUT_S
    for _hop in range(_MAX_REDIRECTS + 1):
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https"):
            raise _Refused(f"unsupported scheme {parts.scheme!r}: only http and https are allowed")
        if not parts.hostname:
            raise _Refused(f"no host in {url!r}")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        ip = _public_address(parts.hostname, port)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        cls = _PinnedHTTPSConnection if parts.scheme == "https" else _PinnedHTTPConnection
        conn = cls(parts.hostname, port, ip, timeout=remaining)
        # The socket timeout is per recv; a server dripping header or chunk-size bytes
        # keeps every recv short. The deadline is enforced by shutting the socket down.
        expired = threading.Event()

        def _expire(conn: http.client.HTTPConnection = conn) -> None:
            expired.set()
            sock = getattr(conn, "raw_sock", None)  # conn.sock is None once a response owns it
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        timer = threading.Timer(remaining, _expire)
        timer.daemon = True
        timer.start()
        try:
            target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
            conn.request("GET", target, headers={"User-Agent": "VectorOS/1.0 WebFetchTool"})
            resp = conn.getresponse()
            location = resp.getheader("Location")
            if resp.status in _REDIRECT_CODES and location:
                url = urllib.parse.urljoin(url, location)
                continue
            if resp.status >= 400:
                raise http.client.HTTPException(f"HTTP error {resp.status}: {resp.reason}")
            body = bytearray()
            while len(body) <= _MAX_BODY_BYTES:
                chunk = resp.read1(65536)
                if expired.is_set():  # a shut-down socket reads as EOF: not a full body
                    raise TimeoutError
                if not chunk:
                    return bytes(body), False
                body += chunk
            return bytes(body[:_MAX_BODY_BYTES]), True
        except (OSError, http.client.HTTPException, ValueError):
            if expired.is_set():
                raise TimeoutError from None
            raise
        finally:
            timer.cancel()
            conn.close()
    raise _Refused(f"too many redirects (more than {_MAX_REDIRECTS})")


# ---------------------------------------------------------------------------
# HTML-to-text helpers
# ---------------------------------------------------------------------------


def _html_to_text(html: str) -> str:
    """Strip HTML tags and collapse whitespace into a readable text blob."""
    text = _TAG_RE.sub("", html)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# WebFetchTool
# ---------------------------------------------------------------------------


@tool(
    name="web_fetch",
    description="Fetch a URL and return its text content",
    read_only=True,
    permission="ask",
)
class WebFetchTool:
    """Fetch a remote URL and return its plain-text content.

    Not concurrency-safe: every call asks first, and prompts must not interleave.
    """

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch (http or https only)",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum number of characters to return",
                "default": _DEFAULT_MAX_CHARS,
            },
        },
        "required": ["url"],
    }

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        url: str = params.get("url", "")
        max_chars: int = int(params.get("max_chars", _DEFAULT_MAX_CHARS))

        try:
            raw_bytes, truncated = _fetch(url)
        except _Refused as exc:
            return ToolResult(content=f"Access denied: {exc}", is_error=True)
        except (TimeoutError, socket.timeout):
            return ToolResult(
                content=f"Request timed out after {_TOTAL_TIMEOUT_S}s: {url}", is_error=True
            )
        except (OSError, http.client.HTTPException, ValueError) as exc:
            return ToolResult(content=f"Request failed: {exc}", is_error=True)

        # 4. Decode
        try:
            html = raw_bytes.decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 — broad catch for decode edge-cases
            return ToolResult(content=f"Decode error: {exc}", is_error=True)

        # 5. Strip HTML, collapse whitespace
        text = _html_to_text(html)

        # 6. Truncate
        if truncated:
            text += f"\n[body cut at {_MAX_BODY_BYTES} bytes]"
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n[truncated — {len(text)} chars total]"

        return ToolResult(content=text)
