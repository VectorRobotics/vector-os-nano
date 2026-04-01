"""WebFetchTool — fetch a URL and return its text content.

Read-only, concurrency-safe.  Uses only stdlib (urllib + re) — no extra deps.

Security: blocks requests to localhost, 127.x.x.x, 0.0.0.0, and RFC-1918
private ranges to prevent SSRF.
"""
from __future__ import annotations

import ipaddress
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from vector_os_nano.vcli.tools.base import PermissionResult, ToolContext, ToolResult, tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_CHARS: int = 10_000
_TIMEOUT_SECONDS: float = 10.0

_BLOCKED_HOSTNAMES: frozenset[str] = frozenset(
    {"localhost", "localhost.localdomain"}
)

_PRIVATE_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),  # link-local
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),  # unique local IPv6
)

# Single-pass tag stripper: matches any HTML/XML tag
_TAG_RE: re.Pattern[str] = re.compile(r"<[^>]+>")

# Collapse runs of whitespace (spaces, tabs, newlines) into a single space
_WHITESPACE_RE: re.Pattern[str] = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# SSRF guard helpers
# ---------------------------------------------------------------------------


def _is_blocked_url(url: str) -> bool:
    """Return True when *url* targets a private/loopback address (SSRF guard).

    Resolves the hostname to an IP and checks it against private ranges.
    Returns True (blocked) on any resolution failure to fail-safe.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return True

    hostname = parsed.hostname
    if hostname is None:
        return True

    if hostname.lower() in _BLOCKED_HOSTNAMES:
        return True

    # Resolve hostname → IP (may raise on lookup failure)
    try:
        addr_str = socket.getaddrinfo(hostname, None)[0][4][0]
        addr = ipaddress.ip_address(addr_str)
    except (OSError, ValueError, IndexError):
        # Fail-safe: block on resolution errors
        return True

    for network in _PRIVATE_NETWORKS:
        if addr in network:
            return True

    return False


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
    permission="allow",
)
class WebFetchTool:
    """Fetch a remote URL and return its plain-text content.

    Concurrency-safe: stateless (no shared mutable state).
    Read-only: never writes to disk or external systems.
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

    # Concurrency-safe: no instance state is mutated during execute()
    def is_concurrency_safe(self, params: dict[str, Any]) -> bool:
        return True

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        url: str = params.get("url", "")
        max_chars: int = int(params.get("max_chars", _DEFAULT_MAX_CHARS))

        # 1. Validate scheme — only http/https allowed
        try:
            parsed = urllib.parse.urlparse(url)
        except ValueError as exc:
            return ToolResult(content=f"Invalid URL: {exc}", is_error=True)

        if parsed.scheme not in ("http", "https"):
            return ToolResult(
                content=f"Unsupported scheme '{parsed.scheme}': only http and https are allowed.",
                is_error=True,
            )

        # 2. SSRF guard
        if _is_blocked_url(url):
            return ToolResult(
                content=f"Access denied: '{url}' resolves to a private or loopback address.",
                is_error=True,
            )

        # 3. Fetch
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "VectorOS/1.0 WebFetchTool"},
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as response:
                raw_bytes: bytes = response.read()
        except urllib.error.HTTPError as exc:
            return ToolResult(
                content=f"HTTP error {exc.code}: {exc.reason}",
                is_error=True,
            )
        except urllib.error.URLError as exc:
            return ToolResult(
                content=f"Request failed: {exc.reason}",
                is_error=True,
            )
        except TimeoutError:
            return ToolResult(
                content=f"Request timed out after {_TIMEOUT_SECONDS}s: {url}",
                is_error=True,
            )
        except OSError as exc:
            return ToolResult(content=f"Network error: {exc}", is_error=True)

        # 4. Decode
        try:
            html = raw_bytes.decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 — broad catch for decode edge-cases
            return ToolResult(content=f"Decode error: {exc}", is_error=True)

        # 5. Strip HTML, collapse whitespace
        text = _html_to_text(html)

        # 6. Truncate
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n[truncated — {len(text)} chars total]"

        return ToolResult(content=text)
