"""OAuth2 Authorization Code + PKCE flow for Anthropic/Claude subscription.

Implements the same OAuth flow as Claude Code to authenticate with
the user's Claude subscription (Pro/Max). Produces an access token
that works as an x-api-key with its own independent rate limit pool.

Endpoints:
    authorize:  https://platform.claude.com/oauth/authorize
    token:      https://platform.claude.com/v1/oauth/token
    client_id:  https://claude.ai/oauth/claude-code-client-metadata
"""
from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import secrets
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

# OAuth endpoints (same as Claude Code)
CLIENT_ID = "https://claude.ai/oauth/claude-code-client-metadata"
AUTHORIZE_URL = "https://platform.claude.com/oauth/authorize"
TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
SCOPES = "user:inference user:profile"

# Credential storage
CREDS_PATH = Path.home() / ".vector" / "oauth_credentials.json"


def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _find_free_port() -> int:
    """Find a free TCP port for the local callback server."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler that captures the OAuth authorization code."""

    auth_code: str | None = None
    error: str | None = None
    expected_state: str | None = None

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        # Verify state for CSRF protection
        returned_state = params.get("state", [None])[0]
        if self.expected_state and returned_state != self.expected_state:
            _OAuthCallbackHandler.error = "State mismatch (CSRF)"
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h2>Error: state mismatch</h2></body></html>")
            return

        if "code" in params:
            _OAuthCallbackHandler.auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h2>Authenticated! You can close this tab.</h2></body></html>")
        elif "error" in params:
            _OAuthCallbackHandler.error = params.get("error_description", params["error"])[0]
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(f"<html><body><h2>Error: {_OAuthCallbackHandler.error}</h2></body></html>".encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Suppress HTTP log noise


def login_oauth() -> dict[str, Any] | None:
    """Run the full OAuth2 Authorization Code + PKCE flow.

    1. Start local HTTP server for callback
    2. Open browser to Anthropic's authorize URL
    3. Wait for callback with authorization code
    4. Exchange code for access token
    5. Save credentials to ~/.vector/oauth_credentials.json

    Returns:
        Credential dict with accessToken, refreshToken, etc., or None on failure.
    """
    port = _find_free_port()
    redirect_uri = f"http://localhost:{port}/callback"
    verifier, challenge = _generate_pkce()
    state = secrets.token_urlsafe(32)

    # Build authorize URL
    auth_params = urllib.parse.urlencode({
        "client_id": CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPES,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    })
    auth_url = f"{AUTHORIZE_URL}?{auth_params}"

    # Reset handler state
    _OAuthCallbackHandler.auth_code = None
    _OAuthCallbackHandler.error = None
    _OAuthCallbackHandler.expected_state = state

    # Start local server
    server = http.server.HTTPServer(("127.0.0.1", port), _OAuthCallbackHandler)
    server.timeout = 120  # 2 minute timeout

    # Open browser
    webbrowser.open(auth_url)

    # Wait for callback (blocking, with timeout)
    deadline = time.time() + 120
    while _OAuthCallbackHandler.auth_code is None and _OAuthCallbackHandler.error is None:
        if time.time() > deadline:
            server.server_close()
            return None
        server.handle_request()

    server.server_close()

    if _OAuthCallbackHandler.error:
        return None

    auth_code = _OAuthCallbackHandler.auth_code
    if not auth_code:
        return None

    # Exchange code for token
    token_data = urllib.parse.urlencode({
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "code": auth_code,
        "redirect_uri": redirect_uri,
        "code_verifier": verifier,
    }).encode()

    req = urllib.request.Request(
        TOKEN_URL,
        data=token_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
    except Exception:
        return None

    access_token = result.get("access_token", "")
    refresh_token = result.get("refresh_token", "")
    expires_in = result.get("expires_in", 3600)

    if not access_token:
        return None

    creds = {
        "accessToken": access_token,
        "refreshToken": refresh_token,
        "expiresAt": int((time.time() + expires_in) * 1000),
        "scopes": SCOPES.split(),
    }

    # Save
    _save_credentials(creds)
    return creds


def refresh_oauth(refresh_token: str) -> dict[str, Any] | None:
    """Refresh an expired OAuth token."""
    token_data = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "refresh_token": refresh_token,
    }).encode()

    req = urllib.request.Request(
        TOKEN_URL,
        data=token_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
    except Exception:
        return None

    access_token = result.get("access_token", "")
    new_refresh = result.get("refresh_token", refresh_token)
    expires_in = result.get("expires_in", 3600)

    if not access_token:
        return None

    creds = {
        "accessToken": access_token,
        "refreshToken": new_refresh,
        "expiresAt": int((time.time() + expires_in) * 1000),
        "scopes": SCOPES.split(),
    }
    _save_credentials(creds)
    return creds


def load_credentials() -> dict[str, Any] | None:
    """Load saved OAuth credentials, refreshing if expired."""
    if not CREDS_PATH.exists():
        return None
    try:
        creds = json.loads(CREDS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    access_token = creds.get("accessToken", "")
    if not access_token:
        return None

    # Check expiry
    expires_at = creds.get("expiresAt", 0)
    if time.time() * 1000 > expires_at:
        # Try refresh
        rt = creds.get("refreshToken", "")
        if rt:
            refreshed = refresh_oauth(rt)
            if refreshed:
                return refreshed
        return None

    return creds


def _save_credentials(creds: dict[str, Any]) -> None:
    CREDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CREDS_PATH.write_text(json.dumps(creds, indent=2), encoding="utf-8")
    os.chmod(str(CREDS_PATH), 0o600)
