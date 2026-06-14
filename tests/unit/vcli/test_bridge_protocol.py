# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Batch 1 (campaign #4) — bridge line-protocol pairing + timeout tiers (#15).

The main channel used a flat 60 s socket timeout against unbounded paced
motion: "walk 20 m" took 67 s, ``readline`` raised a BARE socket.timeout
bypassing the error contract, and every later response was offset by one
line (a pano parsed as a navigate result) with no recovery path.

Contract now: every request carries a monotonically increasing ``rid`` the
server echoes; a STALE response (rid lower than expected) is discarded and
the read continues — that is the resync path (a main-channel reconnect is
impossible by design: closing the main connection shuts the server down).
A rid from the future / torn JSON marks the bridge dead (every later call
fails loud). Read timeouts are estimated per op (walk: duration+margin;
navigate/renders: generous tiers) and a timeout raises HabitatBridgeError —
never a bare socket.timeout. Responses WITHOUT a rid are accepted
unchanged (legacy fake servers in older tests).
"""
from __future__ import annotations

import json
import socket

import pytest

from vector_os_nano.playground.habitat.bridge import (
    HabitatBridge,
    HabitatBridgeError,
)


def _wired_bridge():
    """Bridge with its socket replaced by one end of a socketpair."""
    a, b = socket.socketpair()
    a.settimeout(5.0)   # test harness guard: a buggy bridge must FAIL, not hang
    br = HabitatBridge("test-scene")
    br._sock = a
    br._rfile = a.makefile("r", encoding="utf-8")
    br._wfile = a.makefile("w", encoding="utf-8")
    peer_r = b.makefile("r", encoding="utf-8")
    peer_w = b.makefile("w", encoding="utf-8")
    return br, peer_r, peer_w


def _peer_reply(peer_r, peer_w, extra=None, rid_override=None, count=1):
    """Read `count` requests off the peer side; answer each."""
    reqs = []
    for _ in range(count):
        req = json.loads(peer_r.readline())
        reqs.append(req)
        resp = {"ok": True, "echo": req.get("op", "")}
        if extra:
            resp.update(extra)
        rid = req.get("rid") if rid_override is None else rid_override
        if rid is not None:
            resp["rid"] = rid
        peer_w.write(json.dumps(resp) + "\n")
        peer_w.flush()
    return reqs


class TestRidPairing:
    def test_rid_injected_and_matched(self):
        br, pr, pw = _wired_bridge()
        import threading
        t = threading.Thread(target=_peer_reply, args=(pr, pw))
        t.start()
        resp = br.request({"op": "ping"})
        t.join()
        assert resp["ok"] is True
        # next request increments the rid
        t = threading.Thread(target=_peer_reply, args=(pr, pw))
        t.start()
        br.request({"op": "ping"})
        t.join()

    def test_stale_response_discarded(self):
        br, pr, pw = _wired_bridge()
        import threading

        def peer():
            req = json.loads(pr.readline())
            rid = req["rid"]
            # a LATE response from a previous (timed-out) request first…
            pw.write(json.dumps({"ok": True, "echo": "stale", "rid": rid - 1})
                     + "\n")
            # …then the real answer
            pw.write(json.dumps({"ok": True, "echo": "fresh", "rid": rid})
                     + "\n")
            pw.flush()

        t = threading.Thread(target=peer)
        t.start()
        resp = br.request({"op": "get_state"})
        t.join()
        assert resp["echo"] == "fresh"          # stale line skipped, not parsed

    def test_future_rid_marks_bridge_dead(self):
        br, pr, pw = _wired_bridge()
        import threading
        t = threading.Thread(
            target=_peer_reply, args=(pr, pw), kwargs={"rid_override": 9999})
        t.start()
        with pytest.raises(HabitatBridgeError, match="desync"):
            br.request({"op": "ping"})
        t.join()
        # the bridge is DEAD now: no further protocol traffic
        with pytest.raises(HabitatBridgeError, match="dead"):
            br.request({"op": "ping"})

    def test_legacy_response_without_rid_accepted(self):
        br, pr, pw = _wired_bridge()
        import threading

        def peer_no_rid():
            req = json.loads(pr.readline())
            pw.write(json.dumps({"ok": True, "echo": req["op"]}) + "\n")
            pw.flush()

        t = threading.Thread(target=peer_no_rid)
        t.start()
        resp = br.request({"op": "ping"})
        t.join()
        assert resp["echo"] == "ping"

    def test_torn_line_marks_bridge_dead(self):
        br, pr, pw = _wired_bridge()
        import threading

        def peer():
            pr.readline()
            pw.write("{not json\n")
            pw.flush()

        t = threading.Thread(target=peer)
        t.start()
        with pytest.raises(HabitatBridgeError):
            br.request({"op": "ping"})
        t.join()
        with pytest.raises(HabitatBridgeError, match="dead"):
            br.request({"op": "ping"})


class TestTimeoutTiers:
    def test_walk_timeout_scales_with_duration(self):
        br = HabitatBridge("test-scene")
        assert br._op_timeout({"op": "walk", "duration": 50.0}) >= 80.0
        assert br._op_timeout({"op": "walk"}) >= 30.0

    def test_navigate_and_render_tiers(self):
        br = HabitatBridge("test-scene")
        assert br._op_timeout({"op": "navigate_to", "x": 1, "y": 2}) >= 300.0
        assert br._op_timeout({"op": "pano"}) >= 120.0
        assert br._op_timeout({"op": "render"}) >= 120.0
        assert br._op_timeout({"op": "ping"}) <= 60.0

    def test_read_timeout_is_bridge_error_and_recoverable(self):
        br, pr, pw = _wired_bridge()
        br._op_timeout = lambda payload: 0.2     # force a tiny read timeout
        import threading

        with pytest.raises(HabitatBridgeError, match="timeout"):
            br.request({"op": "get_state"})       # peer never answers

        # the LATE response for rid 1 arrives now; the next request must
        # discard it by rid and pick up its own answer (resync, not death)
        def peer():
            req1 = json.loads(pr.readline())      # the timed-out request
            req2 = json.loads(pr.readline())      # the follow-up
            pw.write(json.dumps({"ok": True, "echo": "late",
                                 "rid": req1["rid"]}) + "\n")
            pw.write(json.dumps({"ok": True, "echo": "current",
                                 "rid": req2["rid"]}) + "\n")
            pw.flush()

        br._op_timeout = lambda payload: 5.0
        t = threading.Thread(target=peer)
        t.start()
        resp = br.request({"op": "get_state"})
        t.join()
        assert resp["echo"] == "current"
