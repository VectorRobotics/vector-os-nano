# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Live ROS2 subscriber to SysNav's ``/object_nodes_list``.

Imports rclpy and ``tare_planner.msg`` lazily so:

* unit tests run without a sourced SysNav workspace,
* CI without ROS2 still imports the package cleanly,
* a missing dependency at runtime degrades to a logged warning rather
  than crashing the agent.

Each ``ObjectNodeList`` message is fanned into ``WorldModel.add_object``
calls via the existing :func:`object_node_to_state` adapter (see
``topic_interfaces.py``).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from vector_os_nano.core.world_model import WorldModel
from vector_os_nano.integrations.sysnav_bridge.topic_interfaces import (
    object_node_to_state,
)

logger = logging.getLogger(__name__)


class LiveSysnavBridge:
    """rclpy subscriber + WorldModel updater.

    The bridge owns its own ROS2 node + ``MultiThreadedExecutor`` so it
    does not contend with other rclpy users in the process. Stopping
    the bridge is idempotent and joins the spin thread with a short
    timeout.

    Args:
        world_model: target ``WorldModel`` for object updates.
        node_name: rclpy node name.
        topic: SysNav scene-graph topic (default ``/object_nodes_list``).
        queue_size: subscription queue depth.
        on_disconnect_after_s: emit a single WARNING if no messages
            arrive within this many seconds. Reset on each new message.
    """

    def __init__(
        self,
        world_model: WorldModel,
        node_name: str = "vector_os_nano_sysnav_bridge",
        topic: str = "/object_nodes_list",
        queue_size: int = 50,
        on_disconnect_after_s: float = 5.0,
        on_batch: "Any | None" = None,
    ) -> None:
        self._world_model = world_model
        self._node_name = str(node_name)
        self._topic = str(topic)
        self._queue_size = int(queue_size)
        self._on_disconnect_after_s = float(on_disconnect_after_s)
        # Optional post-batch hook (additive, rule 6): called once after a
        # non-empty node batch lands in the world model — e.g. pushing the
        # fresh object set to a display surface. Exceptions are swallowed;
        # display must never break ingestion.
        self._on_batch = on_batch

        self._active = False
        self._backend: str = "uninitialised"
        self._node: Any = None
        self._sub: Any = None
        self._executor: Any = None
        self._spin_thread: threading.Thread | None = None
        self._rclpy: Any = None

        self._last_msg_t: float | None = None
        self._disconnect_warned: bool = False
        self._lock = threading.Lock()
        self._dropped_count: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        return self._active

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def dropped_count(self) -> int:
        """Number of messages skipped due to malformed-node errors."""
        return self._dropped_count

    def start(self) -> bool:
        """Initialise rclpy + subscription. Idempotent and resilient.

        Returns ``True`` if the bridge is now actively subscribed,
        ``False`` if a runtime dependency is missing (rclpy or
        tare_planner.msg). Either way the agent can continue: a
        ``False`` bridge is a no-op that logs once.
        """
        if self._active:
            return True

        try:
            import rclpy
            from rclpy.executors import MultiThreadedExecutor
            from tare_planner.msg import ObjectNodeList
        except ImportError as exc:
            logger.warning(
                "[sysnav_bridge] dependency missing (%s); "
                "running as no-op until SysNav workspace is sourced", exc,
            )
            self._backend = "missing-dependency"
            self._active = False
            return False

        try:
            if not rclpy.ok():
                rclpy.init(args=[])
        except Exception as exc:
            logger.warning("[sysnav_bridge] rclpy.init failed: %s", exc)
            self._backend = "rclpy-init-failed"
            return False

        self._rclpy = rclpy
        self._node = rclpy.create_node(self._node_name)
        self._sub = self._node.create_subscription(
            ObjectNodeList, self._topic, self._callback, self._queue_size,
        )
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True, name=f"{self._node_name}-spin",
        )
        self._spin_thread.start()
        self._active = True
        self._backend = "live"
        logger.info(
            "[sysnav_bridge] subscribed to %s queue=%d node=%s",
            self._topic, self._queue_size, self._node_name,
        )
        return True

    def stop(self) -> None:
        """Tear down the bridge. Safe to call multiple times."""
        if not self._active and self._executor is None:
            return
        try:
            if self._executor is not None:
                self._executor.shutdown()
        except Exception:
            pass
        try:
            if self._node is not None:
                self._node.destroy_node()
        except Exception:
            pass
        if self._spin_thread is not None and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)

        self._executor = None
        self._node = None
        self._sub = None
        self._spin_thread = None
        self._active = False
        self._backend = "stopped"

    # ------------------------------------------------------------------
    # Disconnect monitoring
    # ------------------------------------------------------------------

    def check_disconnect(self, now: float | None = None) -> bool:
        """Log a WARNING once if no message arrived within the threshold.

        Returns ``True`` after the warning has fired.
        """
        if not self._active or self._on_disconnect_after_s <= 0:
            return self._disconnect_warned
        now = float(now) if now is not None else time.monotonic()
        with self._lock:
            if self._last_msg_t is None or self._disconnect_warned:
                return self._disconnect_warned
            elapsed = now - self._last_msg_t
            if elapsed >= self._on_disconnect_after_s:
                logger.warning(
                    "[sysnav_bridge] no /object_nodes_list message for %.1fs — "
                    "is SysNav running?", elapsed,
                )
                self._disconnect_warned = True
        return self._disconnect_warned

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _callback(self, msg: Any) -> None:
        """ROS2 callback — fan ``msg.nodes`` into world_model."""
        with self._lock:
            self._last_msg_t = time.monotonic()
            self._disconnect_warned = False

        nodes = list(getattr(msg, "nodes", ()) or ())
        for node in nodes:
            try:
                canonical_id = self._canonical_id(node)
                prior = self._world_model.get_object(canonical_id)
                state = object_node_to_state(node, prior=prior)
                self._world_model.add_object(state)
            except Exception as exc:
                self._dropped_count += 1
                logger.warning(
                    "[sysnav_bridge] dropping malformed ObjectNode "
                    "(total dropped=%d): %s", self._dropped_count, exc,
                )
        if nodes and self._on_batch is not None:
            try:
                self._on_batch()
            except Exception as exc:  # noqa: BLE001
                logger.debug("[sysnav_bridge] on_batch hook failed: %s", exc)

    @staticmethod
    def _canonical_id(node: Any) -> str:
        ids = list(getattr(node, "object_id", ()) or ())
        if not ids:
            return "sysnav_unknown"
        try:
            return f"sysnav_{int(ids[0])}"
        except (TypeError, ValueError):
            return "sysnav_unknown"
