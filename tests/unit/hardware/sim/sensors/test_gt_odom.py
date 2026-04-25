# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""GroundTruthOdomPublisher unit tests (v2.4 T2).

Exercises body-pose readback, finite-difference twist, quaternion
normalisation + ROS xyzw ordering, rate-limit caching.
"""
from __future__ import annotations

import math

import pytest

from vector_os_nano.hardware.sim.sensors import (
    GroundTruthOdomPublisher,
    OdomSample,
)


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_position_matches_body_xpos(tiny_model_data, trunk_translated_to) -> None:
    model, data = tiny_model_data
    _, move = trunk_translated_to
    move(1.5, 2.0, 0.5)

    pub = GroundTruthOdomPublisher(model, data)
    sample = pub.step(now=0.0)

    assert sample.position == (1.5, 2.0, 0.5)


def test_orientation_quaternion_normalised(tiny_model_data) -> None:
    """Even if MuJoCo returns a non-unit quat, output is normalised."""
    model, data = tiny_model_data
    # Inject a non-unit quaternion (wxyz). MuJoCo's mj_forward does not
    # re-normalise data.xquat in-place outside of integration, so we
    # set xquat directly to test our defensive code path.
    data.xquat[1, :] = [2.0, 0.0, 0.0, 0.0]   # body id 1 = trunk

    pub = GroundTruthOdomPublisher(model, data)
    qx, qy, qz, qw = pub.step(now=0.0).orientation
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    assert norm == pytest.approx(1.0, abs=1e-6)
    # MuJoCo wxyz (2, 0, 0, 0) → ROS xyzw (0, 0, 0, 1) after normalisation
    assert (qx, qy, qz, qw) == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-9)


def test_first_call_twist_is_zero(tiny_model_data, trunk_translated_to) -> None:
    model, data = tiny_model_data
    _, move = trunk_translated_to
    move(0.0, 0.0, 0.5)

    pub = GroundTruthOdomPublisher(model, data)
    sample = pub.step(now=0.0)

    assert sample.linear_twist == (0.0, 0.0, 0.0)
    assert sample.angular_twist == (0.0, 0.0, 0.0)


def test_twist_is_finite_difference_after_translation(
    tiny_model_data, trunk_translated_to
) -> None:
    """Translate body 0.1 m forward over 0.1 s → vx ≈ 1.0 m/s."""
    model, data = tiny_model_data
    _, move = trunk_translated_to
    move(0.0, 0.0, 0.5)

    pub = GroundTruthOdomPublisher(model, data)
    pub.step(now=0.0)        # prime
    move(0.1, 0.0, 0.5)
    sample = pub.step(now=0.1)

    vx, vy, vz = sample.linear_twist
    assert vx == pytest.approx(1.0, rel=0.05)
    assert vy == pytest.approx(0.0, abs=1e-6)
    assert vz == pytest.approx(0.0, abs=1e-6)


def test_orientation_unchanged_returns_zero_angular_twist(
    tiny_model_data, trunk_translated_to
) -> None:
    """Translating without rotating → angular twist (0, 0, 0)."""
    model, data = tiny_model_data
    _, move = trunk_translated_to
    move(0.0, 0.0, 0.5)

    pub = GroundTruthOdomPublisher(model, data)
    pub.step(now=0.0)
    move(0.5, 0.5, 0.5)
    sample = pub.step(now=0.1)

    assert sample.angular_twist == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)


# ---------------------------------------------------------------------------
# frame ids and config
# ---------------------------------------------------------------------------


def test_frame_id_default_map(tiny_model_data) -> None:
    pub = GroundTruthOdomPublisher(*tiny_model_data)
    sample = pub.step(now=0.0)
    assert sample.frame_id == "map"


def test_child_frame_id_default_sensor(tiny_model_data) -> None:
    pub = GroundTruthOdomPublisher(*tiny_model_data)
    sample = pub.step(now=0.0)
    assert sample.child_frame_id == "sensor"


def test_custom_frame_ids_propagated(tiny_model_data) -> None:
    pub = GroundTruthOdomPublisher(
        *tiny_model_data, frame_id="odom", child_frame_id="base_link",
    )
    sample = pub.step(now=0.0)
    assert sample.frame_id == "odom"
    assert sample.child_frame_id == "base_link"


def test_unknown_body_name_raises(tiny_model_data) -> None:
    model, data = tiny_model_data
    with pytest.raises(ValueError, match="not found"):
        GroundTruthOdomPublisher(model, data, body_name="nonexistent_body")


# ---------------------------------------------------------------------------
# rate-limit caching
# ---------------------------------------------------------------------------


def test_rate_limit_step_returns_cached_msg(
    tiny_model_data, trunk_translated_to
) -> None:
    """Two step calls within 1/rate_hz return the SAME OdomSample instance."""
    model, data = tiny_model_data
    _, move = trunk_translated_to
    move(0.0, 0.0, 0.5)

    pub = GroundTruthOdomPublisher(model, data, rate_hz=50.0)
    first = pub.step(now=0.0)
    # 0.005 s later — well within 1/50 = 0.02 s
    move(1.0, 0.0, 0.5)              # body MOVED, but rate-limit hides it
    second = pub.step(now=0.005)

    assert second is first              # exact instance
    assert second.position == (0.0, 0.0, 0.5)


def test_rate_limit_yields_after_interval(
    tiny_model_data, trunk_translated_to
) -> None:
    """Past 1/rate_hz, step() returns a fresh sample."""
    model, data = tiny_model_data
    _, move = trunk_translated_to
    move(0.0, 0.0, 0.5)

    pub = GroundTruthOdomPublisher(model, data, rate_hz=50.0)
    pub.step(now=0.0)
    move(1.0, 0.0, 0.5)
    sample = pub.step(now=0.10)

    assert sample.position == (1.0, 0.0, 0.5)


def test_due_returns_true_initially(tiny_model_data) -> None:
    pub = GroundTruthOdomPublisher(*tiny_model_data, rate_hz=50.0)
    assert pub.due(now=0.0) is True


def test_due_returns_false_within_interval(
    tiny_model_data, trunk_translated_to
) -> None:
    model, data = tiny_model_data
    _, move = trunk_translated_to
    move(0.0, 0.0, 0.5)
    pub = GroundTruthOdomPublisher(model, data, rate_hz=50.0)
    pub.step(now=0.0)
    assert pub.due(now=0.005) is False


def test_dt_clamp_against_zero_division(
    tiny_model_data, trunk_translated_to
) -> None:
    """Back-to-back step() calls at the same wall time must not divide-by-zero.

    The clamp logic returns the cached sample once rate-limited, but
    this test directly exercises the dt floor by forcing two finite-
    difference computations at the same `now`.
    """
    model, data = tiny_model_data
    _, move = trunk_translated_to
    move(0.0, 0.0, 0.5)

    pub = GroundTruthOdomPublisher(model, data, rate_hz=1e9)   # never cache
    pub.step(now=0.0)
    move(0.001, 0.0, 0.5)
    # At the same monotonic time but rate=∞ so caching is bypassed:
    sample = pub.step(now=0.0)
    # Twist must be finite (clamped dt = 1e-3 s → vx = 1.0 m/s here)
    assert all(math.isfinite(v) for v in sample.linear_twist)


# ---------------------------------------------------------------------------
# OdomSample dataclass invariants
# ---------------------------------------------------------------------------


def test_odom_sample_is_frozen(tiny_model_data) -> None:
    sample = GroundTruthOdomPublisher(*tiny_model_data).step(now=0.0)
    with pytest.raises(Exception):       # FrozenInstanceError or AttributeError
        sample.position = (9.0, 9.0, 9.0)   # type: ignore[misc]


def test_body_id_property_resolves(tiny_model_data) -> None:
    pub = GroundTruthOdomPublisher(*tiny_model_data)
    assert pub.body_id > 0


def test_rate_hz_property(tiny_model_data) -> None:
    pub = GroundTruthOdomPublisher(*tiny_model_data, rate_hz=42.0)
    assert pub.rate_hz == 42.0
