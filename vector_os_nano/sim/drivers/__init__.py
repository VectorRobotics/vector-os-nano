# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Generic drivers, one per morphology class (charter 3.3): a body is a manifest.

Each is built from its manifest block and a ``SimRuntime`` and talks to the world only
through that runtime (snapshot reads, step-boundary jobs, the grasp decision, the
camera path), so it runs unchanged on every backend. Actor code: no simulator import,
no ground truth, no ``attach`` / ``detach``.
"""
