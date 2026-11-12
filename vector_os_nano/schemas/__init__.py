# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The config-file and file interfaces (charter sections 4 and 5).

* :mod:`.paths` — the state directory ``$VECTOR_HOME``: the one owner of every path in it.
* :mod:`.env` — every environment variable the package reads.
* :mod:`.user` — ``$VECTOR_HOME/config.yaml``, strictly parsed.

This package imports the standard library and yaml only, never the rest of the
package (tests/unit/test_layering.py).
"""
