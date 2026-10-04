# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""INT2 — a Ctrl-C that lands while the MPC gait solves its QP still reaches the REPL.

OSQP (casadi's conic plugin) installs its own SIGINT handler for the length of every
solve and swallows the signal: it prints "Solver interrupted" and the solve fails, so
a Ctrl-C typed while the go2 walks on the MPC gait was lost whenever it landed
mid-solve (E5's "Ctrl-C mid-navigate": the navigate ran on, 'Interrupted' missing).
"""
from __future__ import annotations

import os
import signal
import threading
import time

import pytest

ca = pytest.importorskip("casadi")
pytest.importorskip("pinocchio")

from vector_os_nano.third_party.convex_mpc.centroidal_mpc import CentroidalMPC  # noqa: E402


@pytest.fixture
def sigints():
    got: list[float] = []
    old = signal.signal(signal.SIGINT, lambda *_: got.append(time.monotonic()))
    yield got
    signal.signal(signal.SIGINT, old)


def _wait(got: list, sec: float) -> bool:
    end = time.monotonic() + sec
    while not got and time.monotonic() < end:
        time.sleep(0.01)
    return bool(got)


def _mpc(solver, a: dict) -> CentroidalMPC:
    m = CentroidalMPC.__new__(CentroidalMPC)  # the solve step only, no robot model
    m.solver, m.H_const, m.x_prev = solver, a["h"], None
    m._update_sparse_matrix = lambda traj: [a["g"], a["a"], a["lba"], a["uba"]]
    m._compute_bounds = lambda traj: [a["lbx"], a["ubx"]]
    return m


class _FailingSolver:
    def __init__(self, status: str) -> None:
        self.status = status

    def __call__(self, **kw):
        raise RuntimeError("Problem in osqp_solve")

    def stats(self) -> dict:
        return {"return_status": self.status}


@pytest.mark.parametrize("status, handed_back", [("interrupted", True), ("primal infeasible", False)])
def test_only_a_swallowed_ctrl_c_is_handed_back(sigints, status, handed_back) -> None:
    m = _mpc(_FailingSolver(status), dict.fromkeys(("h", "g", "a", "lba", "uba", "lbx", "ubx"), 0))
    with pytest.raises(RuntimeError):  # the tick still fails (PD fallback), loudly
        m.solve_QP(None, None)
    assert _wait(sigints, 1.0 if handed_back else 0.2) is handed_back


def test_real_osqp_mid_solve_ctrl_c_reaches_the_handler(sigints) -> None:
    np = pytest.importorskip("numpy")
    n, rng = 1500, np.random.default_rng(0)
    m_ = rng.standard_normal((n, n)) * (rng.random((n, n)) < 0.01)
    h = ca.sparsify(ca.DM(m_ @ m_.T + np.eye(n)))
    a = ca.sparsify(ca.DM(rng.standard_normal((n, n)) * (rng.random((n, n)) < 0.01)))
    solver = ca.conic("S", "osqp", {"h": h.sparsity(), "a": a.sparsity()}, {"osqp": {
        "verbose": False, "max_iter": 200000, "eps_abs": 1e-12, "eps_rel": 1e-12,
        "check_termination": 1}})
    m = _mpc(solver, dict(h=h, g=ca.DM.ones(n), a=a, lba=-1, uba=1, lbx=-ca.inf, ubx=ca.inf))
    t0 = time.monotonic()
    m.solve_QP(None, None)
    full = time.monotonic() - t0  # setup (factorisation) + the OSQP iterations
    for frac in (0.7, 0.8, 0.6, 0.75, 0.65, 0.85, 0.55, 0.9):
        m.x_prev = None  # no warm start: every attempt takes the same time
        sigints.clear()
        th = threading.Thread(target=_solve_quietly, args=(m,))
        th.start()
        time.sleep(frac * full)
        os.kill(os.getpid(), signal.SIGINT)
        th.join()
        if solver.stats().get("return_status") != "interrupted":
            continue  # it landed outside osqp_solve (Python saw it directly)
        assert _wait(sigints, 1.0), "OSQP swallowed the Ctrl-C (the REPL never saw it)"
        return
    pytest.skip("no SIGINT landed inside an OSQP solve on this machine")


def _solve_quietly(m: CentroidalMPC) -> None:
    try:
        m.solve_QP(None, None)
    except RuntimeError:
        pass
