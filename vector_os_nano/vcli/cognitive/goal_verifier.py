# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""GoalVerifier — safe sandbox for evaluating SubGoal verify expressions.

Security model:
- The expression must fit an AST ALLOWLIST (``_ALLOWED_NODES``): boolean logic,
  comparisons, calls on a bare NAME, names, literals, tuple/list literals and
  subscripts. No attribute access, no comprehensions / lambdas /
  walrus / f-strings / arithmetic — so there is no path to frames, ``__globals__``
  or the real builtins, and no ``'x' * 10**9`` style allocation bomb.
- Only the caller-supplied primitives_namespace functions plus a restricted set
  of safe builtins (no open, exec, eval, getattr, ...) are resolvable.
- Any "__" (dunder) in the source text is rejected.
- Execution is bounded by a 5-second timeout (SIGALRM on the main thread, a
  watchdog thread elsewhere). A thread-path timeout cannot be cancelled, so the
  verifier refuses to evaluate anything else while that evaluation still runs.
- Any exception (including the truth test of the result) returns False.
"""
from __future__ import annotations

import ast
import logging
import signal
import threading
from typing import Any, Callable

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Restricted built-ins allowed inside verify expressions
# ---------------------------------------------------------------------------

_SAFE_BUILTINS: dict[str, Any] = {
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "tuple": tuple,
    "abs": abs,
    "min": min,
    "max": max,
    "True": True,
    "False": False,
    "None": None,
    "isinstance": isinstance,
    "any": any,
    "all": all,
}

# ---------------------------------------------------------------------------
# AST allowlist — every node of a verify expression must be one of these
# ---------------------------------------------------------------------------

_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp, ast.And, ast.Or,
    ast.UnaryOp, ast.Not, ast.USub, ast.UAdd,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.Is, ast.IsNot,  # harmless to evaluate; never evidence (evidence_classifier)
    ast.Call, ast.keyword, ast.Name, ast.Load,
    ast.Constant, ast.Tuple, ast.List, ast.Subscript, ast.Slice,
)


def _grammar_violation(tree: ast.AST) -> str | None:
    """The first construct outside the verify grammar, or None if *tree* fits it."""
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return type(node).__name__
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            return "call on a non-name"
        if isinstance(node, ast.keyword) and node.arg is None:
            return "**kwargs"
    return None


# ---------------------------------------------------------------------------
# Timeout helpers
# ---------------------------------------------------------------------------

_TIMEOUT_SECONDS = 5

# Sentinel returned by the eval helpers when evaluation failed (security
# violation, timeout, or runtime error). Distinct from any legitimate Python
# value an expression could produce.
_EVAL_FAILED = object()


class _TimeoutError(BaseException):
    """Raised when a verify expression exceeds the time limit.

    A BaseException so an oracle's own ``except Exception`` cannot swallow it.
    """


def _signal_handler(signum: int, frame: object) -> None:  # noqa: ARG001
    raise _TimeoutError("verify expression timed out")


class GoalVerifier:
    """Evaluate SubGoal verify expressions in a restricted Python sandbox."""

    def __init__(self, primitives_namespace: dict[str, Callable]) -> None:
        """Initialise with a mapping of allowed function names to callables.

        Only the functions listed in *primitives_namespace* (plus the safe
        built-ins) will be resolvable inside verify expressions.
        """
        self._namespace: dict[str, Any] = dict(primitives_namespace)
        # A thread-path evaluation that outlived its timeout (it cannot be killed).
        self._runaway: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, expression: str) -> bool:
        """Evaluate *expression* in a restricted sandbox.

        Returns ``True`` if the expression evaluates to a truthy value.
        Returns ``False`` on any error (security violation, timeout, exception).

        Thin wrapper over :meth:`evaluate` — returns only the bool so every
        existing caller and the evidence gate behave byte-identically.
        """
        return self.evaluate(expression)[0]

    def evaluate(self, expression: str) -> tuple[bool, Any]:
        """Evaluate *expression* in the same restricted sandbox as :meth:`verify`.

        Returns ``(bool(result), result)`` — the truthiness AND the raw value the
        expression produced. On any failure (empty/dunder/blocked-AST/syntax/
        compile/timeout/runtime error) returns ``(False, None)``. The raw-value
        path uses the IDENTICAL sandbox, AST checks, safe builtins, and timeout as
        the boolean path — nothing is loosened.
        """
        if not expression or not expression.strip():
            _LOG.warning("GoalVerifier: empty expression")
            return False, None

        if self._runaway is not None and self._runaway.is_alive():
            _LOG.warning("GoalVerifier: a timed-out evaluation is still running; refusing %r", expression)
            return False, None

        # Dunder check on raw source text
        if "__" in expression:
            _LOG.warning("GoalVerifier: dunder name rejected in expression: %r", expression)
            return False, None

        # LLM-authored expressions can carry sloppy escape sequences (e.g. '\.');
        # they parse fine (the escape becomes a literal) but ast.parse would emit a
        # SyntaxWarning to '<unknown>' on the user's console. Suppress that noise.
        import warnings  # noqa: PLC0415
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(expression, mode="eval")
        except SyntaxError:
            _LOG.warning("GoalVerifier: SyntaxError in expression: %r", expression)
            return False, None

        bad = _grammar_violation(tree)
        if bad is not None:
            _LOG.warning("GoalVerifier: %s is outside the verify grammar: %r", bad, expression)
            return False, None

        try:
            code = compile(tree, "<verify>", "eval")
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("GoalVerifier: compile error for %r: %s", expression, exc)
            return False, None

        exec_globals: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
        exec_globals.update(self._namespace)

        # Evaluate with timeout — returns the raw value or _EVAL_FAILED.
        raw = self._eval_with_timeout(code, exec_globals)
        if raw is _EVAL_FAILED:
            return False, None
        try:
            return bool(raw), raw
        except Exception as exc:  # noqa: BLE001 — e.g. an ambiguous numpy truth value
            _LOG.warning("GoalVerifier: result of %r has no truth value: %s", expression, exc)
            return False, None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _eval_with_timeout(
        self,
        code: Any,
        exec_globals: dict[str, Any],
    ) -> Any:
        """Evaluate compiled *code* with a timeout.

        Returns the raw expression value, or the ``_EVAL_FAILED`` sentinel on
        timeout / runtime error.
        """
        use_signal = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()

        if use_signal:
            return self._eval_signal_timeout(code, exec_globals)
        return self._eval_thread_timeout(code, exec_globals)

    def _eval_signal_timeout(
        self,
        code: Any,
        exec_globals: dict[str, Any],
    ) -> Any:
        old_handler = signal.signal(signal.SIGALRM, _signal_handler)
        prior_alarm = signal.alarm(_TIMEOUT_SECONDS)
        try:
            try:
                return eval(code, exec_globals)  # noqa: S307
            finally:
                signal.alarm(0)  # disarm before anything else can be interrupted
        except _TimeoutError:
            _LOG.warning("GoalVerifier: expression timed out")
            return _EVAL_FAILED
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("GoalVerifier: runtime error: %s", exc)
            return _EVAL_FAILED
        finally:
            signal.signal(signal.SIGALRM, old_handler)
            if prior_alarm:
                signal.alarm(prior_alarm)

    def _eval_thread_timeout(
        self,
        code: Any,
        exec_globals: dict[str, Any],
    ) -> Any:
        """Fallback timeout on a watchdog thread (non-main threads / no SIGALRM).

        A thread cannot be cancelled: on timeout the evaluation keeps running, so it
        is remembered and every later ``evaluate`` refuses until it finishes — it can
        never run concurrently with the next step's verify.
        """
        result_box: list[Any] = [_EVAL_FAILED]
        error_box: list[Exception | None] = [None]

        def _target() -> None:
            try:
                result_box[0] = eval(code, exec_globals)  # noqa: S307
            except Exception as exc:  # noqa: BLE001
                error_box[0] = exc

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=_TIMEOUT_SECONDS)

        if thread.is_alive():
            self._runaway = thread
            _LOG.warning("GoalVerifier: expression timed out (thread fallback)")
            return _EVAL_FAILED

        if error_box[0] is not None:
            _LOG.warning("GoalVerifier: runtime error: %s", error_box[0])
            return _EVAL_FAILED

        return result_box[0]
