# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""GoalVerifier — safe sandbox for evaluating SubGoal verify expressions.

Security model:
- Only the caller-supplied primitives_namespace functions are available.
- A restricted set of safe builtins is injected (no open, exec, eval, etc.).
- AST is checked before eval: Import, ImportFrom, Assign, AugAssign,
  FunctionDef, ClassDef, Exec nodes are all blocked.
- Any name containing "__" (dunder) in the source text is rejected.
- Execution is bounded by a 5-second timeout (signal.alarm on Unix,
  threading.Timer fallback on Windows).
- Any exception causes the method to return False and log a warning.
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
# AST node types that are unconditionally blocked
# ---------------------------------------------------------------------------

_BLOCKED_NODE_TYPES = (
    ast.Import,
    ast.ImportFrom,
    ast.Assign,
    ast.AugAssign,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)


# ---------------------------------------------------------------------------
# Timeout helpers
# ---------------------------------------------------------------------------

_TIMEOUT_SECONDS = 5

# Sentinel returned by the eval helpers when evaluation failed (security
# violation, timeout, or runtime error). Distinct from any legitimate Python
# value an expression could produce.
_EVAL_FAILED = object()


class _TimeoutError(Exception):
    """Raised when a verify expression exceeds the time limit."""


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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, expression: str, extra_ns: "dict[str, Any] | None" = None) -> bool:
        """Evaluate *expression* in a restricted sandbox.

        Returns ``True`` if the expression evaluates to a truthy value.
        Returns ``False`` on any error (security violation, timeout, exception).

        Thin wrapper over :meth:`evaluate` — returns only the bool so every
        existing caller and the evidence gate behave byte-identically.
        """
        return self.evaluate(expression, extra_ns=extra_ns)[0]

    def evaluate(
        self, expression: str, extra_ns: "dict[str, Any] | None" = None
    ) -> tuple[bool, Any]:
        """Evaluate *expression* in the same restricted sandbox as :meth:`verify`.

        Returns ``(bool(result), result)`` — the truthiness AND the raw value the
        expression produced. On any failure (empty/dunder/blocked-AST/syntax/
        compile/timeout/runtime error) returns ``(False, None)``. The raw-value
        path uses the IDENTICAL sandbox, AST checks, safe builtins, and timeout as
        the boolean path — nothing is loosened.

        ``extra_ns`` (backlog #3) is a PER-CALL namespace overlay — e.g. the
        executor injecting ``step_output`` bound to the current step's own
        structured output, so a verify can consume what THIS step observed
        (Rule 4) instead of re-querying a separate oracle. The overlay lives
        only in this call's exec globals (``self._namespace`` is never
        mutated), and the AST checks / safe builtins / timeout are identical —
        the sandbox is not loosened.
        """
        if not expression or not expression.strip():
            _LOG.warning("GoalVerifier: empty expression")
            return False, None

        # Dunder check on raw source text
        if "__" in expression:
            _LOG.warning("GoalVerifier: dunder name rejected in expression: %r", expression)
            return False, None

        # AST safety check. LLM-authored expressions can carry sloppy escape
        # sequences (e.g. '\.'); they parse fine (the escape becomes a literal)
        # but ast.parse would emit a SyntaxWarning to '<unknown>' on the user's
        # console. Suppress that noise — it is model output, not a code defect.
        import warnings  # noqa: PLC0415
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(expression, mode="eval")
        except SyntaxError:
            # Try statement mode to produce a better error message for blocked constructs
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    stmt_tree = ast.parse(expression, mode="exec")
                for node in ast.walk(stmt_tree):
                    if isinstance(node, _BLOCKED_NODE_TYPES):
                        _LOG.warning(
                            "GoalVerifier: blocked AST node %s in expression: %r",
                            type(node).__name__,
                            expression,
                        )
                        return False, None
            except SyntaxError:
                pass
            _LOG.warning("GoalVerifier: SyntaxError in expression: %r", expression)
            return False, None

        # Walk the eval-mode AST and reject blocked node types
        for node in ast.walk(tree):
            if isinstance(node, _BLOCKED_NODE_TYPES):
                _LOG.warning(
                    "GoalVerifier: blocked AST node %s in expression: %r",
                    type(node).__name__,
                    expression,
                )
                return False, None

        # Compile for eval
        try:
            code = compile(tree, "<verify>", "eval")
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("GoalVerifier: compile error for %r: %s", expression, exc)
            return False, None

        # Build execution globals
        exec_globals: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS,
        }
        exec_globals.update(self._namespace)
        if extra_ns:
            exec_globals.update(extra_ns)

        # Evaluate with timeout — returns the raw value or _EVAL_FAILED.
        raw = self._eval_with_timeout(code, exec_globals)
        if raw is _EVAL_FAILED:
            return False, None
        return bool(raw), raw

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
        signal.alarm(_TIMEOUT_SECONDS)
        try:
            return eval(code, exec_globals)  # noqa: S307
        except _TimeoutError:
            _LOG.warning("GoalVerifier: expression timed out")
            return _EVAL_FAILED
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("GoalVerifier: runtime error: %s", exc)
            return _EVAL_FAILED
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def _eval_thread_timeout(
        self,
        code: Any,
        exec_globals: dict[str, Any],
    ) -> Any:
        """Fallback timeout using a threading.Timer (for Windows / non-main threads)."""
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
            _LOG.warning("GoalVerifier: expression timed out (thread fallback)")
            return _EVAL_FAILED

        if error_box[0] is not None:
            _LOG.warning("GoalVerifier: runtime error: %s", error_box[0])
            return _EVAL_FAILED

        return result_box[0]
