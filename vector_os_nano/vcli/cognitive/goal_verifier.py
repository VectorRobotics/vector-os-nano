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

    def verify(self, expression: str) -> bool:
        """Evaluate *expression* in a restricted sandbox.

        Returns ``True`` if the expression evaluates to a truthy value.
        Returns ``False`` on any error (security violation, timeout, exception).
        """
        if not expression or not expression.strip():
            _LOG.warning("GoalVerifier: empty expression")
            return False

        # Dunder check on raw source text
        if "__" in expression:
            _LOG.warning("GoalVerifier: dunder name rejected in expression: %r", expression)
            return False

        # AST safety check
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError:
            # Try statement mode to produce a better error message for blocked constructs
            try:
                stmt_tree = ast.parse(expression, mode="exec")
                for node in ast.walk(stmt_tree):
                    if isinstance(node, _BLOCKED_NODE_TYPES):
                        _LOG.warning(
                            "GoalVerifier: blocked AST node %s in expression: %r",
                            type(node).__name__,
                            expression,
                        )
                        return False
            except SyntaxError:
                pass
            _LOG.warning("GoalVerifier: SyntaxError in expression: %r", expression)
            return False

        # Walk the eval-mode AST and reject blocked node types
        for node in ast.walk(tree):
            if isinstance(node, _BLOCKED_NODE_TYPES):
                _LOG.warning(
                    "GoalVerifier: blocked AST node %s in expression: %r",
                    type(node).__name__,
                    expression,
                )
                return False

        # Compile for eval
        try:
            code = compile(tree, "<verify>", "eval")
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("GoalVerifier: compile error for %r: %s", expression, exc)
            return False

        # Build execution globals
        exec_globals: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS,
        }
        exec_globals.update(self._namespace)

        # Evaluate with timeout
        return self._eval_with_timeout(code, exec_globals)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _eval_with_timeout(
        self,
        code: Any,
        exec_globals: dict[str, Any],
    ) -> bool:
        """Evaluate compiled *code* with a timeout. Returns False on failure."""
        use_signal = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()

        if use_signal:
            return self._eval_signal_timeout(code, exec_globals)
        return self._eval_thread_timeout(code, exec_globals)

    def _eval_signal_timeout(
        self,
        code: Any,
        exec_globals: dict[str, Any],
    ) -> bool:
        old_handler = signal.signal(signal.SIGALRM, _signal_handler)
        signal.alarm(_TIMEOUT_SECONDS)
        try:
            result = eval(code, exec_globals)  # noqa: S307
            return bool(result)
        except _TimeoutError:
            _LOG.warning("GoalVerifier: expression timed out")
            return False
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("GoalVerifier: runtime error: %s", exc)
            return False
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def _eval_thread_timeout(
        self,
        code: Any,
        exec_globals: dict[str, Any],
    ) -> bool:
        """Fallback timeout using a threading.Timer (for Windows / non-main threads)."""
        result_box: list[Any] = [None]
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
            return False

        if error_box[0] is not None:
            _LOG.warning("GoalVerifier: runtime error: %s", error_box[0])
            return False

        return bool(result_box[0])
