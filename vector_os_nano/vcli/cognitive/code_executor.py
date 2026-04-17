"""CodeExecutor — restricted Python sandbox for LLM-generated code.

Security model:
- AST validation blocks imports (except math) and dunder attribute access.
- Execution uses a stripped __builtins__ dict (no open, exec, eval, __import__).
- set_velocity() calls are clamped to safe velocity limits.
- Thread-based timeout prevents runaway code (signal-free — safe off main thread).
- stdout is captured via io.StringIO.
- Return value is captured when the last statement is an expression.
"""
from __future__ import annotations

import ast
import contextlib
import io
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import math as _math_module

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CodeResult:
    """Immutable result of a sandbox execution."""

    success: bool
    stdout: str
    return_value: Any
    error: str
    duration_sec: float


class _ASTValidator(ast.NodeVisitor):
    """Walk AST and raise ValueError on forbidden constructs."""

    _ALLOWED_IMPORT_MODULES = frozenset({"math"})

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            base = alias.name.split(".")[0]
            if base not in self._ALLOWED_IMPORT_MODULES:
                raise ValueError(
                    f"Import of '{alias.name}' is not allowed in sandbox"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        module = (node.module or "").split(".")[0]
        if module not in self._ALLOWED_IMPORT_MODULES:
            raise ValueError(
                f"Import from '{node.module}' is not allowed in sandbox"
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if node.attr.startswith("__"):
            raise ValueError(
                f"Access to dunder attribute '{node.attr}' is not allowed"
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if node.id.startswith("__"):
            raise ValueError(
                f"Access to dunder name '{node.id}' is not allowed"
            )
        self.generic_visit(node)


def _validate_ast(code: str) -> None:
    """Parse and validate code; raise ValueError on violation."""
    tree = ast.parse(code, mode="exec")
    _ASTValidator().visit(tree)


def _split_last_expr(
    code: str,
) -> tuple[str | None, str | None]:
    """Split code into (body, last_expr) where last_expr is the last expression.

    Returns:
        (body_code, expr_code) — body_code is everything except the last node
        when the last node is an ast.Expr; expr_code is that expression source.
        If the last node is NOT an expression, returns (code, None).
        If code is empty, returns (None, None).
    """
    if not code.strip():
        return None, None

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return code, None

    if not tree.body:
        return None, None

    last = tree.body[-1]
    if not isinstance(last, ast.Expr):
        return code, None

    # Split source lines: body = all lines except last expr's lines
    lines = code.splitlines(keepends=True)
    # ast line numbers are 1-based
    last_start = last.lineno - 1  # 0-indexed

    body_lines = lines[:last_start]
    expr_lines = lines[last_start:]
    body_code = "".join(body_lines) if body_lines else None
    expr_code = "".join(expr_lines)
    return body_code, expr_code


class CodeExecutor:
    """Execute LLM-generated Python code in a restricted sandbox.

    The sandbox:
    - Blocks all imports except ``import math`` / ``from math import ...``.
    - Blocks dunder attribute and name access.
    - Provides a stripped builtins dict with safe functions only.
    - Injects all ``primitives_namespace`` callables (with velocity clamping).
    - Captures stdout via io.StringIO.
    - Enforces a wall-clock timeout via a daemon thread.
    """

    # Velocity safety clamps
    _MAX_VX: float = 1.0
    _MAX_VY: float = 1.0
    _MAX_VYAW: float = 2.0

    def __init__(
        self,
        primitives_namespace: dict[str, Callable],
        timeout_sec: float = 30.0,
    ) -> None:
        self._namespace = primitives_namespace
        self._timeout = timeout_sec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, code: str) -> CodeResult:
        """Execute *code* string in restricted sandbox.

        Steps:
        1. AST validation (fast, no execution).
        2. Build restricted globals.
        3. Split code into body + optional last expression.
        4. Run in daemon thread with timeout.
        5. Return CodeResult.
        """
        start = time.monotonic()

        # Empty code is valid
        if not code.strip():
            return CodeResult(
                success=True,
                stdout="",
                return_value=None,
                error="",
                duration_sec=time.monotonic() - start,
            )

        # AST validation
        try:
            _validate_ast(code)
        except SyntaxError as exc:
            return CodeResult(
                success=False,
                stdout="",
                return_value=None,
                error=f"SyntaxError: {exc}",
                duration_sec=time.monotonic() - start,
            )
        except ValueError as exc:
            return CodeResult(
                success=False,
                stdout="",
                return_value=None,
                error=str(exc),
                duration_sec=time.monotonic() - start,
            )

        # Build sandbox globals
        buf = io.StringIO()

        def _captured_print(*args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("file", buf)
            print(*args, **kwargs)

        _ALLOWED_IMPORT_MODULES = frozenset({"math"})

        def _restricted_import(
            name: str,
            globals: Any = None,  # noqa: A002
            locals: Any = None,  # noqa: A002
            fromlist: tuple = (),
            level: int = 0,
        ) -> Any:
            base = name.split(".")[0]
            if base not in _ALLOWED_IMPORT_MODULES:
                raise ImportError(
                    f"Import of '{name}' is not allowed in sandbox"
                )
            return __import__(name, globals, locals, fromlist, level)

        safe_builtins: dict[str, Any] = {
            "__import__": _restricted_import,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "tuple": tuple,
            "dict": dict,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "sorted": sorted,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "print": _captured_print,
            "isinstance": isinstance,
            "any": any,
            "all": all,
            "map": map,
            "filter": filter,
            "True": True,
            "False": False,
            "None": None,
        }

        globals_dict: dict[str, Any] = {
            "__builtins__": safe_builtins,
            "math": _math_module,
        }
        globals_dict.update(self._wrapped_namespace())

        # Split into body + last expression (for return value capture)
        body_code, expr_code = _split_last_expr(code)

        # Thread-based execution with timeout
        result_holder: dict[str, Any] = {
            "success": False,
            "return_value": None,
            "error": "",
        }

        def _run() -> None:
            try:
                if body_code is not None:
                    exec(compile(body_code, "<sandbox>", "exec"), globals_dict)  # noqa: S102
                if expr_code is not None:
                    with contextlib.redirect_stdout(buf):
                        retval = eval(  # noqa: S307
                            compile(expr_code.strip(), "<sandbox>", "eval"),
                            globals_dict,
                        )
                    result_holder["return_value"] = retval
                result_holder["success"] = True
            except Exception as exc:  # noqa: BLE001
                result_holder["error"] = str(exc)

        # Redirect stdout for the exec body as well
        old_stdout_ref = buf  # already our buffer

        def _run_with_redirect() -> None:
            with contextlib.redirect_stdout(buf):
                _run()

        thread = threading.Thread(target=_run_with_redirect, daemon=True)
        thread.start()
        thread.join(timeout=self._timeout)

        duration = time.monotonic() - start

        if thread.is_alive():
            # Thread is still running — timed out
            return CodeResult(
                success=False,
                stdout=buf.getvalue(),
                return_value=None,
                error=f"timeout: execution exceeded {self._timeout}s",
                duration_sec=duration,
            )

        return CodeResult(
            success=result_holder["success"],
            stdout=buf.getvalue(),
            return_value=result_holder["return_value"],
            error=result_holder["error"],
            duration_sec=duration,
        )

    # ------------------------------------------------------------------
    # Velocity clamping wrapper
    # ------------------------------------------------------------------

    def _wrapped_namespace(self) -> dict[str, Any]:
        """Return primitives namespace with set_velocity clamped to safe limits."""
        ns = dict(self._namespace)
        original_set_velocity = ns.get("set_velocity")
        if original_set_velocity is not None:
            max_vx = self._MAX_VX
            max_vy = self._MAX_VY
            max_vyaw = self._MAX_VYAW

            def clamped_set_velocity(vx: float, vy: float, vyaw: float) -> Any:
                vx = max(-max_vx, min(max_vx, vx))
                vy = max(-max_vy, min(max_vy, vy))
                vyaw = max(-max_vyaw, min(max_vyaw, vyaw))
                return original_set_velocity(vx, vy, vyaw)

            ns["set_velocity"] = clamped_set_velocity
        return ns
