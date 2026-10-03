# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""vector-eval — headless verify-as-eval harness for the dev world.

Runs a list of ``{task, expect?}`` cases through the VGG kernel over the dev
world and uses the kernel's *own* per-step verify predicates as a self-grading
signal (verify-as-eval). A case is green when:

1. the run succeeds, AND
2. its outcome is backed by deterministic evidence (``evidence_passed``), AND
3. an optional ``expect`` predicate holds over the project after execution.

Exit code is non-zero if any case is red, so it slots into CI.

No robot, no hardware. ``ask``-level tools auto-deny by default (an eval must not
silently mutate outside its sandbox); pass ``--allow`` to auto-approve them.

The grading core (``EvalRunner``) takes an injected ``run_task`` callable and an
optional verifier, so it is unit-testable with a mocked backend / fake traces;
``main`` wires the real dev-world engine.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from vector_os_nano.vcli.cognitive.trace_store import evidence_passed


@dataclass(frozen=True)
class EvalCaseResult:
    """Outcome of a single eval case."""

    task: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class EvalReport:
    """Aggregate outcome over a list of cases."""

    results: list[EvalCaseResult]

    @property
    def green(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def all_passed(self) -> bool:
        # Vacuously true for an empty suite; main() surfaces "no cases"
        # separately with a distinct exit code so an empty file is not a
        # silent generic failure.
        return all(r.passed for r in self.results)


class EvalRunner:
    """Grade eval cases by running each task and checking the resulting trace.

    Args:
        run_task: Maps a task string -> ExecutionTrace (decompose + execute).
            Returning None means decomposition declined — counted as a failure.
        verifier: Optional GoalVerifier (over the dev namespace) used to evaluate
            a case's ``expect`` predicate *after* the run. None disables ``expect``.
    """

    def __init__(self, run_task: Callable[[str], Any], verifier: Any = None) -> None:
        self._run_task = run_task
        self._verifier = verifier

    def _oracle_names(self) -> frozenset[str]:
        """Live dev evidence-oracle names (``@oracle(scope="dev")``, rule 3).

        The eval runner is dev-world by construction (module docstring): its oracle
        names are the keys of the SAME dev verify namespace the injected
        ``GoalVerifier`` wraps. When a verifier is present we read ITS namespace
        (never a second copy); otherwise we read ``dev_verify_namespace()``
        directly — never an empty set, which would falsely fail every dev case.
        """
        from vector_os_nano.vcli.cognitive.oracle_registry import evidence_names

        ns = getattr(self._verifier, "_namespace", None)
        if isinstance(ns, dict) and ns:
            return evidence_names(ns, "dev")
        from vector_os_nano.vcli.worlds.dev import dev_verify_namespace

        return evidence_names(dev_verify_namespace(), "dev")

    def run_case(self, case: dict[str, Any]) -> EvalCaseResult:
        task = str(case.get("task", "")).strip()
        expect = case.get("expect")
        if not task:
            return EvalCaseResult(task="", passed=False, detail="empty task")

        try:
            trace = self._run_task(task)
        except Exception as exc:  # noqa: BLE001
            return EvalCaseResult(task=task, passed=False, detail=f"run error: {exc}")

        if trace is None:
            return EvalCaseResult(task=task, passed=False, detail="no trace (decompose declined)")
        if not getattr(trace, "success", False):
            return EvalCaseResult(task=task, passed=False, detail="execution failed")
        if not evidence_passed(trace, self._oracle_names()):
            return EvalCaseResult(task=task, passed=False, detail="no verifiable evidence")
        if expect:
            if self._verifier is None:
                return EvalCaseResult(task=task, passed=False, detail="expect given but no verifier")
            if not self._verifier.verify(str(expect)):
                return EvalCaseResult(task=task, passed=False, detail=f"expect failed: {expect}")
        return EvalCaseResult(task=task, passed=True, detail="ok")

    def run(self, cases: list[dict[str, Any]]) -> EvalReport:
        return EvalReport(results=[self.run_case(c) for c in cases])


# ---------------------------------------------------------------------------
# Real dev-world wiring (used by main; not exercised in unit tests)
# ---------------------------------------------------------------------------


def _load_cases(path: str | Path) -> list[dict[str, Any]]:
    """Load eval cases from a JSON file: a list, or ``{"cases": [...]}``."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "cases" in data:
        data = data["cases"]
    if not isinstance(data, list):
        raise ValueError(
            "eval file must be a JSON list of {task, expect?} (or {\"cases\": [...]})"
        )
    return data


def build_dev_engine(allow_ask: bool = False) -> Any:
    """Construct a headless dev-world VGG engine.

    ``ask``-level tools resolve to deny (default) or always-allow (``allow_ask``).
    Raises SystemExit with a clear message when no API key is configured.
    """
    from vector_os_nano.vcli.backends import create_backend
    from vector_os_nano.vcli.config import resolve_credentials
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.permissions import PermissionContext
    from vector_os_nano.vcli.tools import (
        CategorizedToolRegistry,
        discover_categorized_tools,
    )
    from vector_os_nano.vcli.worlds import DevWorld

    api_key, provider, model, base_url = resolve_credentials()
    if not api_key:
        raise SystemExit(
            "vector-eval: no API key configured — set ANTHROPIC_API_KEY or run "
            "`vector-cli` and /login first."
        )
    backend = create_backend(provider=provider, api_key=api_key, model=model, base_url=base_url)

    registry = CategorizedToolRegistry()
    tools_list, cat_map = discover_categorized_tools()
    for t in tools_list:
        cat = "default"
        for c, names in cat_map.items():
            if t.name in names:
                cat = c
                break
        registry.register(t, category=cat)
    # Dev world: robot/diag/system tools are never advertised or dispatchable.
    for _c in ("robot", "diag", "system"):
        registry.disable_category(_c)

    intent_router = None
    try:
        from vector_os_nano.vcli.intent_router import IntentRouter
        intent_router = IntentRouter()
    except ImportError:
        pass

    engine = VectorEngine(
        backend=backend,
        registry=registry,
        permissions=PermissionContext(),
        intent_router=intent_router,
    )
    resolver = (lambda _n, _p: "a") if allow_ask else (lambda _n, _p: "n")
    engine.init_vgg(
        agent=None,
        skill_registry=None,
        world=DevWorld(),
        tool_permission_resolver=resolver,
    )
    return engine


def _engine_run_task(engine: Any) -> Callable[[str], Any]:
    """Return a run_task that decomposes + executes a task over *engine*.

    Calls the decomposer directly (bypassing the IntentRouter complexity gate)
    so an eval always decomposes and runs the task it was given.
    """

    def run_task(task: str) -> Any:
        tree = engine._goal_decomposer.decompose(task, "")
        return engine.vgg_execute(tree)

    return run_task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vector-eval",
        description="Headless verify-as-eval over the dev world.",
    )
    parser.add_argument(
        "eval_file",
        help='JSON file: a list of {"task", "expect"?} (or {"cases": [...]}).',
    )
    parser.add_argument(
        "--allow",
        action="store_true",
        help="Auto-approve ask-level tools (default: auto-deny).",
    )
    args = parser.parse_args(argv)

    cases = _load_cases(args.eval_file)
    if not cases:
        print("vector-eval: no cases in eval file")
        return 2  # distinct from "a case failed" (1)

    engine = build_dev_engine(allow_ask=args.allow)
    from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
    from vector_os_nano.vcli.worlds.dev import dev_verify_namespace

    verifier = GoalVerifier(dev_verify_namespace())
    runner = EvalRunner(run_task=_engine_run_task(engine), verifier=verifier)
    report = runner.run(cases)

    for r in report.results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"[{mark}] {r.task}  — {r.detail}")
    print(f"\n{report.green}/{report.total} passed")
    return 0 if report.all_passed else 1


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
