# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Batch 3 (campaign #4) — pure grading helpers of the E2E GUI smoke script.

The smoke script's verdicts must be deterministic transcript facts; these
tests pin the grading contract (ok > honest > fail; segment diffing is
scroll-safe; outcome counting drives the wait loop).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "e2e_gui_smoke.py"
spec = importlib.util.spec_from_file_location("_smoke", _SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class TestGradeSegment:
    def test_ok_pattern_wins(self):
        seg = "Steps: [PASS] nav | verify geodesic\nOutcome: [PASS] 1/1"
        assert smoke.grade_segment(seg, [r"Outcome: \[PASS\]"],
                                   [r"bad_params"]) == "ok"

    def test_honest_failure_detected(self):
        seg = "[FAIL] nav | -- navigate_to requires a label OR numeric x and y"
        assert smoke.grade_segment(seg, [r"Outcome: \[PASS\]"],
                                   [r"requires a label"]) == "honest"

    def test_silent_wrong_is_fail(self):
        seg = "something unrecognizable happened"
        assert smoke.grade_segment(seg, [r"Outcome: \[PASS\]"],
                                   [r"bad_params"]) == "fail"

    def test_ok_beats_honest_when_both_present(self):
        seg = "bad_params earlier... then Outcome: [PASS] 1/1"
        assert smoke.grade_segment(seg, [r"Outcome: \[PASS\]"],
                                   [r"bad_params"]) == "ok"


class TestSegmentDiff:
    def test_appended_text_extracted(self):
        prev = "line1\nline2\n"
        cur = prev + "line3\nOutcome: [PASS]\n"
        assert smoke.new_segment(prev, cur) == "line3\nOutcome: [PASS]\n"

    def test_scrolled_pane_falls_back_to_full(self):
        assert smoke.new_segment("old text gone", "fresh pane") == "fresh pane"


class TestOutcomeCount:
    def test_counts_both_pass_and_fail(self):
        t = "Outcome: [PASS] 1/1\nnoise\nOutcome: [FAIL] 0/1\n"
        assert smoke.count_outcomes(t) == 2

    def test_zero_on_boot_banner(self):
        assert smoke.count_outcomes("Vector OS Nano\nvector>") == 0
