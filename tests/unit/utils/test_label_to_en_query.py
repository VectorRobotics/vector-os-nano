"""Unit tests for label_to_en_query helper — T5 Deliverable A.

7 tests covering:
  - Pure English passthrough
  - CN color + CN noun mapping
  - Possessive '的' stripping
  - Empty / None / whitespace → None
  - Mixed CN + EN
  - Unknown CN passes through (possessive stripped, no map hit)
  - 'all objects' passthrough

No MuJoCo, no network.  Import time < 0.1 s.
"""
from __future__ import annotations

import pytest

from vector_os_nano.skills.utils import label_to_en_query


# ---------------------------------------------------------------------------
# Test 1 — pure English passthrough
# ---------------------------------------------------------------------------


def test_pure_english_passthrough():
    """Pure English input is returned lowercase, whitespace-collapsed."""
    assert label_to_en_query("blue bottle") == "blue bottle"


# ---------------------------------------------------------------------------
# Test 2 — CN color + CN noun mapped
# ---------------------------------------------------------------------------


def test_cn_color_and_noun_mapped():
    """'蓝色瓶子' → color normalised + noun mapped → 'blue bottle'."""
    assert label_to_en_query("蓝色瓶子") == "blue bottle"


# ---------------------------------------------------------------------------
# Test 3 — possessive '的' stripped
# ---------------------------------------------------------------------------


def test_possessive_de_stripped():
    """'红色的杯子' → '的' stripped, then color+noun mapped → 'red cup'."""
    assert label_to_en_query("红色的杯子") == "red cup"


# ---------------------------------------------------------------------------
# Test 4 — empty / None / whitespace → None
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_input", ["", None, "   "])
def test_empty_returns_none(bad_input):
    """Empty string, None, or whitespace-only → None."""
    assert label_to_en_query(bad_input) is None


# ---------------------------------------------------------------------------
# Test 5 — mixed CN + EN
# ---------------------------------------------------------------------------


def test_mixed_cn_en_works():
    """'blue 瓶子' → EN color kept, CN noun mapped → 'blue bottle'."""
    assert label_to_en_query("blue 瓶子") == "blue bottle"


# ---------------------------------------------------------------------------
# Test 6 — unknown CN passes through (possessive stripped, no map hit)
# ---------------------------------------------------------------------------


def test_unknown_cn_passes_through_as_is():
    """'奇怪的东西': '的' stripped → '奇怪东西'; no color/noun map hit → '奇怪东西'."""
    result = label_to_en_query("奇怪的东西")
    assert result == "奇怪东西"


# ---------------------------------------------------------------------------
# Test 7 — 'all objects' passthrough
# ---------------------------------------------------------------------------


def test_all_objects_passthrough():
    """'all objects' passes through unchanged (no CN tokens)."""
    assert label_to_en_query("all objects") == "all objects"
