"""Tests for unit and label parsing.

These are the highest-value tests in the project. Every dimension in the output
traces back to this module, and a parsing error here is silent — it produces a
building that is confidently the wrong size rather than an error. The area-suffix
case below is a real bug that shipped and was caught only by a cross-check.
"""

from __future__ import annotations

import math

import pytest

from atlas.units import (
    MM_PER_FOOT,
    MM_PER_INCH,
    format_feet_inches,
    infer_scale_from_extents,
    mm_to_sqft,
    parse_area_sqft,
    parse_dimension_pair,
    parse_feet_inches,
)


class TestFeetInches:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("11'0\"", 11 * MM_PER_FOOT),
            ("7'9\"", 7 * MM_PER_FOOT + 9 * MM_PER_INCH),
            ("11'", 11 * MM_PER_FOOT),
            ("5'3\"", 5 * MM_PER_FOOT + 3 * MM_PER_INCH),
            # Curly quotes and prime marks all appear in real drawings.
            ("11’6”", 11 * MM_PER_FOOT + 6 * MM_PER_INCH),
            ("11′6″", 11 * MM_PER_FOOT + 6 * MM_PER_INCH),
            ("11'-6\"", 11 * MM_PER_FOOT + 6 * MM_PER_INCH),
            ("13 ft 6 in", 13 * MM_PER_FOOT + 6 * MM_PER_INCH),
        ],
    )
    def test_parses(self, text: str, expected: float) -> None:
        got = parse_feet_inches(text)
        assert got is not None
        assert got == pytest.approx(expected, abs=0.1)

    def test_rejects_impossible_inches(self) -> None:
        # 14 inches is a typo, not a measurement; guessing would be worse than
        # declining to parse.
        assert parse_feet_inches("11'14\"") is None

    def test_no_measurement(self) -> None:
        assert parse_feet_inches("BEDROOM") is None
        assert parse_feet_inches("") is None


class TestDimensionPair:
    def test_indian_plan_label(self) -> None:
        d = parse_dimension_pair('BEDROOM\n11\'0"X13\'6"')
        assert d is not None
        assert d.imperial is True
        assert d.width_mm == pytest.approx(3352.8, abs=0.1)
        assert d.depth_mm == pytest.approx(4114.8, abs=0.1)
        # 11 x 13.5 ft = 148.5 sqft
        assert d.area_sqft == pytest.approx(148.5, abs=0.5)

    @pytest.mark.parametrize("sep", ["X", "x", "×", "*"])
    def test_separators(self, sep: str) -> None:
        assert parse_dimension_pair(f"9'0\"{sep}8'0\"") is not None

    def test_metric_millimetres(self) -> None:
        d = parse_dimension_pair("3350 X 4100")
        assert d is not None
        assert d.imperial is False
        assert d.width_mm == pytest.approx(3350)
        assert d.depth_mm == pytest.approx(4100)

    def test_metric_metres_with_unit(self) -> None:
        d = parse_dimension_pair("3.35 x 4.10 m")
        assert d is not None
        assert d.width_mm == pytest.approx(3350)
        assert d.depth_mm == pytest.approx(4100)

    def test_bare_small_numbers_read_as_metres(self) -> None:
        # "3.35 x 4.10" cannot be millimetres; a 3 mm room is not a room.
        d = parse_dimension_pair("3.35 x 4.10")
        assert d is not None
        assert d.width_mm == pytest.approx(3350)

    def test_rejects_implausible(self) -> None:
        assert parse_dimension_pair("2 X 3") is None or parse_dimension_pair("2 X 3").width_mm >= 300
        assert parse_dimension_pair("90000 X 90000") is None

    def test_no_dimension(self) -> None:
        assert parse_dimension_pair("KITCHEN") is None
        assert parse_dimension_pair("") is None


class TestArea:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("CARPET AREA: 1056 SQFT", 1056.0),
            ("CARPET AREA:1358 SQFT", 1358.0),
            ("CARPET AREA : 1,056 SQ.FT.", 1056.0),
            ("AREA 776 SFT", 776.0),
            ("710 sq ft", 710.0),
            ("1420.4 sqft", 1420.4),
        ],
    )
    def test_parses(self, text: str, expected: float) -> None:
        assert parse_area_sqft(text) == pytest.approx(expected)

    def test_regression_four_digit_area_not_truncated(self) -> None:
        """A four-digit area must not be read as its last three digits.

        This is the bug this test exists for: 'CARPET AREA:1358 SQFT' parsed as
        358, understating the flat by a factor of nearly four and then failing the
        area cross-check with a misleading message.
        """
        assert parse_area_sqft("CARPET AREA:1358 SQFT") == 1358.0
        assert parse_area_sqft("2450 SQFT") == 2450.0
        assert parse_area_sqft("12500 SQFT") == 12500.0

    def test_no_area(self) -> None:
        assert parse_area_sqft("3 BHK") is None
        assert parse_area_sqft("") is None


class TestScaleInference:
    @pytest.mark.parametrize(
        "span,expected_mm_per_unit",
        [
            (12_000.0, 1.0),      # millimetres
            (1_200.0, 10.0),      # centimetres
            (12.0, 1000.0),       # metres
        ],
    )
    def test_infers_from_extent(self, span: float, expected_mm_per_unit: float) -> None:
        mm, why = infer_scale_from_extents(span, span)
        assert mm == expected_mm_per_unit
        assert why  # the reasoning must always be reported to the user

    def test_degenerate(self) -> None:
        mm, why = infer_scale_from_extents(0.0, 0.0)
        assert mm == 1.0
        assert "degenerate" in why


class TestRoundTrip:
    @pytest.mark.parametrize("feet,inches", [(11, 0), (7, 9), (13, 6), (5, 3)])
    def test_format_inverts_parse(self, feet: int, inches: int) -> None:
        text = f"{feet}'{inches}\""
        mm = parse_feet_inches(text)
        assert mm is not None
        assert format_feet_inches(mm) == text

    def test_area_conversion(self) -> None:
        # A 10 ft x 10 ft room is 100 sqft.
        mm2 = (10 * MM_PER_FOOT) ** 2
        assert mm_to_sqft(mm2) == pytest.approx(100.0, abs=0.01)


def test_furniture_shrinks_to_fit_but_keeps_its_height() -> None:
    """A tight room gets a smaller version of an item, not no item.

    Real furniture comes in sizes: a 1 m desk is still a desk. Nine of seventeen placement
    failures on the demo flat were bedroom desks, chairs and bedside tables rejected in
    rooms that would take a smaller one comfortably.

    Height must not scale with the plan dimensions. A work surface, a seat and a counter are
    set by the human body, not by how much floor happens to be spare, so a shrunk desk is
    narrower and shallower at exactly the same height.
    """
    from atlas.furnish.catalog import CATALOG
    from atlas.furnish.solver import _scaled, _shrink_steps

    desk = CATALOG["desk"]
    assert desk.min_scale < 1.0, "a desk is made in more than one size"

    steps = _shrink_steps(desk.min_scale)
    assert steps[0] == 1.0, "full size must be tried first"
    assert steps == sorted(steps, reverse=True), "largest first, so a full fit always wins"
    assert min(steps) >= desk.min_scale - 1e-9

    small = _scaled(desk, 0.7)
    assert small.width_mm == desk.width_mm * 0.7
    assert small.depth_mm == desk.depth_mm * 0.7
    assert small.height_mm == desk.height_mm, "height is set by the body, not the floor"
    assert small.clearance_front_mm == desk.clearance_front_mm * 0.7

    # Sanitaryware does not come smaller in any meaningful sense: a WC has fixed
    # connections and a 700 mm tray is not a shower.
    for fixed in ("wc-wall-hung", "shower-enclosure"):
        if fixed in CATALOG:
            assert CATALOG[fixed].min_scale == 1.0, f"{fixed} must not be shrunk to fit"
