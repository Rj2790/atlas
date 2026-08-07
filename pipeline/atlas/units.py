"""Unit handling. Everything inside Atlas is millimetres.

Getting this wrong is the single most common way a floor-plan-to-3D pipeline
produces a building where a door is 1.4 m tall. Three separate unit systems meet
in this project and each lies in its own way:

* **DXF header ``$INSUNITS``** declares the drawing's units, but is frequently
  left at 0 (*unitless*) even in files that are plainly in millimetres. When it is
  unset we infer from the drawing extents — a residential floor plate is ~10-30 m
  across, so extents of 12,000 mean millimetres and extents of 12 mean metres.

* **Printed dimension text** on Indian residential plans is imperial
  feet-and-inches (``11'0"X13'6"``) even though the CAD geometry is metric. This
  is the *designer's stated intent* and takes priority over measured geometry when
  the two disagree slightly, because it is what the buyer is sold.

* **Stated areas** (``CARPET AREA: 1056 SQFT``) give an independent check on the
  whole plan. If summed room areas miss the stated carpet area badly, the
  extraction is wrong and we should say so rather than ship a bad model.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

__all__ = [
    "MM_PER_INCH",
    "MM_PER_FOOT",
    "SQFT_PER_SQM",
    "INSUNITS_TO_MM",
    "Dimension",
    "parse_feet_inches",
    "parse_dimension_pair",
    "parse_area_sqft",
    "infer_scale_from_extents",
    "mm_to_sqft",
    "format_feet_inches",
]

MM_PER_INCH = 25.4
MM_PER_FOOT = 304.8
SQFT_PER_SQM = 10.763910416709722

# DXF $INSUNITS enumeration -> millimetres per unit. Only the values that occur in
# architectural practice are mapped; the rest would be a bug worth surfacing.
INSUNITS_TO_MM: dict[int, float] = {
    0: 0.0,  # unitless -> must be inferred
    1: MM_PER_INCH,
    2: MM_PER_FOOT,
    4: 1.0,  # millimetres
    5: 10.0,  # centimetres
    6: 1000.0,  # metres
    8: MM_PER_INCH / 1000.0,  # microinches
    9: MM_PER_INCH / 1000.0,  # mils
    10: MM_PER_FOOT * 3.0,  # yards
    13: 1e-3,  # micrometres
    14: 100.0,  # decimetres
    15: 10_000.0,  # decametres
    16: 100_000.0,  # hectometres
}


# 11'0"X13'6"  /  11'-0" x 13'-6"  /  11' X 13'6  /  11'0” X 13'6”
# Curly quotes, prime/double-prime, optional hyphen and optional inches part all
# appear in real drawings, sometimes within one file.
_FEET_INCHES = r"""
    (?P<feet>\d{1,3})              # feet
    \s*(?:['’′]|ft\.?|feet)        # foot mark
    \s*-?\s*
    (?:                            # optional inches
        (?P<inches>\d{1,2})
        (?:\s*(?:["”″]|''|in\.?|inch(?:es)?))?
    )?
"""
_FEET_INCHES_RE = re.compile(_FEET_INCHES, re.VERBOSE)

_PAIR_RE = re.compile(
    _FEET_INCHES + r"""\s*[xX×*]\s*""" + _FEET_INCHES.replace("feet", "feet2").replace("inches", "inches2"),
    re.VERBOSE,
)

# "CARPET AREA: 1056 SQFT", "CARPET AREA : 1,056 SQ.FT.", "AREA 1056 SFT"
#
# The lookbehind and the alternation order both matter. Written as
# ``\d{1,3}(?:,\d{3})*`` the pattern cannot match all of "1358", so the engine
# advances and matches the *suffix* "358" — silently reporting a 1358 sqft flat as
# 358 sqft, which then fails the area cross-check for the wrong reason. The
# comma-grouped alternative is tried first, plain runs of digits second, and
# ``(?<![\d.])`` forbids starting mid-number.
_AREA_RE = re.compile(
    r"(?<![\d.])(?P<value>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    r"(?:sq\.?\s*ft\.?|sqft|sft|s\.f\.|ft2|ft²)",
    re.IGNORECASE,
)

# Metric fallback: "3350 X 4100" or "3.35 x 4.10 m"
_METRIC_PAIR_RE = re.compile(
    r"(?P<a>\d{1,5}(?:\.\d+)?)\s*(?P<au>mm|cm|m)?\s*[xX×]\s*"
    r"(?P<b>\d{1,5}(?:\.\d+)?)\s*(?P<bu>mm|cm|m)?",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Dimension:
    """A stated room dimension, normalised to millimetres."""

    width_mm: float
    depth_mm: float
    source_text: str
    imperial: bool
    """True when parsed from feet-and-inches, which implies the plan is sold in
    imperial and we should label the UI that way."""

    @property
    def area_mm2(self) -> float:
        return self.width_mm * self.depth_mm

    @property
    def area_sqft(self) -> float:
        return mm_to_sqft(self.area_mm2)

    def as_tuple(self) -> tuple[float, float]:
        return (self.width_mm, self.depth_mm)


def parse_feet_inches(text: str) -> float | None:
    """Parse a single feet-and-inches measurement to millimetres.

    ``"11'0\\""`` -> 3352.8, ``"7'9\\""`` -> 2362.2, ``"11'"`` -> 3352.8.
    Returns None when no measurement is present.
    """
    m = _FEET_INCHES_RE.search(text)
    if not m:
        return None
    feet = int(m.group("feet"))
    inches = int(m.group("inches") or 0)
    if inches >= 12:  # a typo, not a measurement
        return None
    return feet * MM_PER_FOOT + inches * MM_PER_INCH


def parse_dimension_pair(text: str) -> Dimension | None:
    """Extract a ``W x D`` room dimension from a label, in millimetres.

    Handles the imperial form printed on Indian residential plans
    (``"BEDROOM\\n11'0\\"X13'6\\""``) and a metric fallback (``"3350 X 4100"``).
    """
    if not text:
        return None

    m = _PAIR_RE.search(text)
    if m:
        w = int(m.group("feet")) * MM_PER_FOOT + int(m.group("inches") or 0) * MM_PER_INCH
        d = int(m.group("feet2")) * MM_PER_FOOT + int(m.group("inches2") or 0) * MM_PER_INCH
        if w > 0 and d > 0:
            return Dimension(w, d, m.group(0).strip(), imperial=True)

    m = _METRIC_PAIR_RE.search(text)
    if m:
        scale = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
        a = float(m.group("a"))
        b = float(m.group("b"))
        # A shared trailing unit applies to both operands ("3.35 x 4.10 m").
        unit = (m.group("bu") or m.group("au") or "").lower()
        if unit:
            a *= scale[unit]
            b *= scale[unit]
        else:
            # Bare numbers: 3350 is millimetres, 3.35 is metres. Anything under
            # 100 cannot be a room dimension in millimetres.
            factor = 1000.0 if max(a, b) < 100 else 1.0
            a *= factor
            b *= factor
        # Reject anything outside plausible room size (300 mm .. 60 m).
        if 300 <= a <= 60_000 and 300 <= b <= 60_000:
            return Dimension(a, b, m.group(0).strip(), imperial=False)

    return None


def parse_area_sqft(text: str) -> float | None:
    """Extract a stated area in square feet (``"CARPET AREA: 1,056 SQFT"``)."""
    if not text:
        return None
    m = _AREA_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group("value").replace(",", ""))
    except ValueError:
        return None


def infer_scale_from_extents(width: float, height: float) -> tuple[float, str]:
    """Guess millimetres-per-unit from drawing extents when ``$INSUNITS`` is 0.

    Returns ``(mm_per_unit, rationale)``. The reasoning is that a building floor
    plate has a known physical size range, so the magnitude of the extents
    identifies the unit. We deliberately return the rationale so ingest can show
    the user *why* a scale was chosen and let them override it.
    """
    span = max(abs(width), abs(height))
    if span <= 0 or not math.isfinite(span):
        return 1.0, "degenerate extents; assumed millimetres"

    # Candidate units and the plausible physical span each would imply.
    if span < 2:
        return 1000.0, f"extents span {span:.3g} units; too small for anything but metres on a tiny plan"
    if span < 200:
        return 1000.0, f"extents span {span:.3g} units; consistent with metres"
    if span < 2_000:
        return 10.0, f"extents span {span:.3g} units; consistent with centimetres"
    return 1.0, f"extents span {span:.3g} units; consistent with millimetres"


def mm_to_sqft(area_mm2: float) -> float:
    """Convert an area in mm² to square feet."""
    return area_mm2 / 1_000_000.0 * SQFT_PER_SQM


def format_feet_inches(mm: float) -> str:
    """Render millimetres as ``11'6"`` for imperial-labelled plans."""
    if mm < 0:
        return "-" + format_feet_inches(-mm)
    total_inches = mm / MM_PER_INCH
    feet = int(total_inches // 12)
    inches = int(round(total_inches - feet * 12))
    if inches == 12:  # rounding carried
        feet += 1
        inches = 0
    return f"{feet}'{inches}\""
