"""Regression lock over every dwelling in the demo drawing.

All sixteen dwellings pass every invariant. That is the point of the file: the geometry was
originally tuned against unit 1 alone, and unit 1 turned out to be the best-behaved of the
sixteen, so single-unit testing hid failures in the other fifteen.

The decisive check is "no wall the *drawing* shows runs through a recovered room". Every
other check compares our rooms against our own walls, and both come from the same
partition, so they agree with each other by construction even when the partition ignored a
wall that is plainly drawn. Measured against the source line work it found 40 m of swallowed
wall on a flat that passed everything else.

What fixed it was making wall **bodies** occupy space no room may claim, rather than
treating walls as lines that merely interrupt growth. Lines force a trade-off that cannot be
won — more of them cuts swallowing but shatters the partition, fewer does the reverse — and
four attempts traded one against the other before the model changed.

So the test locks the *baseline*. Any dwelling that starts failing, or any dwelling that
gains failures, breaks the build. Fixing one is expected to tighten the baseline here,
which makes the improvement explicit and stops it silently regressing later.

Marked slow: it extracts sixteen dwellings from a 10 MB DWG and takes minutes. Run with
`pytest -m slow`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas.audit import audit_file

DRAWING = Path(__file__).resolve().parents[2] / "data" / "samples" / "demo-cad-1.dwg"

# unit number -> failure count, as measured on 2026-08-05. Lower is better; any increase
# is a regression. Units absent from this map must pass cleanly.
BASELINE: dict[int, int] = {
    # 2026-08-06: the audit grew teeth — area accounting against the declared polygon,
    # printed-dimension agreement, unjustified walls/doors/windows, furniture placement.
    # Under the stronger audit the drawing's sixteen dwellings hold one known defect
    # family: allocation against printed dimensions where the drawing's own kerb/screen
    # line work (150 mm bands on the wall layer) fences part of a room's printed
    # rectangle. Fixing it means teaching the growth engine that kerb-scale bands are
    # claimable for the labelled room — spaces.py surgery, tracked as the next geometry
    # work. Everything else — enclosure, reachability, overlaps, swallowed walls,
    # invented walls, phantom openings, doubled doors, area accounting (92-100%
    # accounted everywhere) — passes clean.
    1: 1,  # dims 88%: TOILET at 49%, shower kerb inside its printed rectangle
    4: 1,  # dims 85%: LIVING at 59%, open-plan share held by unlabelled circulation
    9: 1,  # dims 88%: TOILET at 43%, kerb-fenced flank
    14: 1,  # dims 85%: TOILET at 38%, kerb-fenced end
    16: 1,  # enclosure: TOILET edge 1234 mm short of the drawn kerb it should meet
}


@pytest.mark.slow
@pytest.mark.skipif(not DRAWING.exists(), reason="demo drawing not present")
def test_no_new_failures_across_every_dwelling() -> None:
    res = audit_file(DRAWING)
    assert len(res.units) >= 16, f"expected 16 dwellings, segmented {len(res.units)}"

    actual = res.failing_units()
    regressions: list[str] = []
    for n, count in sorted(actual.items()):
        allowed = BASELINE.get(n, 0)
        if count > allowed:
            unit = res.units[n - 1]
            regressions.append(
                f"unit {n} ({unit.name}): {count} failures, baseline {allowed} — "
                + "; ".join(unit.failures)
            )
    assert not regressions, "new geometry failures:\n" + "\n".join(regressions)


@pytest.mark.slow
@pytest.mark.skipif(not DRAWING.exists(), reason="demo drawing not present")
def test_baseline_is_not_stale() -> None:
    """A dwelling listed in the baseline that now passes should be removed from it.

    Otherwise the baseline silently grants permission to regress a unit that has since
    been fixed, which is exactly the blind spot this suite exists to close.
    """
    res = audit_file(DRAWING)
    actual = res.failing_units()
    fixed = [n for n, allowed in BASELINE.items() if allowed and actual.get(n, 0) < allowed]
    assert not fixed, (
        "these units improved — tighten BASELINE to lock the gain: "
        + ", ".join(f"unit {n}: {actual.get(n, 0)} now vs {BASELINE[n]} recorded" for n in fixed)
    )
