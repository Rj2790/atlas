"""Tests for the house-mapping stage.

Each of these pins down a failure that actually happened while the space-first
restructure was built, so they are regressions rather than illustrations.
"""

from __future__ import annotations

import math

from shapely.geometry import Polygon, box

from atlas.ingest.spaces import MapConfig, SpaceSeed, map_spaces
from atlas.planir import RoomKind
from atlas.units import Dimension


def _dim(w_mm: float, d_mm: float) -> Dimension:
    return Dimension(width_mm=w_mm, depth_mm=d_mm, source_text=f"{w_mm}x{d_mm}", imperial=False)


def test_partition_covers_the_whole_footprint() -> None:
    """The point of mapping space-first: no voids and no unknowns.

    The wall-first pipeline left 24% of the demo flat belonging to no room. A complete
    partition is the property that makes that impossible.
    """
    fp = box(0, 0, 8000, 6000)
    seeds = [
        SpaceSeed("LIVING", (2000, 3000), _dim(4000, 6000)),
        SpaceSeed("KITCHEN", (6000, 3000), _dim(4000, 6000)),
    ]
    smap = map_spaces(fp, seeds)

    assert len(smap.spaces) == 2
    assert smap.coverage() > 0.98
    covered = sum(s.area for s in smap.spaces)
    assert covered == 0 or abs(covered - fp.area) / fp.area < 0.05
    # And no two spaces overlap, so "complete" also means "a partition".
    a, b = smap.spaces
    assert a.polygon.intersection(b.polygon).area < fp.area * 0.01


def test_small_room_is_not_inflated_to_fill_the_plan() -> None:
    """Printed sizes are absolute targets, not ratios.

    Normalising the printed areas onto the footprint turned a 40 sqft toilet into a
    121 sqft one, because the footprint also contains wall thickness and unlabelled
    passage. The toilet must come out the size the drawing prints.
    """
    fp = box(0, 0, 10000, 6000)
    seeds = [
        SpaceSeed("LIVING", (2000, 3000), _dim(4000, 6000)),
        SpaceSeed("TOILET", (9000, 1000), _dim(1500, 2400)),
    ]
    smap = map_spaces(fp, seeds)
    toilet = next(s for s in smap.spaces if s.kind is RoomKind.BATHROOM)
    # Generous bound: the point is that it stays in the region of its printed size
    # rather than growing to a quarter of the flat.
    assert toilet.area < _dim(1500, 2400).area_mm2 * 2.2


def test_unlabelled_region_becomes_circulation_not_extra_bathroom() -> None:
    """A large leftover is a corridor nobody labelled, not a bigger toilet.

    Absorbing it into whichever room touched it produced a 40 sqft toilet measuring
    83 sqft with a 2.8 x 4.8 m footprint.
    """
    fp = box(0, 0, 12000, 6000)
    seeds = [
        SpaceSeed("BEDROOM", (1500, 3000), _dim(3000, 6000)),
        SpaceSeed("TOILET", (11000, 5000), _dim(1500, 2000)),
    ]
    smap = map_spaces(fp, seeds)
    kinds = [s.kind for s in smap.spaces]
    assert RoomKind.CORRIDOR in kinds, "the unlabelled middle should be named as passage"
    toilet = next(s for s in smap.spaces if s.kind is RoomKind.BATHROOM)
    assert toilet.area < _dim(1500, 2000).area_mm2 * 2.0


def test_recovered_walls_constrain_the_boundary() -> None:
    """A wall in the drawing decides where the split goes, overriding the areas.

    This is what "walls as a constraint, not a definition" has to mean in practice: the
    printed areas would put the boundary at the middle, the wall puts it at 3 m.
    """
    fp = box(0, 0, 8000, 4000)
    wall = ((3000.0, 0.0), (3000.0, 4000.0))
    # The bedroom is *told* it is 5 m wide. The wall says 3 m. The wall wins, and the
    # bedroom may not reclaim the shortfall from the far side of its own partition.
    seeds = [
        SpaceSeed("BEDROOM", (1500, 2000), _dim(5000, 4000)),
        SpaceSeed("KITCHEN", (6000, 2000), _dim(3000, 4000)),
    ]
    smap = map_spaces(fp, seeds, [wall])
    bed = next(s for s in smap.spaces if s.kind is RoomKind.BEDROOM)
    assert bed.polygon.bounds[2] < 3400, "bedroom should stop at the wall, not at mid-span"


def test_boundaries_are_axis_aligned() -> None:
    """Boundaries snap to the drawing's wall directions.

    Euclidean growth made round rooms, and the walls derived from them ran diagonally
    across the plan. Every boundary here should be horizontal or vertical.
    """
    fp = box(0, 0, 8000, 6000)
    seeds = [
        SpaceSeed("BEDROOM", (2000, 3000), _dim(4000, 6000)),
        SpaceSeed("BEDROOM", (6000, 3000), _dim(4000, 6000)),
    ]
    smap = map_spaces(fp, seeds, [((4000.0, 0.0), (4000.0, 6000.0))])
    assert smap.boundaries
    for b in smap.boundaries:
        (x0, y0), (x1, y1) = b.line.coords[0], b.line.coords[-1]
        ang = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 90.0
        assert min(ang, 90.0 - ang) < 3.0, f"oblique boundary at {ang:.1f} degrees"


def test_bedroom_to_corridor_is_always_walled() -> None:
    """An unmatched boundary defaults to walled; only open-plan pairs may be open.

    Treating every boundary with no recovered wall as an open threshold left bedrooms
    continuous with the corridor, which made every room trivially reachable and meant
    not one door was placed in the whole flat.
    """
    fp = box(0, 0, 9000, 4000)
    seeds = [
        SpaceSeed("BEDROOM", (2250, 2000), _dim(4500, 4000)),
        SpaceSeed("LOBBY", (6750, 2000), _dim(4500, 4000)),
    ]
    smap = map_spaces(fp, seeds)  # deliberately no recovered walls
    shared = [
        b
        for b in smap.boundaries
        if b.space_b is not None and {b.space_a, b.space_b} == {s.id for s in smap.spaces}
    ]
    assert shared
    assert all(b.walled for b in shared)


def test_living_and_dining_may_share_an_open_threshold() -> None:
    """The other half of the same rule: open plan is allowed where it is plausible.

    This is the user-visible complaint the restructure set out to fix — living and
    dining separated by a strip belonging to neither.
    """
    fp = box(0, 0, 9000, 4000)
    seeds = [
        SpaceSeed("LIVING", (2250, 2000), _dim(4500, 4000)),
        SpaceSeed("DINING", (6750, 2000), _dim(4500, 4000)),
    ]
    smap = map_spaces(fp, seeds)
    assert len(smap.spaces) == 2
    shared = [b for b in smap.boundaries if b.space_b is not None]
    assert shared
    assert any(not b.walled for b in shared), "living into dining should be continuous"
    # And they must actually touch — no void between them.
    a, b = smap.spaces
    assert a.polygon.distance(b.polygon) < 1.0


def test_adjacency_graph_is_reported() -> None:
    """Door placement walks this graph, so it has to be populated."""
    fp = box(0, 0, 9000, 4000)
    seeds = [
        SpaceSeed("BEDROOM", (2250, 2000), _dim(4500, 4000)),
        SpaceSeed("LOBBY", (6750, 2000), _dim(4500, 4000)),
    ]
    smap = map_spaces(fp, seeds)
    adj = smap.adjacency()
    ids = {s.id for s in smap.spaces}
    assert set(adj) == ids
    assert any(adj[i] for i in ids), "the two spaces are neighbours and should say so"


def test_no_labels_inside_footprint_is_reported_not_guessed() -> None:
    fp = box(0, 0, 4000, 4000)
    smap = map_spaces(fp, [SpaceSeed("BEDROOM", (50_000, 50_000), _dim(3000, 3000))])
    assert not smap.spaces
    assert smap.warnings


def test_l_shaped_footprint_is_partitioned_without_leaking_outside() -> None:
    """Spaces stay inside a non-convex dwelling."""
    fp = Polygon([(0, 0), (8000, 0), (8000, 4000), (4000, 4000), (4000, 8000), (0, 8000)])
    seeds = [
        SpaceSeed("LIVING", (2000, 2000), _dim(4000, 4000)),
        SpaceSeed("BEDROOM", (2000, 6000), _dim(4000, 4000)),
        SpaceSeed("KITCHEN", (6000, 2000), _dim(4000, 4000)),
    ]
    smap = map_spaces(fp, seeds, cfg=MapConfig(cell_mm=120.0))
    assert len(smap.spaces) == 3
    grown = max(s.polygon.difference(fp.buffer(150.0)).area for s in smap.spaces)
    assert grown < fp.area * 0.01, "a space escaped the footprint"


def test_unlabelled_region_splits_into_balcony_and_circulation() -> None:
    """A leftover region that is part outdoor strip and part corridor must be split.

    Labelling the whole of it merged the circulation spine with two balcony strips into a
    single 297 sqft region that touched every room in the flat and carried three french
    windows 2.7-3.6 m wide on its own walls. Classifying the whole of it by exposure does
    not separate them either: real balconies scored 40-53% of perimeter facing outdoors,
    pure circulation ~0%, and the merged region 26% — there is no threshold between two
    things that are stuck together. The split is geometric: balconies are shallow strips
    along the building edge, circulation is what is left inboard.
    """
    # A deep plan: rooms down the left, and a wide unlabelled remainder to the right whose
    # outer 2.4 m is against the building edge.
    fp = box(0, 0, 16000, 9000)
    seeds = [
        SpaceSeed("BEDROOM", (2000, 4500), _dim(4000, 9000)),
        SpaceSeed("KITCHEN", (6000, 4500), _dim(3000, 9000)),
    ]
    smap = map_spaces(fp, seeds, cfg=MapConfig(cell_mm=150.0))
    kinds = {s.kind for s in smap.spaces}
    assert RoomKind.BALCONY in kinds, "the strip along the building edge should be outdoor"
    assert RoomKind.CORRIDOR in kinds, "the inboard remainder should be circulation"
    # And no single leftover should swallow everything.
    biggest = max(s.area for s in smap.spaces if s.kind in (RoomKind.BALCONY, RoomKind.CORRIDOR))
    assert biggest < fp.area * 0.6
