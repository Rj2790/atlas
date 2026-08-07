"""Map a dwelling into a complete set of spaces, before any wall is built.

This inverts the order the pipeline used to work in, and the reason is worth stating
plainly because it was a real design error.

The old order was **wall-first**: pair up the drawn wall faces, then take the leftover
cycles of that wall network as the rooms. Rooms were therefore *derived from* the
noisiest step in the whole pipeline, so every wall the face-pairing missed became a
hole in the room map. Measured on the demo flat: 24% of the dwelling footprint
(39.6 m²) belonged to no room at all, four fragments were unlabelled — one of them
154 sqft — and adjacent spaces sat 150–240 mm apart with nothing in between.

The evidence is the other way round. The drawing *tells* us the spaces: sixteen
labels, each with a name and a printed size. That is ground truth. The wall geometry
is accurate where it was recovered and simply absent where it was not.

So this module makes **space the primitive**:

1. Take the dwelling footprint and every label inside it.
2. Grow all spaces at once from their label positions, balanced toward each label's
   printed area, over a raster of the footprint — so the partition is *complete by
   construction*. Every point belongs to exactly one named space. There is no
   "unknown" and no void, because there is nowhere for one to hide.
3. Recovered walls act as **barriers** to that growth. Where a wall exists the space
   boundary lands exactly on it; where none was recovered, the areas decide. Wall
   geometry is used as a constraint on the answer instead of as its definition.
4. Boundaries between spaces then fall out as a by-product, along with which spaces
   are adjacent and how long each shared edge is — which is exactly what the wall
   extrusion, the door placement and the circulation graph all need next.

Walls come *after* this, centred on the boundaries, which is why no gaps can remain:
a shared boundary is one line, and the wall straddles it.

Everything is millimetres.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box as shp_box
from shapely.ops import linemerge, unary_union
from shapely.prepared import prep

from ..planir import RoomKind, classify_room_name
from ..units import Dimension, mm_to_sqft, parse_dimension_pair

__all__ = ["SpaceSeed", "Space", "Boundary", "SpaceMap", "MapConfig", "map_spaces"]


_OPEN_PLAN = frozenset({RoomKind.LIVING, RoomKind.DINING})
"""The one pairing that is routinely a single continuous room: combined living-dining.

Only a boundary with both sides in this set may be an open threshold; everything else
gets a wall, with a door in it if the topology needs one.

This was briefly widened to include LOBBY, CORRIDOR and ENTRANCE, and that was wrong in
a way that wrecked the model. It made living-to-corridor, dining-to-corridor and
lobby-to-living all wall-less, so the entire centre of the flat became one undivided
space: the living room ended up with 36% of its perimeter walled and the dining room 46%.
A corridor is *defined* by the walls that form it — take them away and it is not a
corridor, it is the absence of rooms. Living into dining is the exception, not the rule.
"""


def _n4(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    i, j = cell
    return ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1))


def _flood(
    pool: set[tuple[int, int]],
    seeds: Sequence[tuple[int, int]],
    budget: float,
    cell_area: float,
    centre: dict[tuple[int, int], tuple[float, float]],
    toward: tuple[float, float],
    basis: float = 0.0,
    blocked: "Callable[[tuple[int, int], tuple[int, int]], bool] | None" = None,
) -> list[tuple[int, int]]:
    """Take up to ``budget`` of area out of ``pool``, starting from ``seeds``.

    Nearest-to-``toward`` first, so a room reclaiming area gets the part closest to
    itself rather than an arm reaching across the flat.

    ``blocked`` must be honoured here as it is during growth. Without it this pass walked
    the leftover pool straight through drawn walls: the living room reclaimed its way to
    179 sqft against a printed 204 and ended up with 14 m of drawn wall inside it, undoing
    the barriers the growth phase had respected.
    """
    if not seeds or budget < cell_area:
        return []
    cb, sb = math.cos(basis), math.sin(basis)

    def cost(cell: tuple[int, int]) -> float:
        x, y = centre[cell]
        dx, dy = x - toward[0], y - toward[1]
        return max(abs(dx * cb + dy * sb), abs(-dx * sb + dy * cb))

    heap = [(cost(c), c) for c in seeds]
    heapq.heapify(heap)
    seen, taken = set(seeds), []
    while heap and len(taken) * cell_area < budget:
        _, cur = heapq.heappop(heap)
        taken.append(cur)
        for nb in _n4(cur):
            if nb in pool and nb not in seen and not (blocked and blocked(cur, nb)):
                seen.add(nb)
                heapq.heappush(heap, (cost(nb), nb))
    return taken


def _components(
    cells: set[tuple[int, int]],
    blocked: "Callable[[tuple[int, int], tuple[int, int]], bool] | None" = None,
) -> list[list[tuple[int, int]]]:
    """4-connected components of a cell set, not crossing a wall.

    Without the ``blocked`` test a leftover region that spans a partition came out as one
    component and was handed whole to a single space, which is how rooms ended up
    containing metres of wall the drawing plainly shows. A wall divides a region into two
    regions; that is what a wall is.
    """
    seen: set[tuple[int, int]] = set()
    out: list[list[tuple[int, int]]] = []
    for start in cells:
        if start in seen:
            continue
        comp, q = [], deque([start])
        seen.add(start)
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nb in _n4(cur):
                if nb in cells and nb not in seen and not (blocked and blocked(cur, nb)):
                    seen.add(nb)
                    q.append(nb)
        out.append(comp)
    return out


def _erodes(comp: Iterable[tuple[int, int]], rounds: int) -> bool:
    """Does this component survive ``rounds`` of erosion — is it wider than a wall?

    A 150 mm wall footprint is one or two cells across and vanishes immediately. A
    corridor is ten or more and survives. This is what separates "slack that belongs to
    the rooms either side" from "a room nobody labelled".
    """
    live = set(comp)
    for _ in range(rounds):
        live = {
            (i, j)
            for (i, j) in live
            if all(nb in live for nb in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)))
        }
        if not live:
            return False
    return True


@dataclass(slots=True)
class SpaceSeed:
    """A label read off the drawing: the primary evidence a space exists."""

    label: str
    point: tuple[float, float]
    dimension: Dimension | None = None

    @property
    def kind(self) -> RoomKind:
        return classify_room_name(self.label)

    @property
    def target_area(self) -> float | None:
        return self.dimension.area_mm2 if self.dimension else None


@dataclass(slots=True)
class Space:
    id: str
    label: str
    kind: RoomKind
    polygon: Polygon
    """The space's full extent, wall centrelines included. The floor is inset from
    this once wall thicknesses are known."""

    seed: tuple[float, float]
    target_area: float | None
    dimension: Dimension | None = None
    """The printed ``11'0"X13'6"`` this space was grown to match, kept so the room can
    still be reconciled against the drawing's stated intent downstream."""

    cells: int = 0

    @property
    def area(self) -> float:
        return self.polygon.area

    def agreement(self) -> float | None:
        """How close the mapped area is to the printed one, in [0, 1]."""
        if not self.target_area:
            return None
        return min(self.area, self.target_area) / max(self.area, self.target_area)


@dataclass(slots=True)
class Boundary:
    """A shared edge between two spaces, or between a space and outside."""

    space_a: str
    space_b: str | None
    """None means this edge faces outside the dwelling."""

    line: LineString
    thickness: float
    walled: bool
    exterior: bool = False
    """True only when this edge is on the dwelling envelope. A boundary with no space
    behind it is not automatically external — it may face a gap the map could not
    attribute — and it gets a wall either way; this only decides the finish."""
    """True when a real wall was recovered along this edge. False means the drawing
    shows the two spaces as continuous — an open plan connection, not a wall, which is
    how living-and-dining stops being two rooms with a mystery strip between them."""

    @property
    def length(self) -> float:
        return self.line.length


@dataclass
class SpaceMap:
    footprint: Polygon
    spaces: list[Space] = field(default_factory=list)
    boundaries: list[Boundary] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def coverage(self) -> float:
        """Fraction of the footprint assigned to a space. Should be ~1.0."""
        if self.footprint.area <= 0:
            return 0.0
        return min(1.0, sum(s.area for s in self.spaces) / self.footprint.area)

    def adjacency(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {s.id: set() for s in self.spaces}
        for b in self.boundaries:
            if b.space_b is None:
                continue
            out.setdefault(b.space_a, set()).add(b.space_b)
            out.setdefault(b.space_b, set()).add(b.space_a)
        return out


@dataclass(slots=True)
class MapConfig:
    cell_mm: float = 90.0
    """Raster resolution. 90 mm resolves a 100 mm partition and keeps a typical flat
    under ~40k cells."""

    max_cells: int = 320_000
    interior_wall_mm: float = 150.0
    exterior_wall_mm: float = 230.0
    wall_snap_mm: float = 260.0
    """How far from a boundary a recovered wall may be and still be considered to be
    *that* boundary's wall — supplying its real thickness."""

    min_space_area_mm2: float = 700_000.0
    """0.7 m². Below this a seed is a duct or a mislabelled scrap."""

    balcony_depth_mm: float = 2400.0
    """How deep a balcony is, used to peel the outdoor strip off an unlabelled region that
    is part balcony and part circulation. Residential balconies here run 1.4-2.6 m."""

    reach_factor: float = 1.35
    """How far a space may grow from its label, as a multiple of the square root of its
    printed area. A room of area A is roughly sqrt(A) across, so this allows a comfortably
    L-shaped room while stopping a space sending an arm down a corridor to make up area it
    cannot reach compactly."""

    outdoor_edge_fraction: float = 0.25
    """How much of an unlabelled region's perimeter must face beyond the building before
    it is read as outdoor rather than as internal circulation. A corridor scores near
    zero; a balcony strip runs along the building edge and scores high."""

    corridor_min_area_mm2: float = 2_000_000.0
    """2 m². A leftover region this large, and wider than a wall, is passage the
    drawing never labelled — so it is mapped as circulation instead of being absorbed
    into whichever room happens to touch it."""

    min_boundary_mm: float = 300.0
    """Ignore shared edges shorter than this: they are raster corner-touching, not a
    real party wall."""


# ---------------------------------------------------------------------------


def _basis_angle(
    segments: Sequence[tuple[tuple[float, float], tuple[float, float]]],
) -> float:
    """The drawing's dominant wall direction, in radians, folded into [0, pi/2).

    Buildings are orthogonal but drawings are not always axis-aligned — this one sits on
    survey coordinates at a slight rotation. Everything downstream that wants to snap a
    boundary to "straight" needs to know which way straight is, and the walls that *were*
    recovered say so: they are accurate even when incomplete. Weighted by length, so a
    long external wall counts for more than a short return.
    """
    buckets: dict[int, float] = {}
    for a, b in segments:
        L = math.dist(a, b)
        if L < 500.0:
            continue
        ang = math.atan2(b[1] - a[1], b[0] - a[0]) % (math.pi / 2)
        key = int(round(math.degrees(ang)))
        buckets[key] = buckets.get(key, 0.0) + L
    if not buckets:
        return 0.0
    return math.radians(max(buckets, key=lambda k: buckets[k]))


def _snap_to_basis(line: LineString, basis: float, tol_deg: float = 26.0) -> LineString:
    """Rotate a boundary onto the nearest of the drawing's two wall directions.

    Region growing produces the odd oblique edge wherever no wall constrained it, and
    those became diagonal walls slicing across rooms — visible in the plan as grey
    45-degree segments through the living room. A partition in a flat is parallel or
    perpendicular to the others; snapping says so, and only when the edge is already
    close enough that snapping is a correction rather than an invention.
    """
    (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
    ang = math.atan2(y1 - y0, x1 - x0)
    best, err = ang, None
    for k in range(-2, 3):
        cand = basis + k * math.pi / 2
        d = abs((ang - cand + math.pi) % (2 * math.pi) - math.pi)
        if err is None or d < err:
            best, err = cand, d
    if err is None or err > math.radians(tol_deg):
        return line
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = math.dist((x0, y0), (x1, y1)) / 2.0
    dx, dy = math.cos(best) * half, math.sin(best) * half
    return LineString([(mx - dx, my - dy), (mx + dx, my + dy)])


def _rectify(poly: Polygon, basis: float, tol: float) -> Polygon:
    """Square up a raster-derived outline against the drawing's wall directions.

    Region growing works on a grid, so every edge that no wall constrained comes out as
    a staircase. Once the growth metric was made rectilinear the walls came out straight,
    but the *floor* outlines kept their jagged treads — visible as a zig-zag along the
    living-into-dining threshold where the two spaces meet with no wall between them.

    Each edge is snapped to whichever of the two wall directions it is closer to, and
    consecutive vertices are pulled into line. Rooms are rectilinear; this states it.
    """
    if poly.is_empty or len(poly.exterior.coords) < 5:
        return poly
    cb, sb = math.cos(-basis), math.sin(-basis)

    def fwd(p: tuple[float, float]) -> list[float]:
        return [p[0] * cb - p[1] * sb, p[0] * sb + p[1] * cb]

    def back(p: Sequence[float]) -> tuple[float, float]:
        return (p[0] * cb + p[1] * sb, -p[0] * sb + p[1] * cb)

    pts = [fwd(p) for p in poly.simplify(tol, preserve_topology=True).exterior.coords[:-1]]
    if len(pts) < 4:
        return poly
    for _ in range(3):
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
            if dx <= dy:  # meant to be vertical
                mid = (a[0] + b[0]) / 2.0
                a[0] = b[0] = mid
            else:  # meant to be horizontal
                mid = (a[1] + b[1]) / 2.0
                a[1] = b[1] = mid
    out = Polygon([back(p) for p in pts]).buffer(0)
    if isinstance(out, MultiPolygon):
        out = max(out.geoms, key=lambda g: g.area)
    # Refuse a rectification that moved real area: better a stepped floor than a wrong one.
    if not isinstance(out, Polygon) or out.is_empty or abs(out.area - poly.area) > poly.area * 0.18:
        return poly
    return out


def _wall_barrier_index(
    segments: Sequence[tuple[tuple[float, float], tuple[float, float]]],
) -> object:
    """Spatial index of recovered wall centrelines, for fast crossing tests."""
    lines = [LineString([a, b]) for a, b in segments if math.dist(a, b) > 1e-6]
    return prep(unary_union(lines)) if lines else None


def map_spaces(
    footprint: Polygon,
    seeds: Sequence[SpaceSeed],
    wall_segments: Sequence[tuple[tuple[float, float], tuple[float, float]]] = (),
    cfg: MapConfig | None = None,
    *,
    prefix: str = "s",
    wall_solid: Any | None = None,
    outdoor_evidence: Any | None = None,
) -> SpaceMap:
    """Partition ``footprint`` into one space per seed. See the module docstring.

    ``wall_solid`` is the walls' own **footprint** — their bodies, not their centrelines.
    Cells inside it are not offered to any space, which is what finally makes "a room
    cannot contain a wall" true by construction rather than by tuning.

    Passing walls only as *lines* forced a trade-off that could not be won: more barrier
    lines cut down the walls rooms swallowed but shattered the partition into stubs, and
    fewer did the reverse. Four attempts traded one against the other. A body occupies
    space; a line merely interrupts a step, and the 150 mm strip between two faces of the
    same wall became its own region. With bodies excluded there is no strip to become a
    region, and no cell for a room to claim inside a wall.
    """
    cfg = cfg or MapConfig()
    result = SpaceMap(footprint=footprint)

    usable = [s for s in seeds if footprint.buffer(cfg.cell_mm).contains(Point(*s.point))]
    if not usable:
        result.warnings.append("no room labels fall inside this dwelling's footprint")
        return result

    # --- raster ------------------------------------------------------------
    minx, miny, maxx, maxy = footprint.bounds
    cell = cfg.cell_mm
    while ((maxx - minx) / cell) * ((maxy - miny) / cell) > cfg.max_cells:
        cell *= 1.3
    nx = max(1, int(math.ceil((maxx - minx) / cell)))
    ny = max(1, int(math.ceil((maxy - miny) / cell)))

    prepared = prep(footprint)
    # Not dilated. Half a cell of dilation does make exclusion contiguous for a thin wall,
    # but it also pulls every room half a cell back from every wall, and the rooms then stop
    # touching their own walls — unsealed rooms went from 1 to 8. The thin-wall case is
    # handled by the line barriers instead, which block a cell-to-cell move whatever the
    # wall's thickness.
    solid = prep(wall_solid) if wall_solid is not None and not wall_solid.is_empty else None
    centre: dict[tuple[int, int], tuple[float, float]] = {}
    wall_cells: dict[tuple[int, int], tuple[float, float]] = {}
    for j in range(ny):
        cy = miny + (j + 0.5) * cell
        for i in range(nx):
            cx = minx + (i + 0.5) * cell
            pt = Point(cx, cy)
            if not prepared.contains(pt):
                continue
            if solid is not None and solid.contains(pt):
                wall_cells[(i, j)] = (cx, cy)  # this is wall, not floor
            else:
                centre[(i, j)] = (cx, cy)
    if not centre:
        result.warnings.append("dwelling footprint rasterised to nothing")
        return result

    cell_area = footprint.area / len(centre)
    if wall_cells:
        result.notes.append(
            f"{len(wall_cells) * cell_area / 1e6:.1f} m² of wall body excluded from the "
            "partition, so no room can contain a wall"
        )
    barrier = _wall_barrier_index(wall_segments)
    basis = _basis_angle(wall_segments)
    cos_b, sin_b = math.cos(basis), math.sin(basis)

    def blocked(a: tuple[int, int], b: tuple[int, int]) -> bool:
        """Does moving between these cells cross a recovered wall?

        This is what makes the partition respect the drawing. Growth stops at a real
        wall, so the boundary lands on it exactly; where no wall was recovered, growth
        continues and the printed areas decide where the division falls.
        """
        if barrier is None:
            return False
        return barrier.intersects(LineString([centre[a], centre[b]]))

    # --- targets -----------------------------------------------------------
    # Printed sizes are used as *absolute* targets, not as ratios to be scaled onto the
    # footprint. Scaling them up was measurably wrong: the footprint also contains the
    # walls' own thickness and any circulation the drawing never labelled, so spreading
    # that slack proportionally inflated every room — it turned a 40 sqft toilet into a
    # 121 sqft one. A space grows to the size the drawing prints and then stops.
    known = [s.target_area for s in usable if s.target_area]
    fill = (sum(known) / len(known)) if known else footprint.area / len(usable)
    want = [s.target_area or fill for s in usable]

    # --- seed --------------------------------------------------------------
    owner: dict[tuple[int, int], int] = {}
    space_id: dict[int, str] = {}
    frontier: list[list[tuple[float, tuple[int, int]]]] = [[] for _ in usable]
    expanded: list[set[tuple[int, int]]] = [set() for _ in usable]
    claimed = [0.0] * len(usable)

    for idx, seed in enumerate(usable):
        key = (int((seed.point[0] - minx) / cell), int((seed.point[1] - miny) / cell))
        if key not in centre or key in owner:
            free = [k for k in centre if k not in owner]
            if not free:
                continue
            key = min(free, key=lambda k: math.dist(seed.point, centre[k]))
        owner[key] = idx
        claimed[idx] += cell_area
        # Seeded already-owned; step() pops it, sees it is ours, and expands from it.
        expanded[idx].add(key)
        heapq.heappush(frontier[idx], (0.0, key))

    # --- grow, phase 1: to the printed size ---------------------------------
    def step(idx: int) -> bool:
        """Claim the unowned cell nearest space ``idx``'s label. False when boxed in.

        Nearest-first rather than breadth-first, and the difference is visible in the
        output. Plain BFS expands along whatever is open, so a living room with no wall
        between it and the hallway grows *down the hallway* — it came out as a 6.2 × 3.8 m
        sprawl of the right area but the wrong shape. Growing outward from the label
        instead keeps a space compact, and the walls it runs into give it its rectangle.
        """
        heap = frontier[idx]
        seen = expanded[idx]
        sx, sy = usable[idx].point
        while heap:
            _, cur = heapq.heappop(heap)
            fresh = cur not in owner
            if not fresh and owner[cur] != idx:
                continue
            if fresh:
                owner[cur] = idx
                claimed[idx] += cell_area
            i, j = cur
            for nb in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                # `seen` is per-space and never cleared, so each cell is queued and
                # barrier-tested once. Without it a cell reachable from four
                # neighbours is pushed four times, each duplicate re-pushes its own
                # neighbours on pop, and the heap grows combinatorially — the run
                # stopped terminating.
                if nb in seen or nb in owner or nb not in centre:
                    continue
                if blocked(cur, nb):
                    continue
                seen.add(nb)
                nx, ny = centre[nb]
                # Chebyshev distance in the drawing's wall directions, not Euclidean.
                # Euclidean grows a disc, so any edge a wall did not constrain came out
                # curved — the living room was a rounded blob and the walls derived from
                # it were diagonal. The L-infinity metric grows an axis-aligned
                # rectangle, which is the shape rooms actually are.
                du = (nx - sx) * cos_b + (ny - sy) * sin_b
                dv = -(nx - sx) * sin_b + (ny - sy) * cos_b
                cost = max(abs(du), abs(dv))
                # A space may not reach indefinitely from its own label.
                #
                # Nearest-first growth is compact while there is compact room to take. Once
                # a space is boxed in but still short of its printed area it carries on
                # along whatever happens to be open, and the result is an arm rather than a
                # room: the living room on one flat filled 31% of its bounding box, so a
                # sofa fitted nowhere despite the area being there on paper.
                #
                # The limit scales with the printed area, because that is how big the room
                # is *meant* to be: a room of area A is about sqrt(A) across, so anything
                # much beyond that from the label is reaching into somewhere else. Area lost
                # this way is left for the leftover pass, which names it honestly rather
                # than hiding it inside a room it does not belong to.
                if cost > reach[idx]:
                    continue
                heapq.heappush(heap, (cost, nb))
            if fresh:
                return True
        return False

    reach = [cfg.reach_factor * math.sqrt(max(w, 1.0)) for w in want]

    active = {i for i, f in enumerate(frontier) if f}
    while active:
        # Whichever space is proportionally furthest behind its printed size grows next,
        # so a small utility room is not starved while a large living room fills. Spaces
        # that have reached their printed size drop out and stop competing.
        hungry = {i for i in active if claimed[i] < want[i]}
        if not hungry:
            break
        idx = min(hungry, key=lambda k: claimed[k] / max(want[k], 1e-9))
        if not step(idx):
            active.discard(idx)

    # --- phase 2: account for what is left over -----------------------------
    # The footprint is larger than the printed rooms sum to, for two quite different
    # reasons, and conflating them was producing nonsense.
    #
    # Thin strips are the walls' own footprints. They belong to the rooms either side,
    # split down the middle, which is exactly where the wall centreline goes.
    #
    # Anything substantial is a space the drawing simply did not label — passage between
    # the bedrooms, the run from the front door to the living room. Absorbing that into
    # whatever happens to touch it is how a 40 sqft toilet came out 83 sqft with a
    # 2.8 x 4.8 m footprint, and a 132 sqft bedroom came out 294. It is a corridor. The
    # honest answer is to *name* it as one, not to inflate a bathroom.
    unclaimed = {k for k in centre if k not in owner}
    strips = 0.0
    for comp in _components(unclaimed, blocked):
        pool = set(comp)

        # A big leftover touching a room that came out *smaller than the drawing says*
        # is not a corridor — it is that room's missing area, cut off by a barrier the
        # wall solver got wrong. Give it back before considering anything else: the
        # living room measured 135 sqft against a printed 204, with a 168 sqft blob
        # sitting against it, and calling that blob a corridor would have been a
        # mapping error dressed up as a finding.
        while len(pool) * cell_area >= cfg.corridor_min_area_mm2:
            # Reachability, not mere adjacency. Cells on the far side of a wall touch it
            # on the grid but are not *in* the room, so reclaiming them would grow a
            # bedroom straight through its own partition.
            touching = {
                owner[nb]
                for c in pool
                for nb in _n4(c)
                if nb in owner and not blocked(c, nb)
            }
            short = [i for i in touching if want[i] - claimed[i] >= cfg.corridor_min_area_mm2 * 0.4]
            if not short:
                break
            idx = max(short, key=lambda i: want[i] - claimed[i])
            budget = want[idx] - claimed[idx]
            seedset = [
                c
                for c in pool
                if any(owner.get(nb) == idx and not blocked(c, nb) for nb in _n4(c))
            ]
            took = _flood(
                pool, seedset, budget, cell_area, centre, usable[idx].point, basis, blocked
            )
            if not took:
                break
            for c in took:
                owner[c] = idx
                pool.discard(c)
                claimed[idx] += cell_area
            result.notes.append(
                f"returned {len(took) * cell_area / 1e6:.1f} m² to {usable[idx].label} — it "
                "measured short of its printed size and this area adjoined it"
            )

        if len(pool) * cell_area >= cfg.corridor_min_area_mm2 and _erodes(pool, 2):
            # An unlabelled region is often *both* outdoor space and circulation, and it
            # has to be split rather than labelled one thing.
            #
            # Naming the whole of it "PASSAGE" merged the circulation spine with two
            # balcony strips into a single 297 sqft region that touched every room in the
            # flat and carried three french windows 2.7-3.6 m wide on its own walls —
            # nobody glazes a corridor like that. Classifying the whole of it by how much
            # of its perimeter faces outdoors does not work either: real balconies here
            # score 40-53%, pure circulation ~0%, and the merged region 26% — there is no
            # threshold that separates two things that are stuck together.
            #
            # So split it geometrically. Balconies are shallow strips along the building
            # edge; circulation is what is left inboard of them.
            # When the drawing marks its balconies (a hatch layer), that evidence decides
            # outdoors-vs-circulation outright and the geometric banding below is skipped.
            # The heuristic exists only for drawings that give us nothing.
            if outdoor_evidence is not None and not outdoor_evidence.is_empty:
                ev = prep(outdoor_evidence)
                covered = {c for c in pool if ev.contains(Point(*centre[c]))}
                parts = []
                parts += [("BALCONY", comp) for comp in _components(covered)]
                parts += [("PASSAGE", comp) for comp in _components(pool - covered)]
                big = cfg.corridor_min_area_mm2
                for label, comp in parts:
                    if len(comp) * cell_area < big * 0.5:
                        strips += len(comp) * cell_area
                        for c in comp:
                            votes: dict[int, int] = {}
                            for nb in _n4(c):
                                if nb in owner and not blocked(c, nb):
                                    votes[owner[nb]] = votes.get(owner[nb], 0) + 1
                            if votes:
                                owner[c] = max(votes, key=lambda v: votes[v])
                        continue
                    idx = len(usable)
                    cx = sum(centre[c][0] for c in comp) / len(comp)
                    cy = sum(centre[c][1] for c in comp) / len(comp)
                    usable.append(SpaceSeed(label=label, point=(cx, cy)))
                    want.append(len(comp) * cell_area)
                    claimed.append(len(comp) * cell_area)
                    frontier.append([])
                    expanded.append(set())
                    for c in comp:
                        owner[c] = idx
                    result.notes.append(
                        f"an unlabelled {len(comp) * cell_area / 1e6:.1f} m² region was mapped as "
                        + ("outdoor space per the drawing's balcony hatch" if label == "BALCONY"
                           else "circulation (no balcony hatch on it)")
                    )
                continue

            depth_cells = max(1, int(round(cfg.balcony_depth_mm / cell)))
            band: set[tuple[int, int]] = set()
            seeded = {c for c in pool if any(nb not in centre for nb in _n4(c))}
            queue = deque((c, 0) for c in seeded)
            reached = set(seeded)
            while queue:
                c, dist = queue.popleft()
                band.add(c)
                if dist >= depth_cells - 1:
                    continue
                for nb in _n4(c):
                    if nb in pool and nb not in reached:
                        reached.add(nb)
                        queue.append((nb, dist + 1))
            inboard = pool - band

            parts: list[tuple[str, list[tuple[int, int]]]] = []
            big = cfg.corridor_min_area_mm2
            if len(band) * cell_area >= big and len(inboard) * cell_area >= big:
                parts += [("BALCONY", comp) for comp in _components(band)]
                parts += [("PASSAGE", comp) for comp in _components(inboard)]
            else:
                # Not separable: decide the whole region on how exposed it is.
                edge = sum(1 for c in pool for nb in _n4(c) if nb not in centre)
                perim = sum(1 for c in pool for nb in _n4(c) if nb not in pool)
                frac = edge / perim if perim else 0.0
                parts = [
                    ("BALCONY" if frac >= cfg.outdoor_edge_fraction else "PASSAGE", list(pool))
                ]

            for label, comp in parts:
                if len(comp) * cell_area < big * 0.5:
                    strips += len(comp) * cell_area
                    for c in comp:
                        votes: dict[int, int] = {}
                        for nb in _n4(c):
                            if nb in owner:
                                votes[owner[nb]] = votes.get(owner[nb], 0) + 1
                        if votes:
                            owner[c] = max(votes, key=lambda v: votes[v])
                    continue
                idx = len(usable)
                cx = sum(centre[c][0] for c in comp) / len(comp)
                cy = sum(centre[c][1] for c in comp) / len(comp)
                usable.append(SpaceSeed(label=label, point=(cx, cy)))
                want.append(len(comp) * cell_area)
                claimed.append(len(comp) * cell_area)
                frontier.append([])
                expanded.append(set())
                for c in comp:
                    owner[c] = idx
                result.notes.append(
                    f"an unlabelled {len(comp) * cell_area / 1e6:.1f} m² region was mapped as "
                    + ("outdoor space along the building edge" if label == "BALCONY"
                       else "circulation rather than absorbed into the rooms around it")
                )
        else:
            strips += len(pool) * cell_area
            for c in pool:
                votes: dict[int, int] = {}
                for nb in _n4(c):
                    if nb in owner and not blocked(c, nb):
                        votes[owner[nb]] = votes.get(owner[nb], 0) + 1
                if votes:
                    owner[c] = max(votes, key=lambda v: votes[v])
    if strips:
        result.notes.append(f"{strips / 1e6:.1f} m² of wall footprint split between its rooms")

    # --- straighten the partition, on the grid ------------------------------
    # Region growing leaves single-cell jags along any boundary a wall did not
    # constrain. Squaring those up *after* vectorising was the mistake: each space was
    # rectified independently, so two neighbours drifted apart and no longer shared an
    # edge, and everything downstream that depends on adjacency — walls, doors,
    # enclosure — broke in ways that could not be fixed with a tolerance.
    #
    # Doing it here fixes it once, for both sides at the same time. A majority filter
    # over the four neighbours removes protrusions and notches while keeping the
    # partition complete: a cell is only ever reassigned to an owner that already
    # surrounds it, so no space becomes disconnected and no cell becomes unowned.
    for _ in range(12):
        moved = 0
        for c in list(owner.keys()):
            me = owner[c]
            votes: dict[int, int] = {}
            for nb in _n4(c):
                if nb in owner:
                    votes[owner[nb]] = votes.get(owner[nb], 0) + 1
            if not votes:
                continue
            mine = votes.get(me, 0)
            top = max(votes, key=lambda v: votes[v])
            if top == me:
                continue
            # Never hand a cell across a drawn wall. This filter and the straightener
            # below both reassigned cells purely on neighbour counts, so they quietly
            # undid the barriers growth had respected — the living room ended up 179 sqft
            # against a printed 204 with 14 m of drawn wall inside it.
            if all(blocked(c, nb) for nb in _n4(c) if owner.get(nb) == top):
                continue
            # Two conditions, and the second one is what removes diagonals. A clear
            # majority (3 of 4) flattens a protrusion. But a 45-degree staircase has
            # *exactly two* neighbours of each owner at every tooth, so a majority rule
            # never fires on it and the diagonal survives — it came through as zig-zag
            # walls across a toilet and a lobby. A cell that is barely attached to its
            # own space (at most one orthogonal neighbour) while another space holds two
            # is a tooth on such a staircase, and giving it up straightens the run.
            if votes[top] >= 3 or (mine <= 1 and votes[top] >= 2):
                owner[c] = top
                moved += 1
        if not moved:
            break

    n = _straighten_diagonals(owner, cell_area, blocked=blocked)
    if n:
        result.notes.append(f"{n} diagonal interface(s) between spaces squared up")

    # --- vectorise ---------------------------------------------------------
    for idx, seed in enumerate(usable):
        keys = [k for k, o in owner.items() if o == idx]
        if not keys:
            result.warnings.append(f"space {seed.label!r} won no area and was dropped")
            continue
        boxes = [
            shp_box(minx + i * cell, miny + j * cell, minx + (i + 1) * cell, miny + (j + 1) * cell)
            for (i, j) in keys
        ]
        # A hair of dilation makes edge-adjacency unambiguous to GEOS, which otherwise
        # returns a MultiPolygon of slivers for cells that genuinely share an edge.
        merged = unary_union([b.buffer(cell * 0.02, join_style=2) for b in boxes])
        if isinstance(merged, MultiPolygon):
            merged = max(merged.geoms, key=lambda g: g.area)
        if not isinstance(merged, Polygon) or merged.area < cfg.min_space_area_mm2:
            continue
        # Simplify only enough to drop collinear vertices. Anything stronger moves the
        # outline off the grid lines the walls are built on, and then the floor no longer
        # meets its own wall.
        # Keep the interior rings. Rebuilding from `merged.exterior` alone fills any hole,
        # so a space that wraps around another — a balcony round a kitchen, a lobby inside
        # a living room — absorbed its neighbour entirely and two rooms then claimed the
        # same 7.5 m² of floor. The cell sets are disjoint; only this reconstruction was
        # not.
        poly = merged.simplify(cell * 0.12, preserve_topology=True)
        poly = poly.intersection(footprint)
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        space_id[idx] = f"{prefix}{len(result.spaces) + 1}"
        result.spaces.append(
            Space(
                id=space_id[idx],
                label=seed.label,
                kind=seed.kind,
                polygon=poly,
                seed=seed.point,
                target_area=seed.target_area,
                dimension=seed.dimension,
                cells=len(keys),
            )
        )

    # Boundaries come from the cell partition, not from the polygons.
    #
    # Deriving them by intersecting polygon outlines needed a distance tolerance to
    # decide adjacency, and that tolerance could never be right: too tight and a bedroom
    # and the passage beside it registered as non-adjacent, so no wall was built and the
    # room was open to the corridor for 3.9 m; too loose and the runs got chopped
    # differently on each side. Tuning it made the numbers oscillate rather than
    # converge, which is the signature of a wrong approach rather than a wrong constant.
    #
    # In the raster there is nothing to tune. Two cells side by side either have
    # different owners or they do not, and the grid line between them is the boundary
    # exactly. Since the partition covers the whole footprint, every millimetre of every
    # room's perimeter is either a shared edge, an envelope edge, or interior to the room.
    # Prefer the wall band when wall bodies were supplied; fall back to shared grid edges
    # for a drawing that gave us no thicknesses to work with.
    if wall_cells:
        result.boundaries = _boundaries_from_wall_band(
            owner,
            wall_cells,
            space_id,
            result.spaces,
            minx,
            miny,
            cell,
            centre,
            usable,
            wall_segments,
            cfg,
        )
        # Two rooms can also meet with *no band between them* — the drawing simply has no
        # wall there. Those pairs get nothing from the band derivation, which is why every
        # dwelling came back with unsealed rooms: half of some room perimeters had no wall
        # within 120 mm because none was ever built. Their shared grid edge supplies the
        # boundary, and a wall is synthesised on it downstream unless the pair is one that
        # is genuinely continuous (living into dining).
        # Filtering these out per *pair* was wrong: a bedroom and its balcony can have a
        # wall along part of their shared edge and touch directly along the rest, and
        # dropping the whole pair left the touching part with no wall at all. Overlap is
        # judged per *line* instead — a shared edge already covered by a band contributes
        # nothing, one that is not is a stretch the drawing left open.
        # Nothing to add: the band derivation now covers all three cases — walls between
        # rooms, the building envelope, and rooms that touch with no wall between them —
        # and each one is skeletonised, so a straight stretch is one wall rather than one
        # per grid step.
    else:
        result.boundaries = _boundaries_from_cells(
            owner, space_id, minx, miny, cell, centre, usable, wall_segments, cfg
        )

    cov = result.coverage()
    result.notes.append(
        f"mapped {len(result.spaces)} spaces covering {cov:.1%} of the dwelling "
        f"footprint ({len(result.boundaries)} boundaries)"
    )
    if cov < 0.9:
        result.warnings.append(
            f"space map covers only {cov:.0%} of the footprint — labels may be missing"
        )
    weak = [s for s in result.spaces if (a := s.agreement()) is not None and a < 0.6]
    if weak:
        names = ", ".join(f"{s.label} {mm_to_sqft(s.area):.0f}/{mm_to_sqft(s.target_area or 0):.0f}sf" for s in weak[:5])
        result.warnings.append(
            f"{len(weak)} space(s) differ from their printed area by more than 40% ({names})"
        )
    return result


def _straighten_diagonals(
    owner: dict[tuple[int, int], int],
    cell_area: float,
    max_shift: float = 0.22,
    blocked: "Callable[[tuple[int, int], tuple[int, int]], bool] | None" = None,
) -> int:
    """Replace a diagonal interface between two spaces with a single straight one.

    Where two spaces meet without a wall to divide them, their growth fronts collide
    along a 45-degree line. That is geometrically reasonable and architecturally absurd:
    it came out as zig-zag walls stepping across a lobby and a toilet, and no
    neighbourhood filter can remove it — on a staircase every tooth has exactly two
    neighbours of each owner, so majority rules see nothing to fix.

    The interface has to be replaced wholesale. For each adjacent pair, the shared faces
    are collected; if they are spread over more than two grid lines in both directions
    the interface is diagonal, and the cells the two spaces own in that region are
    re-split along one axis-aligned line at the median of the existing interface.

    Reverted for that pair if it would move more than ``max_shift`` of either space's
    area — better a stepped boundary than a room resized to make a wall straight.
    """
    pairs: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for c, me in owner.items():
        for nb in _n4(c):
            them = owner.get(nb)
            if them is None or them == me:
                continue
            pairs.setdefault((min(me, them), max(me, them)), []).append(c)

    straightened = 0
    for (a, b), seam in pairs.items():
        rows = {c[1] for c in seam}
        cols = {c[0] for c in seam}
        if len(rows) < 3 or len(cols) < 3:
            continue  # already essentially straight

        # A seam that runs along a drawn wall is already exactly where it belongs;
        # "straightening" it would move the boundary off the wall. Only the seams the
        # drawing does not dictate are ours to tidy.
        if blocked is not None:
            on_wall = sum(
                1
                for c in seam
                for nb in _n4(c)
                if owner.get(nb) in (a, b) and owner.get(nb) != owner[c] and blocked(c, nb)
            )
            if on_wall >= max(2, len(seam) // 4):
                continue

        i0, i1 = min(cols), max(cols)
        j0, j1 = min(rows), max(rows)
        region = [c for c, o in owner.items() if o in (a, b) and i0 <= c[0] <= i1 and j0 <= c[1] <= j1]
        if not region:
            continue
        before = {a: sum(1 for c in owner.values() if c == a), b: sum(1 for c in owner.values() if c == b)}

        horizontal = (i1 - i0) >= (j1 - j0)
        axis = 1 if horizontal else 0
        split = sorted(c[axis] for c in seam)[len(seam) // 2]
        mean_a = sum(c[axis] for c in region if owner[c] == a) / max(
            1, sum(1 for c in region if owner[c] == a)
        )
        mean_b = sum(c[axis] for c in region if owner[c] == b) / max(
            1, sum(1 for c in region if owner[c] == b)
        )
        low, high = (a, b) if mean_a <= mean_b else (b, a)

        undo = {c: owner[c] for c in region}
        for c in region:
            owner[c] = low if c[axis] < split else high

        after = {a: sum(1 for c in owner.values() if c == a), b: sum(1 for c in owner.values() if c == b)}
        moved = max(
            abs(after[k] - before[k]) / max(before[k], 1) for k in (a, b)
        )
        if moved > max_shift:
            owner.update(undo)
        else:
            straightened += 1
    return straightened


def _band_runs(
    band: set[tuple[int, int]],
    minx: float,
    miny: float,
    cell: float,
    min_length: float,
) -> list[tuple[LineString, float]]:
    """Skeletonise a band of wall cells into straight centrelines with thicknesses.

    Fitting one line per band was the flaw that stopped wall-body exclusion from working.
    A band is not a line: an L-shaped wall between two rooms is one connected band, and a
    single fitted line lies along neither leg, so half the wall vanished and rooms came
    back unsealed.

    A wall *is* locally straight, though, so each cell can be asked which way the wall runs
    through it — compare how far the band extends horizontally through that cell against
    how far it extends vertically. Cells agreeing on a direction and touching form one run;
    the corner belongs to whichever leg is longer, which is where a drawn corner sits
    anyway. Thickness falls out of the run's own depth rather than being assumed.
    """
    if not band:
        return []

    def extent(c: tuple[int, int], di: int, dj: int) -> int:
        n, (i, j) = 1, c
        k, l = i - di, j - dj
        while (k, l) in band:
            n += 1
            k, l = k - di, l - dj
        k, l = i + di, j + dj
        while (k, l) in band:
            n += 1
            k, l = k + di, l + dj
        return n

    horizontal = {c for c in band if extent(c, 1, 0) >= extent(c, 0, 1)}
    vertical = band - horizontal

    runs: list[tuple[LineString, float]] = []
    for subset, is_h in ((horizontal, True), (vertical, False)):
        for comp in _components(subset):
            cols = [i for i, _ in comp]
            rows = [j for _, j in comp]
            if is_h:
                length = (max(cols) - min(cols) + 1) * cell
                depth = (max(rows) - min(rows) + 1) * cell
                if length < min_length:
                    continue
                y = miny + (sum(rows) / len(rows) + 0.5) * cell
                line = LineString(
                    [(minx + min(cols) * cell, y), (minx + (max(cols) + 1) * cell, y)]
                )
            else:
                length = (max(rows) - min(rows) + 1) * cell
                depth = (max(cols) - min(cols) + 1) * cell
                if length < min_length:
                    continue
                x = minx + (sum(cols) / len(cols) + 0.5) * cell
                line = LineString(
                    [(x, miny + min(rows) * cell), (x, miny + (max(rows) + 1) * cell)]
                )
            runs.append((line, depth))
    return runs


def _boundaries_from_wall_band(
    owner: dict[tuple[int, int], int],
    wall_cells: dict[tuple[int, int], tuple[float, float]],
    space_id: dict[int, str],
    spaces: Sequence[Space],
    minx: float,
    miny: float,
    cell: float,
    centre: dict[tuple[int, int], tuple[float, float]],
    seeds: Sequence[SpaceSeed],
    wall_segments: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    cfg: MapConfig,
) -> list[Boundary]:
    """Boundaries taken from the *band of wall cells* separating two rooms.

    With wall bodies excluded from the partition, two rooms are no longer edge-to-edge —
    the wall between them is a band of unowned cells. That band is the wall, and its
    centreline is the boundary, which is a better answer than the shared grid edge the
    previous derivation used: it comes from where the wall actually is rather than from
    where two growth fronts happened to meet.

    A band touching only one room faces outside: that is the envelope.
    """
    walls = [LineString([a, b]) for a, b in wall_segments if math.dist(a, b) > 1e-6]
    wall_union = unary_union(walls) if walls else None

    # Which rooms does each wall cell touch? Diagonals included, so a cell in the middle of
    # a thick wall still associates with the rooms either side.
    touch: dict[tuple[int, int], set[int]] = {}
    for c in wall_cells:
        i, j = c
        near: set[int] = set()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                o = owner.get((i + di, j + dj))
                if o is not None:
                    near.add(o)
        touch[c] = near

    # Group the band cells by which pair of rooms they divide — **every** pair, not the
    # first two.
    #
    # `sorted(near)[:2]` was silently dropping walls. A cell at a T-junction, or anywhere a
    # wall thickens near a corner, touches three or more rooms; filing it under one pair
    # left the other pairs with no band and therefore no wall. That is why room edges came
    # back with the nearest wall 133-1508 mm away: not misplaced, absent. Only 2 of 23
    # exterior walls were actually mis-positioned, which is what ruled out the placement
    # theory.
    bands: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for c, near in touch.items():
        if not near:
            continue
        if len(near) == 1:
            bands.setdefault((min(near), -1), []).append(c)  # envelope
            continue
        for a, b in itertools.combinations(sorted(near), 2):
            bands.setdefault((a, b), []).append(c)

    # The envelope comes from each room's own outline, not from its edge cells.
    #
    # Skeletonising the edge cells classifies each one as horizontal or vertical, and on a
    # jagged building edge those alternate — so the strip broke into components all shorter
    # than the minimum and every one was dropped. Four stretches totalling 7.5 m of room
    # perimeter ended up with no boundary and therefore no wall, which was the single
    # largest cause of unsealed rooms.
    #
    # `_straight_runs` is built for exactly this: it simplifies a ring into straight
    # stretches and tolerates the stepping. What is already covered by a wall band or by a
    # direct-touch strip is subtracted first, so this only supplies what nothing else has.
    # Rooms that touch *directly*, with no wall cells between them, kept in their own
    # strips.
    #
    # Two mistakes were stacked here. Skipping any pair that already had a band elsewhere
    # meant a bedroom part-walled and part-touching a living room got nothing for the
    # touching stretch — 4.3 m of unsealed edge, and the same per-pair filtering error I had
    # already fixed in one place and left in another. And folding these cells into the wall
    # bands would corrupt the thickness, because these are *room* cells: their depth says
    # nothing about how thick the wall should be.
    touch: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for c, o in owner.items():
        for nb in _n4(c):
            other = owner.get(nb)
            if other is None or other == o:
                continue
            touch.setdefault((min(o, other), max(o, other)), set()).add(c)

    out: list[Boundary] = []
    for (a_idx, b_idx), strip in touch.items():
        if a_idx not in space_id or b_idx not in space_id:
            continue
        ka, kb = seeds[a_idx].kind, seeds[b_idx].kind
        continuous = (ka.is_outdoor and kb.is_outdoor) or (
            ka in _OPEN_PLAN and kb in _OPEN_PLAN
        )
        for seg, _depth in _band_runs(strip, minx, miny, cell, cfg.min_boundary_mm):
            out.append(
                Boundary(
                    space_id[a_idx],
                    space_id[b_idx],
                    seg,
                    cfg.interior_wall_mm,
                    not continuous,
                )
            )

    for (a_idx, b_idx), band in bands.items():
        if a_idx not in space_id or (b_idx >= 0 and b_idx not in space_id):
            continue
        if b_idx < 0:
            continue  # envelope handled from the outline, below
        for seg, depth in _band_runs(set(band), minx, miny, cell, cfg.min_boundary_mm):
            envelope = b_idx < 0
            thickness = depth if not envelope else None
            t = None
            if wall_union is not None:
                mid = seg.interpolate(0.5, normalized=True)
                if wall_union.distance(mid) <= cfg.wall_snap_mm:
                    nearest = min(walls, key=lambda w: w.distance(mid))
                    t = getattr(nearest, "_thickness", None)

            if envelope:
                walled = True
            else:
                ka, kb = seeds[a_idx].kind, seeds[b_idx].kind
                continuous = (ka.is_outdoor and kb.is_outdoor) or (
                    ka in _OPEN_PLAN and kb in _OPEN_PLAN
                )
                walled = True  # a band of wall cells *is* a wall
                if continuous and t is None:
                    walled = False
            out.append(
                Boundary(
                    space_id[a_idx],
                    None if envelope else space_id[b_idx],
                    seg,
                    t
                    or thickness
                    or (cfg.exterior_wall_mm if envelope else cfg.interior_wall_mm),
                    walled,
                    exterior=envelope,
                )
            )

    by_id = {sp.id: sp for sp in spaces}
    covered_by_id: dict[str, list[LineString]] = {}
    for b in out:
        covered_by_id.setdefault(b.space_a, []).append(b.line)
        if b.space_b:
            covered_by_id.setdefault(b.space_b, []).append(b.line)
    for sid, sp_obj in by_id.items():
        done = covered_by_id.get(sid) or []
        ring = sp_obj.polygon.exterior
        remainder = (
            ring.difference(unary_union(done).buffer(cell * 1.2)) if done else ring
        )
        for seg in _straight_runs(remainder, cfg.min_boundary_mm):
            mid = seg.interpolate(0.5, normalized=True)
            # Only claim this as envelope if there is genuinely nothing beyond it.
            #
            # Claiming every uncovered stretch masked missing adjacency instead of
            # revealing it: two toilets came back with all four sides marked exterior —
            # sealed in a box with no neighbour and therefore no possible door. A stretch
            # with another room a wall's width beyond it is an interior wall whose pair
            # boundary went missing, and it must stay visible rather than be papered over.
            (x0, y0), (x1, y1) = seg.coords[0], seg.coords[-1]
            dx, dy = x1 - x0, y1 - y0
            n = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / n, dx / n
            reach = cfg.exterior_wall_mm + cell * 2.0
            beyond = None
            for direction in (1.0, -1.0):
                probe = Point(mid.x + nx * reach * direction, mid.y + ny * reach * direction)
                if sp_obj.polygon.contains(probe):
                    continue  # that way is into this room
                for other_id, other in by_id.items():
                    if other_id != sid and other.polygon.contains(probe):
                        beyond = other_id
                        break
                if beyond:
                    break
            t = None
            if wall_union is not None and wall_union.distance(mid) <= cfg.wall_snap_mm:
                nearest = min(walls, key=lambda w: w.distance(mid))
                t = getattr(nearest, "_thickness", None)
            out.append(
                Boundary(
                    sid,
                    beyond,
                    seg,
                    t or (cfg.interior_wall_mm if beyond else cfg.exterior_wall_mm),
                    True,
                    exterior=beyond is None,
                )
            )
    return out


def _boundaries_from_cells(
    owner: dict[tuple[int, int], int],
    space_id: dict[int, str],
    minx: float,
    miny: float,
    cell: float,
    centre: dict[tuple[int, int], tuple[float, float]],
    seeds: Sequence[SpaceSeed],
    wall_segments: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    cfg: MapConfig,
) -> list[Boundary]:
    """Every boundary in the partition, read straight off the grid.

    Walks each owned cell's four faces. A face whose neighbour has a *different* owner
    is a shared boundary; a face whose neighbour is outside the footprint is an envelope
    boundary; anything else is interior to a room and not a boundary at all. Faces on
    the same grid line between the same pair are then merged into runs.

    Exact and complete by construction, with no tolerance to tune.
    """
    walls = [LineString([a, b]) for a, b in wall_segments if math.dist(a, b) > 1e-6]
    wall_union = unary_union(walls) if walls else None

    # key: (orientation, line index, owner pair) -> set of positions along the line
    faces: dict[tuple[str, int, tuple[int, int]], set[int]] = {}
    for (i, j), me in owner.items():
        for di, dj, orient, line_idx, pos in (
            (1, 0, "V", i + 1, j),
            (-1, 0, "V", i, j),
            (0, 1, "H", j + 1, i),
            (0, -1, "H", j, i),
        ):
            nb = (i + di, j + dj)
            them = owner.get(nb, -1 if nb in centre else -2)
            if them == me:
                continue  # interior to this room
            pair = (min(me, them), max(me, them))
            faces.setdefault((orient, line_idx, pair), set()).add(pos)

    out: list[Boundary] = []
    for (orient, line_idx, pair), positions in faces.items():
        a_idx, b_idx = pair
        # -2 is outside the footprint (the envelope). -1 is an unowned cell inside it,
        # which phase 2 should have eliminated; treat it as envelope rather than lose it.
        if a_idx < 0 and b_idx < 0:
            continue
        # `other_idx` used to be `a_idx if a_idx >= 0 else -1`, which for a genuine
        # shared boundary returned a_idx — the *same* space as `inside_idx`. Every
        # boundary in the flat therefore claimed to be between a room and itself, so no
        # two spaces were ever adjacent, no partition walls were built between them, and
        # the rooms came out as open boxes. The pair is (a, b); the envelope is the case
        # where one of them is negative.
        if a_idx < 0:
            inside_idx, other_idx = b_idx, -1
        else:
            inside_idx, other_idx = a_idx, b_idx
        if inside_idx not in space_id:
            continue
        if other_idx >= 0 and other_idx not in space_id:
            continue

        for run_start, run_end in _merge_runs(sorted(positions)):
            lo = run_start * cell
            hi = (run_end + 1) * cell
            if orient == "V":
                x = minx + line_idx * cell
                seg = LineString([(x, miny + lo), (x, miny + hi)])
            else:
                y = miny + line_idx * cell
                seg = LineString([(minx + lo, y), (minx + hi, y)])
            # No minimum length. A short run is a step in the partition, and a step with
            # no wall on it is a hole in the room — dropping them under a 300 mm floor is
            # what left twelve rooms leaking.
            envelope = other_idx < 0
            t = None
            if wall_union is not None:
                mid = seg.interpolate(0.5, normalized=True)
                if wall_union.distance(mid) <= cfg.wall_snap_mm:
                    nearest = min(walls, key=lambda w: w.distance(mid))
                    t = getattr(nearest, "_thickness", None)

            if envelope:
                walled = True
            else:
                ka = seeds[a_idx].kind if a_idx >= 0 else RoomKind.UNKNOWN
                kb = seeds[b_idx].kind
                # Two outdoor spaces are one outdoor space. The balcony strips come out
                # split because the band that peels them off the building edge is cut by
                # every wall it meets, and putting a wall between the pieces fenced a
                # continuous wrap-around balcony into compartments.
                continuous = (ka.is_outdoor and kb.is_outdoor) or (
                    ka in _OPEN_PLAN and kb in _OPEN_PLAN
                )
                walled = t is not None or not continuous

            out.append(
                Boundary(
                    space_id[inside_idx],
                    None if envelope else space_id[other_idx],
                    seg,
                    t or (cfg.exterior_wall_mm if envelope else cfg.interior_wall_mm),
                    walled,
                    exterior=envelope,
                )
            )
    return out


def _merge_runs(positions: Sequence[int]) -> list[tuple[int, int]]:
    """Consecutive integers collapsed into (first, last) spans."""
    runs: list[tuple[int, int]] = []
    for p in positions:
        if runs and p == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], p)
        else:
            runs.append((p, p))
    return runs


def _boundaries(
    spaces: Sequence[Space],
    footprint: Polygon,
    wall_segments: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    cfg: MapConfig,
    basis: float = 0.0,
) -> list[Boundary]:
    """Shared edges between spaces, and each space's edges facing outside.

    A boundary carries whether a wall was actually *recovered* along it. That single
    flag is what later lets living-and-dining be modelled as one continuous space with
    an open threshold, instead of two rooms separated by a wall nobody drew.
    """
    walls = [LineString([a, b]) for a, b in wall_segments if math.dist(a, b) > 1e-6]
    wall_union = unary_union(walls) if walls else None
    envelope = footprint.exterior

    def measured(line: LineString) -> float | None:
        """Thickness from a recovered wall lying along this boundary, if there is one.

        Sampled at three points rather than just the midpoint: a wall that merely
        crosses the boundary near its middle is not this boundary's wall, and matching
        on the midpoint alone let those through.
        """
        if wall_union is None:
            return None
        probes = [line.interpolate(f, normalized=True) for f in (0.25, 0.5, 0.75)]
        hits = [p for p in probes if wall_union.distance(p) <= cfg.wall_snap_mm]
        if len(hits) < 2:
            return None
        best = min(walls, key=lambda w: w.distance(hits[0]))
        return getattr(best, "_thickness", None)

    # Wide enough to bridge rectification drift. The raster partition hands over
    # polygons that share edges exactly, but `_rectify` squares each one up
    # independently, so two neighbours can end up a hair apart. At 0.7 cells (~63 mm)
    # that hair was enough to miss the shared edge entirely: a bedroom and the passage
    # beside it registered as non-adjacent, so no wall was built between them and the
    # bedroom was left open to the corridor for 3.9 m of its perimeter.
    #
    # There is no risk of over-bridging: spaces tile the footprint, so anything this
    # close really is the neighbour.
    eps = cfg.cell_mm * 2.2
    out: list[Boundary] = []
    for i, a in enumerate(spaces):
        for b in spaces[i + 1 :]:
            touching = a.polygon.exterior.intersection(b.polygon.buffer(eps))
            for run in _straight_runs(touching, cfg.min_boundary_mm, basis):
                t = measured(run)
                # Default to walled. Absence of a recovered wall is *weak* evidence of
                # an opening and strong evidence of nothing: the face-pairing misses
                # partitions routinely. Treating every unmatched boundary as open plan
                # left 41 of 109 boundaries wall-less, which made every bedroom
                # continuous with the corridor and every room trivially "reachable", so
                # not one door got placed.
                #
                # An open threshold therefore needs positive evidence, and the only
                # honest evidence available is the pair of space types. Living into
                # dining is routinely one continuous room. A bedroom into a corridor
                # never is.
                open_pair = a.kind in _OPEN_PLAN and b.kind in _OPEN_PLAN
                walled = t is not None or not open_pair
                out.append(
                    Boundary(a.id, b.id, run, t or cfg.interior_wall_mm, walled)
                )

        # Edges of this space that face outside the dwelling. "Not covered by another
        # space" is not sufficient evidence of that: subtracting the neighbours' buffered
        # outlines leaves slivers wherever two spaces meet at a slight offset, and each
        # sliver was becoming an exterior wall. 38 of 70 walls came out flagged exterior
        # and rendered with the dark outside finish, which is what the black bars along
        # the interior wall tops were. An exterior wall has to be on the *footprint's*
        # edge, so that is what gets checked.
        others = [o.polygon.buffer(eps) for o in spaces if o is not a]
        outer = a.polygon.exterior.difference(unary_union(others)) if others else a.polygon.exterior
        for run in _straight_runs(outer, cfg.min_boundary_mm, basis):
            # Whether this edge faces *outside the building* is a separate question from
            # whether it gets a wall. It always gets a wall — it is a piece of a room's
            # perimeter with no other room behind it, so if nothing is built there the
            # room is an open box. Skipping the runs that were not on the envelope left
            # exactly that: a toilet 57% enclosed, a kitchen 79%, a bedroom 74%. In the
            # render that is a maze of disconnected wall stubs, which is what it looked
            # like.
            #
            # Only the *classification* depends on the envelope: on it, this is an
            # external wall in the external finish; off it, it is an internal partition
            # against a gap the space map could not attribute.
            on_envelope = (
                envelope.distance(run.interpolate(0.5, normalized=True)) <= cfg.wall_snap_mm * 2
            )
            t = measured(run)
            out.append(
                Boundary(
                    a.id,
                    None,
                    run,
                    t or (cfg.exterior_wall_mm if on_envelope else cfg.interior_wall_mm),
                    True,
                    exterior=on_envelope,
                )
            )
    return out


def _straight_runs(geom: object, min_length: float, basis: float = 0.0) -> list[LineString]:
    """Split a boundary into its straight runs — one per wall.

    This has to be done, and getting it wrong was costing real accuracy. A shared
    boundary between two spaces is often L-shaped (a living room wrapping round a
    kitchen), and reducing that to a single line — the earlier approach took the
    overlap's longest diagonal — produces a diagonal that lies along neither leg. It
    matched no recovered wall, so both legs got synthesised at the wrong angle.

    An L-shaped boundary is two walls. Splitting on direction change says so.

    Every millimetre of the input must end up in some run. Dropping the pieces shorter
    than ``min_length`` instead of absorbing them was leaving holes in room perimeters —
    a toilet came out 70% enclosed, a kitchen 79%, a bedroom 74% — because a small jog
    between two long runs produced no wall at all, and a room with a hole in it is an
    open box. A split is therefore only taken when *both* sides of it are long enough to
    be walls in their own right; otherwise the jog is absorbed into the run it interrupts.
    """
    if geom is None or getattr(geom, "is_empty", True):
        return []

    # Contiguous pieces first. A raster outline breaks the shared edge into many
    # touching fragments, and simplifying them separately left a 3.4 m balcony
    # threshold as four sub-metre ones.
    if hasattr(geom, "geoms"):
        merged = linemerge([g for g in geom.geoms if isinstance(g, LineString)])
        geom = merged if not merged.is_empty else geom

    runs: list[LineString] = []
    for part in getattr(geom, "geoms", [geom]):
        if not isinstance(part, LineString) or part.length < 1e-6:
            continue
        # Raster-derived outlines are staircases; simplifying first collapses each
        # tread into the straight run it approximates.
        coords = list(part.simplify(min_length * 0.5, preserve_topology=False).coords)
        if len(coords) < 2:
            continue

        def seg_len(i: int, j: int) -> float:
            return math.dist(coords[i], coords[j])

        start = 0
        for k in range(1, len(coords) - 1):
            ax, ay = coords[k][0] - coords[start][0], coords[k][1] - coords[start][1]
            bx, by = coords[k + 1][0] - coords[k][0], coords[k + 1][1] - coords[k][1]
            la, lb = math.hypot(ax, ay), math.hypot(bx, by)
            if la < 1e-6 or lb < 1e-6:
                continue
            if (ax * bx + ay * by) / (la * lb) >= 0.94:  # within ~20 degrees: same run
                continue
            # Only break here if both halves stand on their own as walls. A short jog
            # otherwise gets absorbed, keeping the perimeter continuous.
            if la >= min_length and seg_len(k, len(coords) - 1) >= min_length:
                runs.append(LineString(coords[start : k + 1]))
                start = k
        runs.append(LineString(coords[start:]))

    # Reduce each run to its chord. A wall is straight; the residual jitter left by
    # simplification is raster noise, not architecture. Nothing is discarded here —
    # a run shorter than `min_length` survived absorption because it is a whole edge of
    # the room, and a room needs it.
    out: list[LineString] = []
    for r in runs:
        if math.dist(r.coords[0], r.coords[-1]) < 1.0:
            continue
        out.append(_snap_to_basis(LineString([r.coords[0], r.coords[-1]]), basis))
    return out
