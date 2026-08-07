"""Extract :class:`~atlas.planir.PlanIR` from a DXF drawing.

The pipeline, and why each stage exists:

1. **Load and flatten.** Repair the byte stream, then resolve the block hierarchy
   into world space (see :mod:`atlas.ingest.dxf_repair` and
   :mod:`atlas.ingest.dxf_flatten`).

2. **Establish scale.** Millimetres or nothing. ``$INSUNITS`` when it is set and
   believable, extents-based inference otherwise, and the reasoning is recorded.

3. **Segment into dwellings.** A production drawing is a whole project. The demo
   file holds 16 apartments across several towers at survey coordinates. We
   prefer the drawing's own carpet-area polygons as unit boundaries — they are
   what the developer legally declares, so they are authoritative — and fall back
   to clustering room labels when the drawing has none.

4. **Per dwelling, in this order: map the house, then walls, then openings.**
   The order is the whole design, and it used to be wrong.

   It originally ran walls-first and took the leftover cycles of the wall network
   as the rooms. That made rooms a *derivative of the noisiest step*, so every
   partition the face-pairing missed became a hole in the room map: 24% of the
   demo flat's footprint belonged to no room, four fragments came out unlabelled,
   and neighbouring spaces sat 150–240 mm apart with nothing between them.

   It now runs space-first, because that is the direction the evidence actually
   points. The drawing *declares* its spaces — sixteen labels, each with a printed
   size — and that is ground truth. Wall geometry is exact where it was recovered
   and simply missing where it was not. So:

   a. **Map the complete house** (:mod:`atlas.ingest.spaces`). Every label becomes
      a space, grown to its printed area over a raster of the dwelling footprint,
      with recovered walls acting as *barriers* to that growth. The partition is
      complete by construction — no unknown regions, no voids — and it yields the
      adjacency graph directly.
   b. **Derive walls from the space boundaries.** A shared boundary becomes one
      wall straddling it, so two adjoining rooms cannot drift apart. Recovered CAD
      geometry supplies the real thickness wherever it exists; boundaries with no
      recovered wall are *open thresholds* (living into dining) and get no wall at
      all, which is what the drawing shows.
   c. **Place openings on those boundaries**, so every space is reachable.

5. **Reconcile against stated intent.** Every extracted room is checked against
   its printed ``11'0"X13'6"`` and the unit against its declared carpet area.
   Disagreement is reported, never silently averaged away.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from shapely.geometry import LineString, MultiPolygon, Point as ShpPoint, Polygon as ShpPolygon, box as shp_box
from shapely.ops import polygonize, unary_union

from ..planir import (
    FixtureHint,
    Level,
    Opening,
    OpeningKind,
    PlanIR,
    PlanUnit,
    Room,
    RoomKind,
    Source,
    Wall,
    classify_room_name,
)
from .spaces import Boundary, MapConfig, SpaceMap, SpaceSeed, map_spaces
from ..units import (
    INSUNITS_TO_MM,
    Dimension,
    infer_scale_from_extents,
    mm_to_sqft,
    parse_area_sqft,
    parse_dimension_pair,
)
from .dxf_flatten import FlatEntity, flatten_layout
from .dxf_repair import load_dxf
from .layers import LayerRole, classify_layer
from .wall_solver import Segment, SolverConfig, solve_walls

__all__ = ["ExtractConfig", "extract_plan", "dwg_to_dxf"]

_TRACE_HEAL = os.environ.get("ATLAS_TRACE_HEAL", "").strip().upper() or None
"""Set to a room-label substring (e.g. ``TOILET``) to print each healing-pass guard
decision for matching spaces. Diagnostic only; costs nothing when unset."""


@dataclass(slots=True)
class ExtractConfig:
    """Tunables for extraction. Defaults suit residential plans in millimetres."""

    wall_height_mm: float = 2900.0
    parapet_height_mm: float = 1050.0
    """Balcony parapets. Standard guard height; anything taller encloses the
    balcony and ruins the elevation."""

    slab_thickness_mm: float = 150.0
    default_door_height_mm: float = 2100.0
    default_window_height_mm: float = 1200.0
    default_window_sill_mm: float = 900.0

    min_unit_area_sqft: float = 200.0
    max_unit_area_sqft: float = 8000.0
    """Filters the carpet-area layer: it also carries the plot boundary
    (636,000 sqft in the demo file) and tiny 5 sqft niche polygons."""

    unit_clip_margin_mm: float = 1400.0
    """How far outside the declared boundary to look for the unit's own geometry.

    A declared carpet area excludes wall thickness, so the unit's walls — and the
    doors and windows *in* those walls — sit outside the polygon. 400 mm covers the
    walls but clipped away much of the opening line work; 1400 mm was measured to
    recover 12 openings instead of 8 and one extra room, with no change to the
    measured-vs-printed area agreement. Beyond this the neighbouring dwelling's
    geometry starts being pulled in."""

    min_room_area_mm2: float = 900_000.0
    """0.9 m². Below this a region is a duct, a wall cavity, or solver noise."""

    barrier_extend_mm: float = 150.0
    """How far to extend an opening barrier past the symbol's ends, so it meets the
    wall stubs either side of the opening and closes the cycle."""

    junction_extend_mm: float = 120.0
    """How far to extend wall centrelines past their ends before noding the graph,
    so a T-junction whose stem stops short of the through-wall still closes its
    room. Larger than the drawing's sloppiness, smaller than a doorway."""

    max_room_area_mm2: float = 120_000_000.0
    """120 m². A single region larger than this means walls failed to close and
    rooms leaked into one another — worth a warning, not silent acceptance."""

    label_search_radius_mm: float = 6000.0
    door_arc_radius_range: tuple[float, float] = (450.0, 1600.0)
    door_cluster_radius_mm: float = 1200.0
    """Door line work within this distance belongs to the same door leaf."""
    sliding_door_min_mm: float = 1300.0
    """At or above this width, a threshold to a balcony is a sliding door rather than
    a hinged one — which is what an opening that wide physically has to be."""

    balcony_door_height_mm: float = 2100.0
    """Full door height, no sill. A balcony threshold is walked through."""

    min_gap_opening_mm: float = 550.0
    """Narrowest wall-face gap accepted as an opening."""

    symbol_match_mm: float = 1400.0
    """How close a door or window symbol must be to a gap to name its kind."""

    raster_heal_mm: float = 550.0
    """How far a room may grow past its rasterised edge to reach the wall face the
    drawing shows. Sized to cover the worst measured standoff: a raster edge can sit a
    cell (90) from where growth stopped, the reach cap can stop growth ~200 short in an
    under-printed room, and wall snapping can move the wall another 260 — traced gaps ran
    415-480 mm. The growth is cut back by the wall solids and drawn ribbons, blocked from
    every other space's polygon, clipped to the declared carpet boundary, and discarded if
    it would create a hole — so its reach is bounded by what it may legitimately claim,
    not by this number."""

    footprint_min_coverage: float = 0.7
    """How much of the declared area the wall cycles must enclose to be trusted as the
    footprint. Below this the walls plainly did not close and the declaration is better."""

    declared_fallback_margin_mm: float = 350.0
    """Outward margin on the declared polygon when it is used as the footprint, to take in
    the wall thickness a carpet area excludes."""

    raw_barrier_explained_mm: float = 200.0
    """How close a drawn line must be to an already-solved wall centreline to count as
    one of its faces rather than as evidence of a wall we missed. Slightly over half the
    thickest common wall, so both faces of a 230 mm wall are absorbed."""

    raw_barrier_min_mm: float = 900.0
    """Shortest piece of drawn wall line work treated as a division during space growth.
    Long enough to skip hatch ticks and symbol detail, short enough to catch a real pier."""

    railing_height_mm: float = 1100.0
    railing_thickness_mm: float = 40.0
    """A glass balcony guard. 1100 mm is the usual minimum guard height; 40 mm covers
    laminated glass in a slim channel."""

    interior_wall_mm: float = 150.0
    exterior_wall_mm: float = 230.0
    """Fallback thicknesses used when a space boundary has no recovered wall to take a
    real measurement from. Standard Indian residential practice: 150 mm block
    partitions, 230 mm external. Recovered geometry always wins over these."""

    opening_attach_tol_mm: float = 900.0
    """How far an opening may sit from a wall centreline and still be attached."""

    opening_serves_tol_mm: float = 700.0
    """How close an opening's centre must be to a room's boundary to count as opening
    into that room."""

    oversize_split_ratio: float = 1.35
    """Split a single-label region when it exceeds its label's printed area by this
    much. Above the honest slack between a printed bounding W x D and a real
    non-rectangular room; below the point where two rooms have plainly merged."""

    dimension_agreement_warn: float = 0.75
    """Below this ratio between measured and printed area, warn the user."""


# ---------------------------------------------------------------------------
# DWG entry point
# ---------------------------------------------------------------------------

def dwg_to_dxf(dwg_path: Path | str, out_dir: Path | str | None = None) -> tuple[Path, list[str]]:
    """Convert DWG to DXF using LibreDWG's ``dwg2dxf``.

    LibreDWG is the only genuinely free, redistributable DWG reader, and it reads
    the AC1032 (AutoCAD 2018) files that practices actually send. It is also
    noisy: it prints thousands of "Unstable Class" warnings and writes DXF that
    needs repairing before ezdxf will touch it. Both are handled rather than
    treated as failures, because the alternative is refusing real input.
    """
    import shutil
    import subprocess

    dwg_path = Path(dwg_path)
    out_dir = Path(out_dir) if out_dir else dwg_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    dxf_path = out_dir / f"{dwg_path.stem}.dxf"

    exe = shutil.which("dwg2dxf")
    if exe is None:
        raise RuntimeError(
            "dwg2dxf not found. Install LibreDWG (`brew install libredwg`) or supply a DXF. "
            "DWG is a proprietary format with no pure-Python reader."
        )

    proc = subprocess.run(
        [exe, "-o", str(dxf_path), str(dwg_path)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    notes = [f"dwg2dxf exit={proc.returncode}"]
    if not dxf_path.exists() or dxf_path.stat().st_size == 0:
        tail = (proc.stderr or "")[-500:]
        raise RuntimeError(f"dwg2dxf produced no output. stderr tail: {tail}")

    # LibreDWG warns per entity; summarise rather than relay tens of thousands.
    warn_kinds = Counter(
        re.sub(r"\s*\(0x0\).*$", "", line).strip()
        for line in (proc.stderr or "").splitlines()
        if line.startswith("Warning:")
    )
    if warn_kinds:
        top = ", ".join(f"{k} ×{v}" for k, v in warn_kinds.most_common(3))
        notes.append(f"dwg2dxf emitted {sum(warn_kinds.values())} warnings ({top})")
    notes.append(f"wrote {dxf_path.name} ({dxf_path.stat().st_size / 1e6:.1f} MB)")
    return dxf_path, notes


# ---------------------------------------------------------------------------
# scale
# ---------------------------------------------------------------------------

def _resolve_scale(doc: Any, entities: Sequence[FlatEntity]) -> tuple[float, str]:
    """Decide millimetres-per-drawing-unit, with the reasoning."""
    insunits = 0
    try:
        insunits = int(doc.header.get("$INSUNITS", 0) or 0)
    except Exception:
        pass

    mm = INSUNITS_TO_MM.get(insunits, 0.0)
    if mm > 0:
        # Sanity-check the declaration against the geometry. A file declaring
        # metres whose extents span 20,000 units is mislabelled, and trusting the
        # header would produce a 20 km building.
        span = _entity_span(entities)
        if span > 0:
            phys_m = span * mm / 1000.0
            if 1.0 <= phys_m <= 100_000.0:
                return mm, f"$INSUNITS={insunits} -> {mm} mm/unit (extents imply {phys_m:.0f} m, plausible)"
            inferred, why = infer_scale_from_extents(span, span)
            return inferred, (
                f"$INSUNITS={insunits} would imply a {phys_m:.0f} m extent, which is implausible; "
                f"overriding to {inferred} mm/unit ({why})"
            )
        return mm, f"$INSUNITS={insunits} -> {mm} mm/unit"

    span = _entity_span(entities)
    inferred, why = infer_scale_from_extents(span, span)
    return inferred, f"$INSUNITS unset; {why}"


def _entity_span(entities: Sequence[FlatEntity]) -> float:
    xs: list[float] = []
    ys: list[float] = []
    for e in entities:
        for x, y in e.points:
            xs.append(x)
            ys.append(y)
    if not xs:
        return 0.0
    # Use a robust span: real drawings carry stray entities at the origin or at
    # absurd coordinates that would blow up a min/max range.
    xs.sort()
    ys.sort()
    lo, hi = int(len(xs) * 0.02), int(len(xs) * 0.98)
    hi = max(hi, lo + 1)
    return max(xs[min(hi, len(xs) - 1)] - xs[lo], ys[min(hi, len(ys) - 1)] - ys[lo])


# ---------------------------------------------------------------------------
# entity bucketing
# ---------------------------------------------------------------------------

@dataclass
class _Buckets:
    """Entities grouped by the role their layer implies."""

    by_role: dict[LayerRole, list[FlatEntity]]
    unknown_layers: dict[str, int]

    def get(self, *roles: LayerRole) -> list[FlatEntity]:
        out: list[FlatEntity] = []
        for r in roles:
            out.extend(self.by_role.get(r, []))
        return out


def _bucket(
    entities: Iterable[FlatEntity],
    overrides: dict[str, LayerRole] | None = None,
) -> _Buckets:
    """Group entities by layer role. ``overrides`` come from convention inference
    and apply only to layers the rule table classified as UNKNOWN."""
    by_role: dict[LayerRole, list[FlatEntity]] = defaultdict(list)
    unknown: Counter[str] = Counter()
    for e in entities:
        role = (overrides or {}).get(e.layer)
        if role is None:
            role = classify_layer(e.layer).role
        if role is LayerRole.UNKNOWN:
            unknown[e.layer] += 1
        by_role[role].append(e)
    return _Buckets(dict(by_role), dict(unknown))


def _scale_entity(e: FlatEntity, k: float) -> FlatEntity:
    if k == 1.0:
        return e
    return FlatEntity(
        e.dxftype,
        e.layer,
        [(x * k, y * k) for x, y in e.points],
        e.closed,
        e.text,
        e.block_path,
        e.depth,
        e.raw,
    )


# ---------------------------------------------------------------------------
# unit segmentation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _UnitRegion:
    """A candidate dwelling: boundary polygon plus the labels inside it."""

    polygon: ShpPolygon
    labels: list[FlatEntity]
    stated_area_sqft: float | None
    name: str
    source: str


def _shoelace(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    s = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _as_polygon(e: FlatEntity) -> ShpPolygon | None:
    if len(e.points) < 3:
        return None
    try:
        p = ShpPolygon(e.points)
        if not p.is_valid:
            p = p.buffer(0)  # standard self-intersection repair
        if isinstance(p, MultiPolygon):
            p = max(p.geoms, key=lambda g: g.area)
        return p if isinstance(p, ShpPolygon) and p.area > 0 else None
    except Exception:
        return None


def _find_units(buckets: _Buckets, cfg: ExtractConfig, notes: list[str]) -> list[_UnitRegion]:
    """Segment the drawing into dwellings.

    Preference order is deliberate: a declared carpet-area polygon is a legal
    document, a room-label cluster is a guess.
    """
    labels = [
        e
        for e in buckets.get(LayerRole.ROOM_TEXT)
        if e.text and e.points and parse_dimension_pair(e.text) is not None
    ]
    # Some drawings put room labels on the wall layer (the demo file does, for a
    # whole tower). Rescue any text that looks like "NAME\n<dims>" wherever it is.
    for role in (LayerRole.WALL, LayerRole.ANNOTATION, LayerRole.UNKNOWN):
        for e in buckets.get(role):
            if e.text and e.points and parse_dimension_pair(e.text) is not None:
                labels.append(e)
    if labels:
        notes.append(f"found {len(labels)} room labels carrying printed dimensions")

    # Candidate boundaries from the area-boundary layer, filtered to dwelling size.
    area_polys: list[ShpPolygon] = []
    rejected = 0
    for e in buckets.get(LayerRole.AREA_BOUNDARY):
        poly = _as_polygon(e)
        if poly is None:
            continue
        sqft = mm_to_sqft(poly.area)
        if cfg.min_unit_area_sqft <= sqft <= cfg.max_unit_area_sqft:
            area_polys.append(poly)
        else:
            rejected += 1

    # Deduplicate: the same unit is often outlined on several area layers
    # (carpet, FSI, built-up). Keep the largest of any heavily overlapping set.
    area_polys.sort(key=lambda p: -p.area)
    kept: list[ShpPolygon] = []
    for p in area_polys:
        if any(p.intersection(q).area > 0.6 * min(p.area, q.area) for q in kept):
            continue
        kept.append(p)
    if rejected:
        notes.append(
            f"area-boundary layer: kept {len(kept)} dwelling outlines, "
            f"rejected {rejected} outside {cfg.min_unit_area_sqft:.0f}-{cfg.max_unit_area_sqft:.0f} sqft "
            "(plot boundaries and niches)"
        )

    # Unit-type captions like "3 BHK\nCARPET AREA: 1056 SQFT".
    captions = [
        e
        for e in buckets.get(LayerRole.ANNOTATION, LayerRole.UNKNOWN, LayerRole.IGNORE, LayerRole.WALL)
        if e.text and e.points and parse_area_sqft(e.text) is not None
    ]

    regions: list[_UnitRegion] = []
    if kept:
        for i, poly in enumerate(kept):
            inside = [e for e in labels if poly.contains(ShpPoint(*e.points[0]))]
            measured = mm_to_sqft(poly.area)
            name, cap_area = _match_caption(poly, captions)
            # The polygon is the drawing's own boundary; a caption is text we
            # guessed belongs to it. When they disagree materially the match was
            # wrong (captions sit between units and get mis-assigned), so keep the
            # polygon's area and drop the caption's number.
            if cap_area is not None and min(cap_area, measured) / max(cap_area, measured) < 0.6:
                notes.append(
                    f"unit {i + 1}: ignoring nearby caption area {cap_area:.0f} sqft, "
                    f"{measured:.0f} sqft measured from its own boundary polygon"
                )
                cap_area = None
            regions.append(
                _UnitRegion(
                    polygon=poly,
                    labels=inside,
                    stated_area_sqft=cap_area if cap_area is not None else round(measured, 1),
                    name=name or f"Unit {i + 1}",
                    source="area_boundary",
                )
            )
        notes.append(f"segmented {len(regions)} dwellings from declared area boundaries")
        return regions

    # Fallback: cluster room labels spatially.
    clusters = _cluster_labels(labels, cfg.label_search_radius_mm)
    for i, group in enumerate(clusters):
        xs = [e.points[0][0] for e in group]
        ys = [e.points[0][1] for e in group]
        pad = cfg.label_search_radius_mm * 0.5
        poly = ShpPolygon(
            [
                (min(xs) - pad, min(ys) - pad),
                (max(xs) + pad, min(ys) - pad),
                (max(xs) + pad, max(ys) + pad),
                (min(xs) - pad, max(ys) + pad),
            ]
        )
        name, cap_area = _match_caption(poly, captions)
        regions.append(
            _UnitRegion(poly, group, cap_area, name or f"Unit {i + 1}", "label_cluster")
        )
    notes.append(
        f"no usable area boundaries; clustered room labels into {len(regions)} candidate dwellings "
        "(boundaries approximate)"
    )
    return regions


def _match_caption(
    poly: ShpPolygon, captions: Sequence[FlatEntity]
) -> tuple[str | None, float | None]:
    """Find the nearest ``N BHK / CARPET AREA`` caption for a unit outline."""
    if not captions:
        return None, None
    cx, cy = poly.centroid.x, poly.centroid.y
    best: tuple[float, FlatEntity] | None = None
    for e in captions:
        d = math.dist((cx, cy), e.points[0])
        if best is None or d < best[0]:
            best = (d, e)
    if best is None:
        return None, None
    dist, e = best
    # A caption further away than the unit's own diagonal belongs to another unit.
    diag = math.dist((poly.bounds[0], poly.bounds[1]), (poly.bounds[2], poly.bounds[3]))
    if dist > diag * 1.6:
        return None, None
    first = e.text.strip().splitlines()[0].strip() if e.text else ""
    name = first if re.search(r"\d\s*(?:\.\d+)?\s*BHK|BR\b|BEDROOM", first, re.I) else None
    return name, parse_area_sqft(e.text or "")


def _cluster_labels(labels: Sequence[FlatEntity], radius: float) -> list[list[FlatEntity]]:
    """Single-link spatial clustering of label points."""
    unassigned = list(labels)
    clusters: list[list[FlatEntity]] = []
    while unassigned:
        seed = unassigned.pop()
        group = [seed]
        frontier = [seed]
        while frontier:
            cur = frontier.pop()
            near = [
                e
                for e in unassigned
                if math.dist(cur.points[0], e.points[0]) <= radius
            ]
            for e in near:
                unassigned.remove(e)
                group.append(e)
                frontier.append(e)
        clusters.append(group)
    clusters.sort(key=len, reverse=True)
    return [c for c in clusters if len(c) >= 3]


# ---------------------------------------------------------------------------
# per-unit extraction
# ---------------------------------------------------------------------------

def _segments_from(entities: Iterable[FlatEntity], clip: ShpPolygon) -> list[Segment]:
    """Explode polylines into straight segments, keeping those inside ``clip``."""
    out: list[Segment] = []
    for e in entities:
        pts = e.points
        if len(pts) < 2:
            continue
        ring = list(pts) + ([pts[0]] if e.closed and len(pts) > 2 else [])
        for i in range(len(ring) - 1):
            a, b = ring[i], ring[i + 1]
            mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
            if clip.contains(ShpPoint(*mid)) or clip.contains(ShpPoint(*a)) or clip.contains(ShpPoint(*b)):
                out.append(Segment(a, b, e.layer))
    return out


def _wall_solid(walls: Sequence[Wall]) -> Any:
    """Union of wall footprints, for subtracting from the unit to leave rooms.

    Mitred joins and flat caps: a round cap would round off every corner and leave
    a crescent-shaped sliver of "room" outside the wall.
    """
    polys = []
    for w in walls:
        if w.length_mm < 1e-6:
            continue
        polys.append(
            LineString([w.start, w.end]).buffer(
                w.thickness_mm / 2.0, cap_style=2, join_style=2
            )
        )
    return unary_union(polys) if polys else None


def _enclosure_barriers(
    door_ents: Sequence[FlatEntity],
    window_ents: Sequence[FlatEntity],
    clip: ShpPolygon,
    cfg: ExtractConfig,
) -> list[LineString]:
    """Barrier lines across openings, so rooms are enclosed by doors and windows too.

    A room is not bounded by masonry alone. Two cases in this drawing make that
    unavoidable:

    * A bedroom opens onto its balcony through a full-width sliding door, leaving
      only 100 mm wall stubs at either jamb — too short to pair into walls. Without
      a barrier the bedroom and balcony are one region.
    * Every internal doorway is a gap cut through both wall faces, so the rooms
      either side connect through it.

    So the enclosure is walls *plus* openings, and the openings are recovered from
    the symbols the draughtsman drew.

    For a door we emit a barrier from the hinge to *each* arc endpoint rather than
    working out which one lies along the wall. This is safe and it is why the code
    can afford to be simple: planar subdivision only creates a face from a closed
    cycle, so the wrong barrier is a dangling edge that contributes nothing, while
    the right one seals the doorway. Guessing and getting it wrong would lose the
    room entirely.
    """
    bars: list[LineString] = []
    lo, hi = cfg.door_arc_radius_range

    # Doors are drawn as a leaf plus, *sometimes*, a swing arc — in this drawing
    # only 5 of 19 door entities are arcs, so arc-only detection misses most
    # doorways and every room behind them merges into its neighbour. Instead,
    # cluster each door's line work spatially and fence its bounding box.
    #
    # All four box edges are emitted. Three of them are wrong, and that is fine
    # for the same reason as above: an edge that closes no cycle creates no face.
    # The one along the wall seals the doorway. A spurious cycle would have to be
    # smaller than a door leaf, and is removed by the minimum-area filter.
    for cluster in _cluster_entities(door_ents, cfg.door_cluster_radius_mm):
        xs = [p[0] for e in cluster for p in e.points]
        ys = [p[1] for e in cluster for p in e.points]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        if max(w, h) < lo or max(w, h) > hi * 2.2:
            continue
        centre = ShpPoint((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
        if not clip.contains(centre):
            continue
        box = [
            (min(xs), min(ys)),
            (max(xs), min(ys)),
            (max(xs), max(ys)),
            (min(xs), max(ys)),
        ]
        for i in range(4):
            a, b = box[i], box[(i + 1) % 4]
            if math.dist(a, b) >= 300.0:
                bars.append(_extended_line(a, b, cfg.barrier_extend_mm))

    # Window and sill line work runs along the wall; each line is already a barrier.
    for s in _segments_from(window_ents, clip):
        if s.length >= 400.0:
            bars.append(_extended_line(s.a, s.b, cfg.barrier_extend_mm))
    return bars


def _cluster_entities(
    entities: Sequence[FlatEntity], radius: float
) -> list[list[FlatEntity]]:
    """Single-link cluster entities by proximity of their bounding-box centres.

    Used to gather the several primitives that together draw one door (leaf,
    swing arc, frame ticks) so they can be reasoned about as a single object.
    """
    items: list[tuple[tuple[float, float], FlatEntity]] = []
    for e in entities:
        if not e.points:
            continue
        xs = [p[0] for p in e.points]
        ys = [p[1] for p in e.points]
        items.append((((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0), e))

    remaining = list(items)
    clusters: list[list[FlatEntity]] = []
    while remaining:
        seed = remaining.pop()
        group = [seed]
        frontier = [seed]
        while frontier:
            (cx, cy), _ = frontier.pop()
            near = [it for it in remaining if math.dist((cx, cy), it[0]) <= radius]
            for it in near:
                remaining.remove(it)
                group.append(it)
                frontier.append(it)
        clusters.append([e for _, e in group])
    return clusters


def _extended_line(a: tuple[float, float], b: tuple[float, float], ext: float) -> LineString:
    """A segment lengthened at both ends, so it reliably nodes against its neighbours."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return LineString([a, b])
    ux, uy = dx / n, dy / n
    return LineString([(a[0] - ux * ext, a[1] - uy * ext), (b[0] + ux * ext, b[1] + uy * ext)])


def _room_faces(
    walls: Sequence[Wall],
    clip: ShpPolygon,
    cfg: ExtractConfig,
    barriers: Sequence[LineString] = (),
    keep_interiors: bool = False,
) -> list[ShpPolygon]:
    """Find enclosed rooms by planar subdivision of the wall centreline network.

    The obvious approach — subtract the wall solids from the unit boundary — does
    not work, and the reason is worth recording. That subtraction leaves rooms
    connected to the outside wherever the envelope has a gap (a balcony opening, a
    wall that stopped short of the boundary), and one gap anywhere merges every
    room in the flat into a single region. It fails silently and catastrophically.

    Planar subdivision has no such dependence on the boundary. Noding the
    centrelines against each other and taking the minimal cycles yields exactly
    the enclosed faces of the wall graph, so a gap in the outer envelope costs you
    the envelope face and nothing else.

    Two details make it work on real drawings:

    * **Ends are extended before noding.** A T-junction in CAD often has the
      stem's endpoint merely *near* the through-wall, not on it. Un-extended, the
      lines never intersect, no cycle closes, and the room silently disappears.
      Extending by half the wall thickness plus a tolerance forces the crossing.

    * **Faces are trimmed back by the wall solids.** A centreline cycle includes
      half of each surrounding wall. Subtracting the wall footprints turns the
      cycle into the true inner face of the room, which is the surface a floor
      finish is actually laid on and quoted by.
    """
    lines: list[LineString] = []
    for w in walls:
        L = w.length_mm
        if L < 1e-6:
            continue
        dx, dy = w.direction()
        # Extend both ends so near-miss junctions actually cross and node.
        ext = w.thickness_mm / 2.0 + cfg.junction_extend_mm
        lines.append(
            LineString(
                [
                    (w.start[0] - dx * ext, w.start[1] - dy * ext),
                    (w.end[0] + dx * ext, w.end[1] + dy * ext),
                ]
            )
        )
    if not lines:
        return []
    lines.extend(barriers)

    solid = _wall_solid(walls)
    faces: list[ShpPolygon] = []
    for face in polygonize(unary_union(lines)):
        if not isinstance(face, ShpPolygon) or face.area < cfg.min_room_area_mm2:
            continue
        if not clip.intersects(face.centroid):
            continue
        inner = face.difference(solid) if solid is not None else face
        if isinstance(inner, MultiPolygon):
            if inner.is_empty:
                continue
            inner = max(inner.geoms, key=lambda g: g.area)
        if not isinstance(inner, ShpPolygon) or inner.area < cfg.min_room_area_mm2:
            continue
        # Flattening to the exterior ring refills anything a wall carved out of the
        # middle of the face — a freestanding wall ends up *inside* the floor. Callers
        # that can represent or open holes ask for them kept.
        faces.append(inner if keep_interiors else ShpPolygon(inner.exterior.coords))
    return faces


def _house_footprint(
    walls: Sequence[Wall],
    unit: _UnitRegion,
    clip: ShpPolygon,
    cfg: ExtractConfig,
    barriers: Sequence[LineString] = (),
) -> ShpPolygon | None:
    """Stage 1a: the dwelling's complete interior extent, as one polygon.

    This deliberately uses only the part of the wall recovery that is reliable. The
    *envelope* — the outer walls of the flat — is the best-recovered geometry in the
    drawing: it is long, continuous, and drawn on the wall layer with both faces. The
    interior partitions are the ones that go missing.

    So the footprint is the union of every cycle in the wall graph, not the individual
    cycles. A missing interior partition merges two cycles, which changes nothing about
    their union. That makes the footprint robust to exactly the failure that used to
    destroy the room map, and it is why the space partition that follows can be
    complete: it is filling a shape we actually know.

    Cycles are *not* trimmed back by the wall solids here, unlike :func:`_room_faces`.
    The footprint must include the interior walls' own footprints, because the walls
    are about to be re-derived from the space boundaries inside it.
    """
    lines: list[LineString] = []
    for w in walls:
        if w.length_mm < 1e-6:
            continue
        dx, dy = w.direction()
        ext = w.thickness_mm / 2.0 + cfg.junction_extend_mm
        lines.append(
            LineString(
                [
                    (w.start[0] - dx * ext, w.start[1] - dy * ext),
                    (w.end[0] + dx * ext, w.end[1] + dy * ext),
                ]
            )
        )
    if not lines:
        return None
    lines.extend(barriers)

    faces = [
        f
        for f in polygonize(unary_union(lines))
        if isinstance(f, ShpPolygon) and f.area >= cfg.min_room_area_mm2 * 0.4
        and clip.intersects(f.centroid)
    ]
    if not faces:
        return None

    merged = unary_union(faces)
    parts = list(getattr(merged, "geoms", [merged]))
    # A drawing sheet holds neighbouring flats and shared circulation too. Keep the
    # connected component that holds the most of *this* unit's labels — that is this
    # dwelling — rather than the largest, which may be the corridor outside it.
    def label_count(p: ShpPolygon) -> tuple[int, float]:
        n = sum(1 for e in unit.labels if p.contains(ShpPoint(*e.points[0])))
        return (n, p.area)

    best = max((p for p in parts if isinstance(p, ShpPolygon)), key=label_count, default=None)
    if best is None or best.area < cfg.min_room_area_mm2:
        best = None

    # Fall back to the declared boundary when the walls cannot enclose the dwelling.
    #
    # The wall-cycle union is the better footprint when the envelope closes, because it
    # follows what was actually drawn. When it does not close it can be badly short: on one
    # flat it enclosed 307 sqft of a declared 710, so seven of the ten room labels fell
    # outside it and the dwelling came back with three rooms. The carpet-area polygon is a
    # legal declaration and is already trusted to segment the drawing into dwellings, so it
    # is the right thing to fall back to — buffered outward, because a carpet area excludes
    # the wall thickness that the footprint has to contain.
    declared = unit.polygon.buffer(cfg.declared_fallback_margin_mm)
    if best is None or best.area < unit.polygon.area * cfg.footprint_min_coverage:
        inside_best = label_count(best)[0] if best is not None else 0
        inside_declared = sum(1 for e in unit.labels if declared.contains(ShpPoint(*e.points[0])))
        if inside_declared > inside_best:
            return ShpPolygon(declared.exterior.coords).simplify(30.0, preserve_topology=True)
    if best is None:
        return None

    # Interior rings are ducts and shafts standing inside the flat; the outer shell is
    # the extent to be partitioned.
    shell = ShpPolygon(best.exterior.coords)

    # The declared polygon is a legal statement of the dwelling's extent, so the
    # footprint must cover it even where the drawn wall cycles never close around it.
    # A footprint that stops at the cycles silently deletes whatever they miss: one
    # 4 BHK lost its entire living-room balcony this way — 13 m² inside the declared
    # polygon with no cells for the partition to claim, while its mirror twin, whose
    # cycles happened to close, kept it. Union, not fallback: the cycles stay
    # authoritative wherever they exist, the declaration fills what they miss.
    merged_shell = unary_union([shell, unit.polygon])
    if isinstance(merged_shell, ShpPolygon):
        shell = ShpPolygon(merged_shell.exterior.coords)
    return shell.simplify(30.0, preserve_topology=True)


def _dominant_axis_deg(segments: Sequence[Any]) -> float:
    """Length-weighted modal direction of drawn line work, folded into [0, 90)."""
    buckets: dict[int, float] = {}
    for seg in segments:
        L = math.dist(seg.a, seg.b)
        if L < 900.0:
            continue
        ang = math.degrees(math.atan2(seg.b[1] - seg.a[1], seg.b[0] - seg.a[0])) % 90.0
        buckets[int(round(ang))] = buckets.get(int(round(ang)), 0.0) + L
    return float(max(buckets, key=lambda k: buckets[k])) if buckets else 0.0


def _map_house(
    unit: _UnitRegion,
    walls: Sequence[Wall],
    cfg: ExtractConfig,
    clip: ShpPolygon,
    barriers: Sequence[LineString],
    notes: list[str],
    warnings: list[str],
    raw_wall_segments: Sequence[Any] = (),
    outdoor_evidence: Any | None = None,
) -> SpaceMap | None:
    """Stage 1: map the complete house before a single wall is extruded."""
    footprint = _house_footprint(walls, unit, clip, cfg, barriers)
    if footprint is None:
        warnings.append(
            f"{unit.name}: the wall network encloses nothing, so the dwelling extent "
            "could not be established"
        )
        return None

    seeds = [
        SpaceSeed(
            label=(e.text or "").strip().splitlines()[0].strip(),
            point=e.points[0],
            dimension=parse_dimension_pair(e.text or ""),
        )
        for e in unit.labels
    ]
    # Kerb-permeable service zones. Toilets and kitchens contain drawn line work at
    # wall-band scale that is not a wall: shower kerbs, screens, platform edges — 150 mm
    # double-line bands the solver pairs exactly like a thin partition. Treated as
    # barriers they fence off part of the room's own printed rectangle: three toilets
    # measured 38-49% of their printed size with the missing floor sitting unclaimed on
    # the far side of a kerb. No toilet has an internal 150 mm wall. Inside a service
    # room's printed rectangle, kerb-scale bands (thin, short) stop being barriers; the
    # rectangle is tested in both orientations because the printed "W x D" does not say
    # which axis is which, and it is buffered because the label anchor is not centred.
    service_rects: list[ShpPolygon] = []
    for sd in seeds:
        if sd.dimension is None:
            continue
        k = sd.kind
        if not (k.is_wet or k is RoomKind.KITCHEN):
            continue
        dw, dd = sd.dimension.width_mm, sd.dimension.depth_mm
        sx, sy = sd.point
        for a, b in ((dw, dd), (dd, dw)):
            service_rects.append(
                shp_box(sx - a / 2 - 250.0, sy - b / 2 - 250.0, sx + a / 2 + 250.0, sy + b / 2 + 250.0)
            )
    service_zone = unary_union(service_rects) if service_rects else None

    def _kerb_candidate(
        a: tuple[float, float], b: tuple[float, float], thickness: float | None
    ) -> bool:
        """Could this piece of line work be a kerb rather than a wall?

        Candidacy only — thin-ish (under 200 mm; real partitions here pair at 230),
        short (under 3 m), near a service room's printed rectangle. Whether it *is*
        a kerb is decided after growth, by what it fences: a kerb fences its room
        from unclaimed cells, a real wall fences it from another room. Judging by
        shape alone cut both ways on the same drawing — it freed two toilets and
        cost two others their real walls.
        """
        if service_zone is None:
            return False
        if math.dist(a, b) > 3000.0:
            return False
        if thickness is not None and thickness > 200.0:
            return False
        return service_zone.buffer(200.0).contains(LineString([a, b]))

    # Parapets count. They were excluded on the grounds that "a parapet does not divide
    # anything", which is wrong twice over: a balcony is bounded by its guard, so without
    # parapets the wall graph never encloses one and the balcony is simply not part of the
    # dwelling footprint. On the demo flat the BALCONY label then landed on interior area
    # and the mapped balcony came out with no edge facing outdoors at all — six full-height
    # walls around it, which is not a balcony.
    segments = [(w.start, w.end) for w in walls if w.length_mm > 1e-6]
    # The **drawn** wall line work constrains growth too, not only the walls the solver
    # managed to reconstruct.
    #
    # Wall solving pairs opposing faces and discards what it cannot pair, so a partition
    # drawn with one face, or with faces the pairer rejected, never became a barrier and
    # growth walked straight through it. Measured against the source line work, that left
    # 40 m of drawn wall sitting *inside* recovered rooms on this flat: 9.2 m through one
    # bedroom, 6.7 m through a 37 sqft toilet.
    #
    # A drawn line is evidence of a division whether or not we could give it a thickness.
    # The solver still supplies thickness where it succeeded; this supplies position
    # everywhere. Short segments are skipped so hatch ticks and symbol detail do not
    # fragment the partition.
    for seg in raw_wall_segments:
        if math.dist(seg.a, seg.b) >= cfg.raw_barrier_min_mm:
            segments.append((seg.a, seg.b))
    # Wide glazed thresholds divide space as surely as a wall does, and they are the only
    # evidence available for where a balcony begins: the strip beyond a 3.6 m glazed
    # opening is outdoor space, and without that line it merged into the internal
    # circulation — the two came out as one 297 sqft region touching every room in the flat.
    #
    # Only the *wide* ones. Feeding in every door and window symbol split the partition at
    # each 700 mm doorway too, which fragmented it into 149 walls with 88 sub-300 mm stubs.
    # A doorway is a hole in a wall that already divides those spaces; it is not itself a
    # division.
    for line in barriers:
        if line.length < cfg.sliding_door_min_mm:
            continue
        cs = list(line.coords)
        segments.extend(zip(cs, cs[1:]))

    # The walls' own footprint, so the partition can exclude it. Solved walls contribute
    # their measured thickness; drawn lines the solver could not pair contribute a nominal
    # one, because a wall we can see but could not measure is still a wall.
    bodies = [
        LineString([w.start, w.end]).buffer(
            max(w.thickness_mm, 90.0) / 2.0, cap_style=2, join_style=2
        )
        for w in walls
        if w.length_mm > 1e-6
    ]
    solved_lines = (
        unary_union([LineString([w.start, w.end]) for w in walls if w.length_mm > 1e-6])
        if walls
        else None
    )
    for seg in raw_wall_segments:
        if math.dist(seg.a, seg.b) < cfg.raw_barrier_min_mm:
            continue
        line = LineString([seg.a, seg.b])
        if solved_lines is not None and solved_lines.distance(
            line.interpolate(0.5, normalized=True)
        ) <= cfg.raw_barrier_explained_mm:
            continue  # already accounted for by a solved wall
        bodies.append(line.buffer(cfg.interior_wall_mm / 2.0, cap_style=2, join_style=2))
    wall_solid = unary_union(bodies) if bodies else None

    smap = map_spaces(
        footprint,
        seeds,
        segments,
        MapConfig(
            interior_wall_mm=cfg.interior_wall_mm,
            exterior_wall_mm=cfg.exterior_wall_mm,
            min_space_area_mm2=cfg.min_room_area_mm2,
        ),
        prefix=f"{unit.name}-",
        # Wall-body exclusion, with the band skeleton behind it, is complete and correct on
        # everything except enclosure. Switch this to `wall_solid` to enable it.
        #
        # It is the only approach that drives "rooms swallow a drawn wall" to zero (15 units
        # -> 0), and the skeleton fixed what the first attempt broke: walls running through
        # rooms went 5 -> 0, overlaps 0, fragmentation 0, and area agreement recovered from
        # 74% to 86%. So the model and the skeleton are both sound.
        #
        # What is not solved: **enclosure**. Excluding bodies puts each room's ring on the
        # wall *face*, and the walls derived for the envelope and for room pairs that touch
        # directly do not reliably cover it — measured coverage stays at 52-88% even with a
        # 150 mm tolerance, so 16 dwellings report unsealed rooms and 8 report rooms with no
        # way in. The remaining work is to build the envelope wall *outboard* of the room
        # edge rather than centred on the room's own edge cells, which needs the footprint
        # boundary carried into the band derivation rather than inferred from missing cells.
        wall_solid=wall_solid,
        outdoor_evidence=outdoor_evidence,
    )

    # --- second pass: kerbs proven by what they fence -------------------------
    # A service room short of its printed size may be fenced by its own kerb or
    # shower screen — a thin drawn band the solver cannot tell from a partition.
    # Shape alone cannot decide (judging by thin-and-short freed two toilets and
    # cost two others their real walls), but after growth the distinction is
    # checkable: a kerb fences its room from *unclaimed* cells, a real wall fences
    # it from another space. Candidates beside an under-budget service room whose
    # far side is claimed by nothing are removed, and the house is mapped once
    # more without them.
    if smap is not None and smap.spaces and service_zone is not None:
        cand_lines: list[tuple[tuple[float, float], tuple[float, float], float]] = []
        for w in walls:
            if w.length_mm > 1e-6 and _kerb_candidate(w.start, w.end, w.thickness_mm):
                cand_lines.append((w.start, w.end, w.thickness_mm))
        for seg in raw_wall_segments:
            if math.dist(seg.a, seg.b) >= cfg.raw_barrier_min_mm and _kerb_candidate(
                seg.a, seg.b, None
            ):
                cand_lines.append((seg.a, seg.b, cfg.interior_wall_mm))
        service_short = [
            sp
            for sp in smap.spaces
            if (sp.kind.is_wet or sp.kind is RoomKind.KITCHEN)
            and sp.dimension is not None
            and not sp.polygon.is_empty
            and sp.polygon.area < sp.dimension.area_mm2 * 0.75
        ]
        proven: list[LineString] = []
        if service_short and cand_lines:
            claimed_all = [sp.polygon for sp in smap.spaces if not sp.polygon.is_empty]
            for a, b, thick in cand_lines:
                line = LineString([a, b])
                near = [sp for sp in service_short if sp.polygon.distance(line) < 350.0]
                if not near:
                    continue
                sp = min(near, key=lambda q: q.polygon.distance(line))
                mid = line.interpolate(0.5, normalized=True)
                dx, dy = b[0] - a[0], b[1] - a[1]
                ln = math.hypot(dx, dy) or 1.0
                nx, ny = -dy / ln, dx / ln
                off = thick / 2.0 + 150.0
                p1 = ShpPoint(mid.x + nx * off, mid.y + ny * off)
                p2 = ShpPoint(mid.x - nx * off, mid.y - ny * off)
                far = p1 if sp.polygon.distance(p1) > sp.polygon.distance(p2) else p2
                if not footprint.contains(far):
                    continue  # fences the outside: a real envelope piece
                if any(q.contains(far) for q in claimed_all):
                    continue  # fences another space: a real wall
                proven.append(line)
        if proven:
            proven_u = unary_union(proven)

            def _keep(a: tuple[float, float], b: tuple[float, float]) -> bool:
                m = LineString([a, b]).interpolate(0.5, normalized=True)
                return proven_u.distance(m) > 60.0

            segments2 = [sgp for sgp in segments if _keep(*sgp)]
            bodies2 = [
                LineString([w.start, w.end]).buffer(
                    max(w.thickness_mm, 90.0) / 2.0, cap_style=2, join_style=2
                )
                for w in walls
                if w.length_mm > 1e-6 and _keep(w.start, w.end)
            ]
            for seg in raw_wall_segments:
                if math.dist(seg.a, seg.b) < cfg.raw_barrier_min_mm or not _keep(seg.a, seg.b):
                    continue
                line = LineString([seg.a, seg.b])
                if solved_lines is not None and solved_lines.distance(
                    line.interpolate(0.5, normalized=True)
                ) <= cfg.raw_barrier_explained_mm:
                    continue
                bodies2.append(
                    line.buffer(cfg.interior_wall_mm / 2.0, cap_style=2, join_style=2)
                )
            remapped = map_spaces(
                footprint,
                seeds,
                segments2,
                MapConfig(
                    interior_wall_mm=cfg.interior_wall_mm,
                    exterior_wall_mm=cfg.exterior_wall_mm,
                    min_space_area_mm2=cfg.min_room_area_mm2,
                ),
                prefix=f"{unit.name}-",
                wall_solid=unary_union(bodies2) if bodies2 else None,
                outdoor_evidence=outdoor_evidence,
            )
            if remapped is not None and remapped.spaces:
                notes.append(
                    f"{unit.name}: {len(proven)} kerb-scale band(s) beside under-size "
                    "service rooms fenced only unclaimed floor and were released as "
                    "growth barriers"
                )
                smap = remapped

    for n in smap.notes:
        notes.append(f"{unit.name}: {n}")
    for w in smap.warnings:
        warnings.append(f"{unit.name}: {w}")
    return smap if smap.spaces else None


def _rooms_from_spaces(
    smap: SpaceMap,
    walls: Sequence[Wall],
    cfg: ExtractConfig,
    prefix: str,
    drawn_segments: Sequence[Any] = (),
    clip_footprint: ShpPolygon | None = None,
    declared: ShpPolygon | None = None,
) -> tuple[list[Room], dict[str, str]]:
    """Turn mapped spaces into rooms, insetting each only where a wall exists.

    The inset is per-boundary rather than uniform, and that is the fix for the
    complaint that "living and dining has a gap between them". A boundary the drawing
    walls off gets a wall, and both floors step back from it by half its thickness. A
    boundary the drawing leaves open gets nothing, so the two floors *meet* and the
    open plan reads as one continuous space instead of two rooms with a mystery strip
    between them.
    """
    # Cut the floors with the walls that were *actually built*, not with the boundary
    # lines they came from. Those are not the same line: a boundary matched to a
    # recovered CAD wall adopts that wall's centreline, which sits up to
    # `wall_snap_mm` away. Insetting from the boundary therefore left the floor edge
    # ~160-240 mm short of its own wall — a visible strip of nothing between every room
    # and the wall around it, and the reason rooms measured only 70-80% enclosed.
    #
    # Deriving the floor from the wall is also simply the right order: the walls are
    # built first, so the room is what they leave behind.
    solids = [
        LineString([w.start, w.end]).buffer(w.thickness_mm / 2.0, cap_style=2, join_style=2)
        for w in walls
        if w.kind not in ("parapet", "railing") and w.length_mm > 1e-6
    ]
    wall_footprint = unary_union(solids) if solids else None
    # Floors are clipped by the footprint of the walls they are cut by — not by the
    # raster-era footprint, and not by a slackened version of it.
    #
    # The original footprint was traced before walls snapped to drawn centrelines, so a
    # snapped envelope wall could sit 254 mm outside it and the floor could never reach
    # its own wall. Slackening the clip by a constant fixed that and broke the opposite
    # case: where an envelope stretch genuinely has nothing, floors drifted outward into
    # space, and sixteen dwellings reported unsealed rooms. The walls the floors are cut
    # by define exactly where floor may exist; their own cycle-footprint is the clip that
    # is right on both sides.
    clip_region = clip_footprint if clip_footprint is not None else smap.footprint
    # The drawn wall line work cuts floors too, as thin ribbons. Our built solids carry a
    # *nominal* thickness wherever the solver could not measure one, so a healed floor can
    # cross the drawn face of a wall that is really thicker — the audit calls that
    # "swallowing a drawn wall", and it is. A drawn line is the room's true boundary; a
    # floor may never cross one. Cutting with a hair-thin ribbon disconnects any growth
    # beyond the line, and the seed-piece selection below then drops it.
    # Every drawn piece down to a pier. The 900 mm floor used for growth *barriers* exists
    # to keep the raster partition from fragmenting; ribbons cannot fragment anything —
    # they only stop floor growth — and it was exactly the sub-900 pieces (piers, jambs,
    # wall returns) that healed floors crossed once the long lines were respected.
    ribbons = [
        LineString([seg.a, seg.b]).buffer(20.0)
        for seg in drawn_segments
        if math.dist(seg.a, seg.b) >= 250.0
    ]
    drawn_cut = unary_union(ribbons) if ribbons else None

    rooms: list[Room] = []
    id_map: dict[str, str] = {}
    others = {s.id: s.polygon for s in smap.spaces}
    claimed: list[ShpPolygon] = []

    # Everything a floor edge may legitimately end against: a wall body, drawn line work,
    # or the declared carpet boundary. Edges ending anywhere else are "bare".
    # Built exactly as the audit builds its fabric — wall solids and drawn lines over
    # 400 mm, both buffered 100 mm — because the heal gate is only correct if it measures
    # the same thing its output is judged by. The first version buffered at 120 and
    # included the ribbon set and the declared ring; a toilet whose wall face sat 110 mm
    # from its floor edge then read as sealed to the gate (inside 120) and leaking to the
    # audit (outside 100), so the gate saw no gap worth healing and dropped the heal that
    # would have fixed it.
    stop_parts = []
    if wall_footprint is not None:
        stop_parts.append(wall_footprint.buffer(100.0))
    audit_drawn = [
        LineString([sg.a, sg.b])
        for sg in drawn_segments
        if math.dist(sg.a, sg.b) > 400.0
    ]
    if audit_drawn:
        stop_parts.append(unary_union(audit_drawn).buffer(100.0))
    stopband = unary_union(stop_parts) if stop_parts else None

    def _bare_len(poly: ShpPolygon, neighbours: Any) -> float:
        """Perimeter ending against nothing — no wall, no drawing, no boundary, no room.

        Counts only stretches over 400 mm, mirroring the audit exactly. Counting every
        uncovered millimetre rejected genuine heals: a strip that closes a 1.6 m gap to a
        wall necessarily adds two short flank edges, and totalling those flanks made the
        heal look worse than the gap it closed even though the audit — which ignores
        sub-400 mm pieces as corner noise — would score it strictly better. A gate must
        measure what its output is judged by.
        """
        ring = poly.exterior
        cover = stopband
        if neighbours is not None and not getattr(neighbours, "is_empty", False):
            nb = neighbours.buffer(150.0)  # the audit's faces-a-room distance
            cover = nb if cover is None else cover.union(nb)
        if cover is None:
            return ring.length
        gap = ring.difference(cover)
        return sum(g.length for g in getattr(gap, "geoms", [gap]) if g.length > 400.0)

    def _open_holes(poly: ShpPolygon) -> ShpPolygon:
        """Open each interior ring to the exterior with a hair-thin slit.

        A Room is a single ring, and flattening a polygon with holes puts whatever the
        hole excluded — a wall, a duct — back *inside* the room. On the one dwelling whose
        footprint falls back to the declared polygon, that meant built walls running 1.8 m
        through a balcony floor. A 50 mm slit from the hole to the nearest exterior point
        merges the hole into the outer ring, so the excluded thing stays excluded, the
        floor loses a sliver nobody can see, and the geometry stays honest.
        """
        for _ in range(4):
            if not poly.interiors:
                return poly
            ring = max(poly.interiors, key=lambda r: ShpPolygon(r).area)
            inner = ShpPolygon(ring).representative_point()
            outer = poly.exterior.interpolate(poly.exterior.project(inner))
            slit = LineString([inner, outer]).buffer(25.0, cap_style=2)
            opened = poly.difference(slit)
            if isinstance(opened, MultiPolygon) and not opened.is_empty:
                opened = max(opened.geoms, key=lambda g: g.area)
            if not isinstance(opened, ShpPolygon) or opened.is_empty:
                return poly
            poly = opened
        return poly

    def _finish(candidate: ShpPolygon, sp: Any) -> ShpPolygon | None:
        """Hole handling, wall cut and guards — shared by the healed and plain paths."""
        poly = candidate
        # A Room polygon is a single ring, so any hole is about to be lost. Subtract the
        # spaces sitting in those holes first, then keep the largest remaining piece —
        # otherwise flattening re-absorbs the neighbour this space wraps around.
        if poly.interiors:
            enclosed = [
                q
                for i, q in others.items()
                if i != sp.id and poly.contains(q.representative_point())
            ]
            if enclosed:
                trimmed = poly.difference(unary_union(enclosed))
                if isinstance(trimmed, MultiPolygon) and not trimmed.is_empty:
                    trimmed = max(trimmed.geoms, key=lambda g: g.area)
                if isinstance(trimmed, ShpPolygon) and not trimmed.is_empty:
                    poly = trimmed
        if wall_footprint is None and drawn_cut is None:
            return _open_holes(poly)
        cut = poly
        if wall_footprint is not None:
            cut = cut.difference(wall_footprint)
        if drawn_cut is not None:
            cut = cut.difference(drawn_cut)
        pieces = [g for g in getattr(cut, "geoms", [cut]) if isinstance(g, ShpPolygon)]
        if pieces:
            # The piece that contains the space's own seed is the room; the largest piece
            # can be something across a wall.
            rep = sp.polygon.representative_point()
            mine = [g for g in pieces if g.contains(rep)] or pieces
            cut = max(mine, key=lambda g: g.area)
        if isinstance(cut, ShpPolygon) and cut.area >= max(
            cfg.min_room_area_mm2 * 0.5, sp.polygon.area * 0.45
        ):
            return _open_holes(cut)
        # Falling back to the *uncut* polygon let two rooms both run out to the shared
        # wall centreline and overlap each other. Half a partition of inset costs nothing.
        inset = poly.buffer(-cfg.interior_wall_mm / 2.0)
        if isinstance(inset, MultiPolygon) and not inset.is_empty:
            inset = max(inset.geoms, key=lambda g: g.area)
        if isinstance(inset, ShpPolygon) and inset.area >= cfg.min_room_area_mm2 * 0.5:
            return _open_holes(inset)
        return _open_holes(poly)

    for order, sp in enumerate(smap.spaces):
        # Grow the space over the raster slivers before cutting it back with the walls.
        #
        # The partition is built on a 90 mm grid, so a room's rasterised edge stops up to
        # a cell short of the wall face the drawing actually shows. Those slivers link up
        # through door openings into one connected web — measured at 169 sqft on a
        # 1558 sqft flat, the single largest reason measured area fell 20%+ short of
        # declared. Growing by ~1.5 cells reaches across the sliver; the wall solids then
        # cut the floor back to the wall face *exactly*, because they are vector geometry,
        # not raster.
        #
        # Two rules keep the healing honest, and both were bought with a failing audit:
        #
        # * **Claiming is sequential.** Both sides of an unclaimed sliver used to grow
        #   into it and overlap, because each subtracted only the other's *original*
        #   polygon. Rooms already finalised are subtracted at their grown extent;
        #   rooms not yet processed at their original extent, so the first claimant
        #   wins deterministically and overlap is impossible by construction.
        # * **A heal that creates a hole is discarded.** A balcony wrapping an interior
        #   wall closed over the wall's ends when grown, the wall became an interior
        #   ring, and the Room's single-ring flattening put the wall back *inside* the
        #   room — a 7.1 m wall through a balcony. If healing produces interior rings
        #   the space reverts to its un-grown polygon, whose behaviour the audit had
        #   already accepted.
        original = sp.polygon
        grown = original.buffer(cfg.raster_heal_mm, join_style=2)
        if clip_region is not None and not clip_region.is_empty:
            grown = grown.intersection(clip_region)
        # Growth may claim only genuinely unowned area — raster slivers and wall zones.
        # Subtracting just the *unprocessed* originals let an earlier room grow 140 mm
        # into a later room's own body wherever the two touch directly, shifting the seam
        # and leaving the later room's edge ~200 mm off its wall. Every other space's
        # original polygon is inviolable, processed or not; `claimed` additionally covers
        # slivers an earlier room has already healed over.
        blockers = claimed + [q for i, q in others.items() if i != sp.id]
        if blockers:
            grown = grown.difference(unary_union(blockers))
        if isinstance(grown, MultiPolygon) and not grown.is_empty:
            grown = max(grown.geoms, key=lambda g: g.area)

        poly = None
        if isinstance(grown, ShpPolygon) and grown.area >= original.area * 0.9:
            healed = _finish(grown, sp)
            if healed is not None and isinstance(healed, ShpPolygon) and not healed.interiors:
                poly = healed
        plain = _finish(original, sp)
        plain = plain if plain is not None else original
        # A heal is only a heal if it ends on something. Growing toward a wall face and
        # stopping there is the point; growing into an interior void and stopping at the
        # reach limit leaves the floor edge hanging in space — the kitchens gained
        # 2.7 m of bare edge exactly this way. Applying the audit's own bare-edge measure
        # here keeps healing self-consistent with what it is later judged by.
        blockers_union = unary_union(blockers) if blockers else None
        if poly is not None and _bare_len(poly, blockers_union) > _bare_len(plain, blockers_union) + 200.0:
            poly = None
        claimed.append(poly if poly is not None else plain)

    # Exact floors from the walls themselves, where the wall cycles close.
    #
    # Growing raster polygons and clipping them was tried three ways — exact old
    # footprint, slackened footprint, recomputed footprint — and the failure count
    # oscillated 8/16/12 without converging, which is the signature of a wrong primitive
    # rather than a wrong constant. The right primitive: a floor *is* the region its
    # walls enclose. Polygonising the final wall network and taking the enclosed inner
    # faces gives floors that meet every wall at its face by construction — vector-exact,
    # no slivers, no clip dilemmas. This is the original wall-first `_room_faces`, which
    # failed only because the wall set used to be incomplete; the space-first stages now
    # guarantee a wall on every boundary, so the cycles close. Each face takes the label
    # of the space that owns most of it; a space keeps its raster-healed floor only if
    # the face route cannot cover at least 60% of it (an unclosed corner of the graph).
    try:
        face_walls = [w for w in walls if w.kind not in ("parapet", "railing")]
        face_clip = (clip_region or smap.footprint).buffer(1000.0)
        faces = _room_faces(face_walls, face_clip, cfg, [], keep_interiors=True)
    except Exception:
        faces = []
    if faces:
        # Each face is *split by the space partition*, not assigned wholesale.
        #
        # Argmax assignment double-counted: where the wall graph fails to close between
        # two rooms, their shared face went entirely to one of them while the other kept
        # its healed floor — overlaps, and 102% of a declared area. Intersecting the face
        # with each space instead is exact at wall faces (the face boundary) and follows
        # the partition at open seams, and the pieces are disjoint by construction. Thin
        # unowned scraps inside a face attach to whichever piece they touch most.
        assigned: dict[int, list[ShpPolygon]] = {}
        space_union = unary_union([sp.polygon for sp in smap.spaces])
        for f in faces:
            pieces: list[tuple[int, ShpPolygon]] = []
            for i, sp in enumerate(smap.spaces):
                cutp = f.intersection(sp.polygon)
                for g in getattr(cutp, "geoms", [cutp]):
                    if isinstance(g, ShpPolygon) and g.area > 1e4:
                        pieces.append((i, g))
            scraps = f.difference(space_union)
            for g in getattr(scraps, "geoms", [scraps]):
                if not isinstance(g, ShpPolygon) or g.area <= 1e4 or not pieces:
                    continue
                ring = g.exterior
                host = max(pieces, key=lambda t: t[1].buffer(30.0).intersection(ring).length)
                pieces.append((host[0], g))
            for i, g in pieces:
                assigned.setdefault(i, []).append(g)
        for i, fs in assigned.items():
            merged = unary_union(fs)
            if isinstance(merged, MultiPolygon) and not merged.is_empty:
                merged = max(merged.geoms, key=lambda g: g.area)
            if (
                isinstance(merged, ShpPolygon)
                and merged.area >= smap.spaces[i].polygon.area * 0.6
            ):
                exact = merged
                if drawn_cut is not None:
                    trimmed = exact.difference(drawn_cut)
                    pieces = [
                        g for g in getattr(trimmed, "geoms", [trimmed]) if isinstance(g, ShpPolygon)
                    ]
                    rep = smap.spaces[i].polygon.representative_point()
                    mine = [g for g in pieces if g.contains(rep)] or pieces
                    if mine:
                        exact = max(mine, key=lambda g: g.area)
                claimed[i] = _open_holes(exact)

    # Second healing pass, against everyone's *final* floors.
    #
    # The first pass blocks growth with the other spaces' original raster polygons, which
    # is right while their finals are unknown — but a snapped wall can move toward a
    # neighbour, and the strip between the old seam and the new wall face is then orphaned:
    # the neighbour's floor was cut back beyond it, and this room's growth was blocked by a
    # polygon the neighbour no longer occupies. With every final known, the strip is
    # claimable by the only room that can reach it without crossing a wall.
    for i, sp in enumerate(smap.spaces):
        base = claimed[i]
        trace = _TRACE_HEAL and _TRACE_HEAL in (sp.label or sp.kind.value).upper()
        raw = base.buffer(cfg.raster_heal_mm, join_style=2)
        grown = raw
        if clip_region is not None and not clip_region.is_empty:
            grown = grown.intersection(clip_region)
        after_clip = grown.area if hasattr(grown, "area") else 0.0
        blockers = [claimed[j] for j in range(len(claimed)) if j != i]
        if blockers:
            grown = grown.difference(unary_union(blockers))
        if isinstance(grown, MultiPolygon) and not grown.is_empty:
            grown = max(grown.geoms, key=lambda g: g.area)
        if not isinstance(grown, ShpPolygon) or grown.is_empty:
            if trace:
                print(f"[heal2 {prefix} {sp.label}] REJECT: grown empty after blockers")
            continue
        healed = _finish(grown, sp)
        others_union = unary_union(blockers) if blockers else None
        if trace:
            g0 = raw.area
            print(
                f"[heal2 {prefix} {sp.label}] base={base.area/1e6:.2f} raw={g0/1e6:.2f} "
                f"clip={after_clip/1e6:.2f} blk={grown.area/1e6:.2f} "
                f"finish={'None' if healed is None else f'{healed.area/1e6:.2f}'} "
                f"holes={len(healed.interiors) if isinstance(healed, ShpPolygon) else '-'} "
                f"area_ok={isinstance(healed, ShpPolygon) and healed.area >= base.area - 1e-6} "
                f"rep_ok={isinstance(healed, ShpPolygon) and healed.contains(base.representative_point())} "
                f"bare_base={_bare_len(base, others_union):.0f} "
                f"bare_heal={_bare_len(healed, others_union):.0f}"
                if isinstance(healed, ShpPolygon)
                else f"[heal2 {prefix} {sp.label}] finish returned {type(healed).__name__}"
            )
        if (
            healed is None
            or not isinstance(healed, ShpPolygon)
            or healed.interiors
            or not healed.contains(base.representative_point())
        ):
            continue
        # Keep growth piece by piece, not all-or-nothing.
        #
        # A single heal can do both things at once: close a real gap to a wall on one
        # flank and spill into an interior void on another. Judging the whole polygon
        # rejected exactly that case — the u4 toilet's heal reached its wall (+0.55 m²)
        # and was thrown away because the spill elsewhere raised total bare edge. Each
        # piece of gained floor is judged on its own outer edge: if most of it ends on a
        # wall, drawn line work, the carpet boundary or a neighbouring floor, the piece
        # is a genuine heal; if it dangles in space, it is invention and only that piece
        # is dropped.
        gain = healed.difference(base)
        # Judged by the audit's own measure, one piece at a time: a piece of gained
        # floor is merged only if the merged polygon's bare-edge length does not grow.
        # An edge-coverage ratio was tried first and let elongated intrusions through —
        # a piece can ride along walls for half its edge while its far end dangles in a
        # void. Monotone-in-the-metric is the only rule that cannot be gamed by shape.
        current = base
        cur_bare = _bare_len(current, others_union)
        kept = 0
        pieces = [
            g
            for g in getattr(gain, "geoms", [gain])
            if isinstance(g, ShpPolygon) and g.area >= 1e4
        ]
        for gpiece in sorted(pieces, key=lambda g: -g.area):
            trial = unary_union([current, gpiece])
            if isinstance(trial, MultiPolygon) and not trial.is_empty:
                mine = [g for g in trial.geoms if g.contains(base.representative_point())]
                trial = mine[0] if mine else max(trial.geoms, key=lambda g: g.area)
            if not isinstance(trial, ShpPolygon) or trial.is_empty:
                continue
            trial = _open_holes(trial)
            if trial.interiors or trial.area < current.area - 1e-6:
                continue
            tb = _bare_len(trial, others_union)
            if tb <= cur_bare + 100.0:
                current, cur_bare, kept = trial, tb, kept + 1
            elif trace:
                print(
                    f"[heal2 {prefix} {sp.label}] drop piece {gpiece.area/1e6:.2f} m² "
                    f"(bare {cur_bare:.0f} -> {tb:.0f})"
                )
        if kept and current.area >= base.area - 1e-6:
            if trace:
                print(
                    f"[heal2 {prefix} {sp.label}] ACCEPT {kept} piece(s), "
                    f"{base.area/1e6:.2f} -> {current.area/1e6:.2f} m²"
                )
            claimed[i] = current

    # No floor beyond the declared dwelling boundary.
    #
    # The RERA carpet polygon is the legal statement of where this dwelling ends, and it
    # is the one input measured exact on every unit (±0.2%). Floors were overshooting it
    # through the entry: the lobby grew out of the doorway into the shared corridor and
    # ended 350 mm past the boundary with 6.8 m of edge facing nothing — because beyond
    # the boundary there *is* nothing of ours. The small buffer admits the shared wall's
    # own zone; everything past it belongs to the corridor or the neighbour.
    if declared is not None and not declared.is_empty:
        # 60 mm is geometry noise, not an allowance. The carpet polygon traces the walls'
        # inner faces, so floor legitimately reaches exactly to it and never past it — the
        # earlier 250 mm "wall zone allowance" let floors overshoot their own envelope
        # walls and hang at the buffer edge, which was most of the remaining leak class.
        limit = declared.buffer(60.0)
        for i, poly in enumerate(claimed):
            if limit.contains(poly):
                continue
            cut = poly.intersection(limit)
            if isinstance(cut, MultiPolygon) and not cut.is_empty:
                rep = smap.spaces[i].polygon.representative_point()
                mine = [g for g in cut.geoms if isinstance(g, ShpPolygon) and g.contains(rep)]
                cut = mine[0] if mine else max(cut.geoms, key=lambda g: g.area)
            if isinstance(cut, ShpPolygon) and cut.area >= cfg.min_room_area_mm2 * 0.5:
                claimed[i] = _open_holes(cut)

        # --- unowned strips resolved ------------------------------------------------
        # Everything inside the declared polygon must end as floor or wall. What the
        # partition, the heals and the clip leave behind is interstitial strips: a
        # 150 mm raster channel through a bedroom's own floor, the flank a toilet lost
        # to a drawn service band, seams between rooms where no wall was built. Left
        # unowned they read as enclosure leaks, hold rooms short of their printed
        # sizes, and understate the measured area. Each strip is set apart from the
        # drawing's own line work first — a drawn band through a strip is wall, never
        # floor — and what remains merges into the room it shares the longest open
        # frontier with, under the same bare-edge gate every heal answers to.
        drawn_lines_u = (
            unary_union(
                [
                    LineString([sg.a, sg.b])
                    for sg in drawn_segments
                    if math.dist(sg.a, sg.b) > 250.0
                ]
            )
            if drawn_segments
            else None
        )
        floors_u = unary_union([q for q in claimed if not q.is_empty])
        leftover = declared.difference(floors_u)
        if wall_footprint is not None:
            leftover = leftover.difference(wall_footprint)
        if drawn_lines_u is not None:
            # 100 mm, exactly the audit fabric's reach around drawn line work. A 120 mm
            # margin left healed floor edges 20 mm outside the fabric, which reads as a
            # leak — the same gate-vs-audit mismatch class as the heal-gate saga.
            leftover = leftover.difference(drawn_lines_u.buffer(100.0))
        pieces = [
            g
            for g in getattr(leftover, "geoms", [leftover])
            if isinstance(g, ShpPolygon) and g.area >= 8e4
        ]
        for piece in sorted(pieces, key=lambda g: -g.area):
            frontier: list[tuple[float, int]] = []
            for i, poly in enumerate(claimed):
                if poly.is_empty or poly.distance(piece) > 100.0:
                    continue
                shared = poly.buffer(70.0).intersection(piece).area
                if shared > 0.0:
                    frontier.append((shared, i))
            if not frontier:
                continue
            frontier.sort(reverse=True)
            _, best_i = frontier[0]
            base = claimed[best_i]
            # Bridge the raster seam, but never into wall body, across drawn work, or
            # into a neighbour's floor — the bridge dilation reaches 70 mm past the
            # strip, which overlapped the room on the strip's far side on two units.
            add = piece.buffer(70.0, join_style=2)
            if wall_footprint is not None:
                add = add.difference(wall_footprint)
            others_floor = unary_union(
                [q for j, q in enumerate(claimed) if j != best_i and not q.is_empty]
            )
            if not others_floor.is_empty:
                add = add.difference(others_floor.buffer(10.0))
            trial = unary_union([base, add])
            if isinstance(trial, MultiPolygon) and not trial.is_empty:
                mine = [g for g in trial.geoms if g.contains(base.representative_point())]
                trial = mine[0] if mine else max(trial.geoms, key=lambda g: g.area)
            if not isinstance(trial, ShpPolygon) or trial.is_empty:
                continue
            trial = _open_holes(trial.intersection(limit))
            if not isinstance(trial, ShpPolygon) or trial.is_empty or trial.interiors:
                continue
            if trial.area < base.area + 1e5:
                continue
            # The audit's own two measures, monotone: no new bare edge, no drawn wall
            # newly swallowed.
            others_union = unary_union(
                [q for j, q in enumerate(claimed) if j != best_i and not q.is_empty]
            )
            if _bare_len(trial, others_union) > _bare_len(base, others_union) + 100.0:
                continue
            if drawn_lines_u is not None:
                swallowed_before = drawn_lines_u.intersection(base.buffer(-120.0)).length
                swallowed_after = drawn_lines_u.intersection(trial.buffer(-120.0)).length
                if swallowed_after > swallowed_before + 50.0:
                    continue
            claimed[best_i] = trial

        # --- wet rooms fenced by their own kerbs --------------------------------
        # Three toilets measured 38-48% of their printed size with the missing floor
        # sitting unclaimed on the far side of a drawn 150 mm band *inside the room's
        # own printed rectangle* — a shower kerb or screen, drawn on the wall layer
        # and indistinguishable from a thin wall to the solver. No toilet has an
        # internal 150 mm wall; the printed dimension is the architect saying the
        # room continues across it. Merge is budget-bounded and wet-only: the room
        # must be short by 30%+, the fence must be kerb-scale (every crossed segment
        # under 2.2 m), and the result may not exceed the printed size by more than
        # 15%.
        floors_now = unary_union([q for q in claimed if not q.is_empty])
        leftover2 = declared.difference(floors_now)
        if wall_footprint is not None:
            leftover2 = leftover2.difference(wall_footprint)
        wet_pieces = [
            g
            for g in getattr(leftover2, "geoms", [leftover2])
            if isinstance(g, ShpPolygon) and g.area >= 2e5
        ]
        for i, sp in enumerate(smap.spaces):
            if not sp.kind.is_wet or sp.dimension is None:
                continue
            base = claimed[i]
            want = sp.dimension.area_mm2
            if base.is_empty or base.area >= want * 0.7:
                continue
            reach = max(sp.dimension.width_mm, sp.dimension.depth_mm) + 300.0
            seed_pt = ShpPoint(*sp.seed)
            for piece in sorted(wet_pieces, key=lambda g: -g.area):
                if piece.distance(base) > 700.0 or piece.distance(seed_pt) > reach:
                    continue
                if base.area + piece.area > want * 1.15:
                    continue
                bridge = piece.buffer(250.0, join_style=2).intersection(
                    base.buffer(250.0, join_style=2)
                )
                addition = unary_union([piece, bridge])
                others_now = unary_union(
                    [q for j, q in enumerate(claimed) if j != i and not q.is_empty]
                )
                if not others_now.is_empty:
                    addition = addition.difference(others_now.buffer(10.0))
                trial = unary_union([base, addition])
                if wall_footprint is not None:
                    trial = trial.difference(wall_footprint.buffer(-10.0))
                if isinstance(trial, MultiPolygon) and not trial.is_empty:
                    mine = [
                        g for g in trial.geoms if g.contains(base.representative_point())
                    ]
                    trial = mine[0] if mine else None
                if not isinstance(trial, ShpPolygon) or trial.is_empty:
                    continue
                trial = _open_holes(trial.intersection(limit))
                if not isinstance(trial, ShpPolygon) or trial.is_empty or trial.interiors:
                    continue
                # Every drawn segment newly inside must be kerb-scale.
                gained = trial.difference(base).buffer(30.0)
                crossed = [
                    sg
                    for sg in drawn_segments
                    if math.dist(sg.a, sg.b) > 250.0
                    and gained.contains(
                        LineString([sg.a, sg.b]).interpolate(0.5, normalized=True)
                    )
                ]
                if any(math.dist(sg.a, sg.b) > 2200.0 for sg in crossed):
                    continue
                claimed[i] = trial
                base = trial
                if base.area >= want * 0.7:
                    break

        # --- open-plan splits follow the printed budgets ------------------------
        # Living and dining share one open space; nothing drawn defines where one
        # ends and the other begins, so growth splits them by raster reach — which
        # left one living room at 59% of its printed size with the dining holding
        # the difference. The printed dimensions are the architect's own statement
        # of the split, so the shared union is re-divided by a straight cut placed
        # where the budgets say, provided both label anchors stay in their own half
        # and the cut does not cross drawn work.
        open_kinds = {RoomKind.LIVING, RoomKind.DINING}
        for i, si in enumerate(smap.spaces):
            for j, sj in enumerate(smap.spaces):
                if j <= i:
                    continue
                if si.kind not in open_kinds or sj.kind not in open_kinds:
                    continue
                if si.dimension is None or sj.dimension is None:
                    continue
                a, b = claimed[i], claimed[j]
                if a.is_empty or b.is_empty or a.distance(b) > 30.0:
                    continue
                want_a, want_b = si.dimension.area_mm2, sj.dimension.area_mm2
                have_a, have_b = a.area, b.area
                # Only intervene when the pair is genuinely lopsided and the union
                # roughly holds both budgets — otherwise the drawing and the labels
                # disagree about more than the split.
                if min(have_a / want_a, have_b / want_b) > 0.72:
                    continue
                total = have_a + have_b
                if not 0.75 <= total / (want_a + want_b) <= 1.3:
                    continue
                both = unary_union([a, b])
                if not isinstance(both, ShpPolygon):
                    continue
                pa, pb = ShpPoint(*si.seed), ShpPoint(*sj.seed)
                if not (both.contains(pa) and both.contains(pb)):
                    continue
                dx0, dy0 = abs(pa.x - pb.x), abs(pa.y - pb.y)
                axis = "x" if dx0 >= dy0 else "y"
                lo_pt, hi_pt = (pa, pb) if (pa.x if axis == "x" else pa.y) <= (
                    pb.x if axis == "x" else pb.y
                ) else (pb, pa)
                lo_want = want_a if lo_pt is pa else want_b
                minx, miny, maxx, maxy = both.bounds
                best_cut: tuple[float, float] | None = None  # (score, position)
                lo_bound = (lo_pt.x if axis == "x" else lo_pt.y) + 200.0
                hi_bound = (hi_pt.x if axis == "x" else hi_pt.y) - 200.0
                pos = lo_bound
                while pos <= hi_bound:
                    if axis == "x":
                        lo_half = both.intersection(shp_box(minx - 10, miny - 10, pos, maxy + 10))
                    else:
                        lo_half = both.intersection(shp_box(minx - 10, miny - 10, maxx + 10, pos))
                    score = abs(lo_half.area - lo_want)
                    if best_cut is None or score < best_cut[0]:
                        best_cut = (score, pos)
                    pos += 90.0
                if best_cut is None:
                    continue
                pos = best_cut[1]
                if axis == "x":
                    lo_half = both.intersection(shp_box(minx - 10, miny - 10, pos, maxy + 10))
                    hi_half = both.intersection(shp_box(pos, miny - 10, maxx + 10, maxy + 10))
                else:
                    lo_half = both.intersection(shp_box(minx - 10, miny - 10, maxx + 10, pos))
                    hi_half = both.intersection(shp_box(minx - 10, pos, maxx + 10, maxy + 10))

                def _main(g: Any, seed_pt: Any) -> ShpPolygon | None:
                    parts = [
                        q
                        for q in getattr(g, "geoms", [g])
                        if isinstance(q, ShpPolygon) and not q.is_empty
                    ]
                    holder = [q for q in parts if q.contains(seed_pt)]
                    return holder[0] if holder else None

                new_lo = _main(lo_half, lo_pt)
                new_hi = _main(hi_half, hi_pt)
                if new_lo is None or new_hi is None:
                    continue
                new_a = new_lo if lo_pt is pa else new_hi
                new_b = new_hi if lo_pt is pa else new_lo
                before = min(have_a / want_a, want_a / have_a) + min(have_b / want_b, want_b / have_b)
                after = min(new_a.area / want_a, want_a / new_a.area) + min(
                    new_b.area / want_b, want_b / new_b.area
                )
                if after <= before + 0.05:
                    continue
                claimed[i] = _open_holes(new_a)
                claimed[j] = _open_holes(new_b)

    for sp, poly in zip(smap.spaces, claimed):
        # The strip and kerb merges can leave a ring that touches itself at a point,
        # which shapely calls invalid and the furnish solver crashes on. Normalise
        # here, once, where every floor leaves the factory.
        if not poly.is_empty and not poly.is_valid:
            fixed = poly.buffer(0)
            if isinstance(fixed, MultiPolygon) and not fixed.is_empty:
                fixed = max(fixed.geoms, key=lambda g: g.area)
            if isinstance(fixed, ShpPolygon) and not fixed.is_empty:
                poly = fixed
        if sp.kind.is_outdoor:
            floor_level, ceil = -50.0, cfg.parapet_height_mm
        elif sp.kind.is_wet:
            floor_level, ceil = -20.0, cfg.wall_height_mm - 300.0
        else:
            floor_level, ceil = 0.0, cfg.wall_height_mm

        rid = f"{prefix}-r{len(rooms) + 1}"
        id_map[sp.id] = rid
        agree = sp.agreement()
        rooms.append(
            Room(
                id=rid,
                kind=sp.kind,
                label=sp.label,
                polygon=[(float(x), float(y)) for x, y in poly.exterior.coords[:-1]],
                centroid=(float(poly.centroid.x), float(poly.centroid.y)),
                stated_dimension=sp.dimension,
                ceiling_height_mm=ceil,
                floor_level_mm=floor_level,
                source=Source.CAD_VECTOR,
                confidence=0.9 if (agree is None or agree > 0.7) else 0.7,
            )
        )
    return rooms, id_map


def _merge_collinear(
    walls: Sequence[Wall], prefix: str
) -> tuple[list[Wall], dict[str, Wall]]:
    """Fuse collinear, touching walls of the same thickness into single walls.

    The boundary extraction works one grid run at a time, so a straight 6 m partition
    arrives as a dozen abutting pieces — 151 walls for a four-bedroom flat. That is wrong
    on three counts. It is not what a wall *is*: a wall with a door in it is one wall, not
    three. It litters the render with hairline seams at every joint. And most seriously,
    subtracting 151 separate solids from a room shattered the floor into fragments, so the
    guard against over-cutting kicked in and kept the *uncut* polygon instead — which is
    why rooms still ran out to the wall centrelines and touched their neighbours after the
    walls were correct.

    Returns the merged walls, plus a map from every original id to the wall that now
    represents it, so openings and boundary links can be re-pointed.
    """
    groups: dict[tuple[str, int, int, int, bool], list[tuple[float, float, Wall]]] = {}
    for w in walls:
        if w.length_mm < 1e-6:
            continue
        dx, dy = w.end[0] - w.start[0], w.end[1] - w.start[1]
        if abs(dx) < abs(dy):  # vertical run
            axis, offset = "V", (w.start[0] + w.end[0]) / 2.0
            lo, hi = sorted((w.start[1], w.end[1]))
        else:
            axis, offset = "H", (w.start[1] + w.end[1]) / 2.0
            lo, hi = sorted((w.start[0], w.end[0]))
        # Height and kind are part of the identity of a wall, not incidental. Merging on
        # position and thickness alone fused a 1.05 m balcony parapet into the
        # full-height partition it happened to be collinear with, and since the merged
        # height was the maximum of the members the parapet came out 2.9 m tall —
        # enclosing the balcony behind a solid wall.
        key = (
            axis,
            int(round(offset / 25.0)),
            int(round(w.thickness_mm / 10.0)),
            int(round(w.height_mm / 50.0)),
            w.kind in ("parapet", "railing"),
        )
        groups.setdefault(key, []).append((lo, hi, w))

    out: list[Wall] = []
    remap: dict[str, Wall] = {}
    for (axis, off_b, _th_b, _h_b, _parapet), items in groups.items():
        items.sort(key=lambda t: t[0])
        runs: list[tuple[float, float, list[Wall]]] = []
        for lo, hi, w in items:
            if runs and lo <= runs[-1][1] + 40.0:
                prev_lo, prev_hi, members = runs[-1]
                runs[-1] = (prev_lo, max(prev_hi, hi), members + [w])
            else:
                runs.append((lo, hi, [w]))
        for lo, hi, members in runs:
            # The longest member is the most trustworthy: prefer its provenance and its
            # exact offset over a short stub's.
            lead = max(members, key=lambda w: w.length_mm)
            offset = off_b * 25.0
            if axis == "V":
                start, end = (offset, lo), (offset, hi)
            else:
                start, end = (lo, offset), (hi, offset)
            exterior = any(m.exterior for m in members)
            merged = Wall(
                id=f"{prefix}-w{len(out) + 1}",
                start=(float(start[0]), float(start[1])),
                end=(float(end[0]), float(end[1])),
                thickness_mm=lead.thickness_mm,
                height_mm=lead.height_mm,
                # Kind follows the exterior flag rather than the lead's, so the two can
                # never disagree. They did: walls came out `kind="structural"` with
                # `exterior=False`, which reads as an external wall to the material
                # assignment and an internal one to everything else.
                kind=(
                    lead.kind
                    if lead.kind in ("parapet", "railing")
                    else ("structural" if exterior else "partition")
                ),
                source=lead.source,
                confidence=min(m.confidence for m in members),
                exterior=exterior,
            )
            out.append(merged)
            for m in members:
                remap[m.id] = merged

    # Drop stubs that other walls already cover.
    #
    # Squaring up a stepped boundary leaves a sub-cell nub at each step, and where two
    # long walls meet at a T or an L their own thickness already fills that nub
    # completely. Emitting it anyway put a little block on top of the junction — the dark
    # cubes dotted along the wall heads in the render. A stub is only removed when it is
    # genuinely redundant, so no room can be opened by this.
    solids = {
        w.id: LineString([w.start, w.end]).buffer(w.thickness_mm / 2.0, cap_style=2, join_style=2)
        for w in out
    }
    # One union of the walls that are definitely staying, tested against each stub.
    # Rebuilding the union per stub was O(n^2) in GEOS operations and took the extraction
    # from seconds to minutes across sixteen dwellings.
    long_walls = [w for w in out if w.length_mm >= 300.0]
    cover = unary_union([solids[w.id] for w in long_walls]) if long_walls else None
    keep: list[Wall] = list(long_walls)
    dropped_ids: list[str] = []
    for w in out:
        if w.length_mm >= 300.0:
            continue
        if cover is not None and solids[w.id].intersection(cover).area >= solids[w.id].area * 0.8:
            dropped_ids.append(w.id)
        else:
            keep.append(w)
    if dropped_ids:
        survivor = {w.id for w in keep}
        fallback = max(keep, key=lambda w: w.length_mm) if keep else None
        for old, w in list(remap.items()):
            if w.id not in survivor:
                # Re-point anything anchored to a dropped stub at the wall that covers it.
                near = [k for k in keep if solids[k.id].intersects(solids[w.id])]
                remap[old] = max(near, key=lambda k: k.length_mm) if near else fallback
    return keep, remap


def _trim_wall(wall: Wall, run: LineString, new_id: str) -> Wall:
    """A copy of ``wall`` shortened to the stretch of it that ``run`` covers.

    The boundary's endpoints are projected onto the wall's own line, so the result keeps
    the drawing's exact direction and thickness and only loses the length that belongs to
    a different room.
    """
    line = LineString([wall.start, wall.end])
    ts = sorted(line.project(ShpPoint(*c)) for c in (run.coords[0], run.coords[-1]))
    lo = max(0.0, ts[0])
    hi = min(line.length, ts[1])
    if hi - lo < 200.0:
        # The projection collapsed — a short jog in the boundary landed on a long wall.
        # Returning the wall as drawn was the bug behind a 15 m wall crossing the whole
        # flat: a 150 mm jog matched it and the untrimmed original went into the model.
        # Give the jog a wall of its own length instead, centred where it projects.
        want = max(run.length, 400.0)
        mid = (lo + hi) / 2.0
        lo = max(0.0, mid - want / 2.0)
        hi = min(line.length, lo + want)
        if hi - lo < 100.0:
            return wall
    a, b = line.interpolate(lo), line.interpolate(hi)
    return Wall(
        id=new_id,
        start=(float(a.x), float(a.y)),
        end=(float(b.x), float(b.y)),
        thickness_mm=wall.thickness_mm,
        height_mm=wall.height_mm,
        kind=wall.kind,
        source=wall.source,
        confidence=wall.confidence,
        exterior=wall.exterior,
    )


_GATE_OPENABLE = frozenset(
    {
        RoomKind.LIVING,
        RoomKind.DINING,
        RoomKind.LOBBY,
        RoomKind.CORRIDOR,
        RoomKind.ENTRANCE,
    }
)
"""Room kinds that may plausibly connect without a wall. The drawn-evidence gate only
opens boundaries where both sides are in this set; everywhere else a walled boundary
keeps its wall even when the matcher found no drawn line, because bedrooms, kitchens and
wet rooms are enclosed by definition and missing evidence means our matching failed."""


def _walls_from_boundaries(
    smap: SpaceMap,
    recovered: Sequence[Wall],
    cfg: ExtractConfig,
    prefix: str,
    notes: list[str],
    drawn_segments: Sequence[Any] | None = None,
) -> tuple[list[Wall], list[tuple[Boundary, Wall | None]]]:
    """Stage 2: extrude walls, one per space boundary.

    Every wall now exists *because* two mapped spaces are divided there, or because a
    space faces outside. That is the property the old pipeline lacked: it emitted the
    walls it happened to recover, so a partition the face-pairing missed left two rooms
    sharing no wall at all, and one it hallucinated left a wall through the middle of a
    room.

    Recovered CAD walls are not discarded — they are *matched* to boundaries. A boundary
    that lies along a recovered wall inherits that wall's real thickness and endpoints,
    which are more accurate than anything the raster can produce. A boundary with no
    recovered wall gets a synthesised one at the fallback thickness, closing the holes.
    Recovered walls matching no boundary were solver noise or belong to the neighbouring
    flat, and are dropped.
    """
    if not smap.boundaries:
        return list(recovered), []

    outdoor = {sp.id for sp in smap.spaces if sp.kind.is_outdoor}
    kind_of = {sp.id: sp.kind for sp in smap.spaces}

    unclaimed = list(recovered)
    used: set[int] = set()
    emitted: set[int] = set()
    walls: list[Wall] = []
    links: list[tuple[Boundary, Wall | None]] = []
    matched = synthesised = 0

    for b in smap.boundaries:
        if not (b.walled or b.space_b is None):
            links.append((b, None))  # an open threshold: deliberately no wall
            continue

        # Find the recovered wall that best explains this boundary: near it, and running
        # the same way. Angle matters — a perpendicular wall crossing the boundary's
        # midpoint is close by but is emphatically not this wall.
        mid = b.line.interpolate(0.5, normalized=True)
        bx, by = b.line.coords[-1][0] - b.line.coords[0][0], b.line.coords[-1][1] - b.line.coords[0][1]
        blen = math.hypot(bx, by) or 1.0
        best: Wall | None = None
        best_score = 0.0
        for w in unclaimed:
            if w.kind in ("parapet", "railing") or w.length_mm < 1e-6:
                continue
            # A long wall may bound several rooms in series; each of their boundaries
            # gets its own trimmed piece of it.
            seg = LineString([w.start, w.end])
            d = seg.distance(mid)
            if d > MapConfig().wall_snap_mm:
                continue
            dx, dy = w.direction()
            align = abs((bx / blen) * dx + (by / blen) * dy)
            if align < 0.85:
                continue
            score = align * (1.0 - d / MapConfig().wall_snap_mm)
            if score > best_score:
                best, best_score = w, score

        # The wall is the wall the drawing shows — matched, not re-derived.
        #
        # Once wall *bodies* are excluded from the partition, deriving a wall from each
        # space boundary is redundant and lossy: the boundary is a fitted line through a
        # band of wall cells, so an L-shaped wall came back as one straight line and the
        # rest of it vanished. Wall counts fell from ~100 to ~30 per flat and rooms went
        # unsealed.
        #
        # We already hold the real geometry — the solver's centreline and thickness where
        # it succeeded. Boundaries keep the job they are actually good at: saying which two
        # rooms adjoin, which is what door placement needs.
        # Build a wall on **every** boundary, taking the thickness from the drawn wall
        # where one matches.
        #
        # These two properties were being treated as alternatives, and they are not:
        # building on boundaries is what guarantees a room is *enclosed* (every edge gets a
        # wall), while excluding wall bodies from the partition is what guarantees a room
        # never *contains* one. Emitting only the drawn walls gave fidelity and left rooms
        # open on the sides the drawing never drew; building only from boundaries gave
        # enclosure and let rooms swallow walls. Together they give both.
        # A synthesised *interior* wall requires drawn evidence along the boundary.
        #
        # "No recovered wall" used to default to building one anyway, and that invented
        # architecture: a wall appeared between the lobby and the living room with only
        # 260 mm of drawn line work along its 1.44 m — the drawing plainly shows the two
        # open to each other. If the drawing draws nothing along an interior boundary,
        # the honest reading is an open threshold, exactly as it already is for living
        # into dining. The envelope is different: a dwelling must close, so its
        # boundaries keep their wall or guard regardless.
        # The evidence gate applies only where an open connection is architecturally
        # plausible: social and circulation spaces. It originally applied to every
        # interior boundary, and that un-walled toilet-bedroom seams whose drawn wall the
        # matcher simply missed — the toilets then read as openly reachable, so no door
        # was placed, and the audit rightly called them unreachable. A wet room is never
        # open to a bedroom; absence of matched evidence there means our matching failed,
        # not that the architect omitted the wall.
        a_kind = kind_of.get(b.space_a)
        b_kind = kind_of.get(b.space_b) if b.space_b else None
        gate_ok = (
            a_kind in _GATE_OPENABLE
            and b_kind in _GATE_OPENABLE
        )
        if gate_ok and best is None and not b.exterior and drawn_segments is not None:
            # Support is measured in the same corridor the wall-matcher searches
            # (`wall_snap_mm`), counting only line work roughly parallel to the boundary.
            # A 100 mm corridor missed genuine walls — boundaries are raster fits sitting
            # up to 250 mm off the drawn line — and dropping those walls unsealed half
            # the drawing. Perpendicular strokes (jambs, returns) do not count as
            # evidence that a wall runs *along* the boundary.
            bx = b.line.coords[-1][0] - b.line.coords[0][0]
            by = b.line.coords[-1][1] - b.line.coords[0][1]
            blen = math.hypot(bx, by) or 1.0
            support = 0.0
            for sg in drawn_segments:
                sl = math.dist(sg.a, sg.b)
                if sl < 200.0:
                    continue
                dx, dy = (sg.b[0] - sg.a[0]) / sl, (sg.b[1] - sg.a[1]) / sl
                if abs(dx * bx / blen + dy * by / blen) < 0.94:
                    continue  # not parallel: a jamb or a return, not this wall
                mid = ((sg.a[0] + sg.b[0]) / 2.0, (sg.a[1] + sg.b[1]) / 2.0)
                if b.line.distance(ShpPoint(*mid)) > MapConfig().wall_snap_mm:
                    continue
                support += sl
            if support < b.length * 0.4:
                b.walled = False
                links.append((b, None))
                continue

        # Position comes from the drawn wall when one matches; extent from the boundary.
        #
        # An earlier rule put every wall on the boundary line and took only *thickness*
        # from the match, on the argument that a boundary sits within a raster cell of the
        # real wall. Measured, that argument failed: a band-fitted boundary sat 274 mm
        # from a wall the drawing shows 18 mm away, and the room edge could then never
        # reach its own wall. The original reason for boundary-positioning is also gone —
        # floors are now grown and cut against the *built* walls, so wherever the wall
        # goes, the floor follows exactly. The drawn centreline is vector-exact input
        # geometry; the boundary is a raster fit; when both exist, the drawing wins.
        # A balcony's outdoor-facing edge gets a glass guard rather than a wall — thin,
        # transparent and waist-high, so the balcony stays part of the elevation.
        guard = b.exterior and (
            b.space_a in outdoor or (b.space_b is not None and b.space_b in outdoor)
        )

        def emit(start: tuple[float, float], end: tuple[float, float], from_drawing: bool) -> Wall:
            walls.append(
                Wall(
                    id=f"{prefix}-w{len(walls) + 1}",
                    start=(float(start[0]), float(start[1])),
                    end=(float(end[0]), float(end[1])),
                    thickness_mm=(best.thickness_mm if best is not None else b.thickness),
                    height_mm=cfg.railing_height_mm if guard else cfg.wall_height_mm,
                    kind="railing" if guard else ("structural" if b.exterior else "partition"),
                    source=Source.CAD_VECTOR if from_drawing else Source.INFERRED,
                    confidence=0.9 if from_drawing else 0.6,
                    exterior=b.exterior,
                )
            )
            return walls[-1]

        if best is not None:
            used.add(id(best))
            matched += 1
            # A partial match must not silently unwall the rest of the boundary.
            #
            # The matcher accepts a drawn wall on midpoint distance alone, so a 1.6 m wall
            # can match a 3.0 m boundary. The trim then correctly keeps only the drawn
            # extent — but the remainder of the boundary got *nothing*, which was the
            # entire remaining failure class: party-wall stretches at unit boundaries left
            # bare for 1.4-2.3 m because the drawn wall covered the boundary only in part.
            # The drawn piece keeps its vector-exact position; the uncovered spans get
            # synthesised walls on the boundary line, exactly as a wholly unmatched
            # boundary would.
            seg = _trim_wall(best, b.line, "tmp")
            primary = emit(seg.start, seg.end, from_drawing=True)
            links.append((b, primary))
            t0 = b.line.project(ShpPoint(*seg.start))
            t1 = b.line.project(ShpPoint(*seg.end))
            lo, hi = sorted((t0, t1))
            for a_t, b_t in ((0.0, lo), (hi, b.line.length)):
                if b_t - a_t < 300.0:
                    continue
                pa = b.line.interpolate(a_t)
                pb = b.line.interpolate(b_t)
                synthesised += 1
                emit((pa.x, pa.y), (pb.x, pb.y), from_drawing=False)
        else:
            synthesised += 1
            links.append(
                (
                    b,
                    emit(
                        (b.line.coords[0][0], b.line.coords[0][1]),
                        (b.line.coords[-1][0], b.line.coords[-1][1]),
                        from_drawing=False,
                    ),
                )
            )

    # Parapets are not partitions between spaces, so they never match a boundary, but a
    # balcony without its guard rail is wrong. Keep them.
    walls.extend(w for w in recovered if w.kind == "parapet")
    dropped = len([w for w in recovered if w.kind != "parapet" and id(w) not in used])

    before = len(walls)
    walls, remap = _merge_collinear(walls, prefix)
    links = [(b, remap.get(w.id) if w is not None else None) for b, w in links]

    notes.append(
        f"walls from boundaries: {matched} matched to recovered CAD geometry, "
        f"{synthesised} synthesised where none was recovered, {dropped} recovered "
        f"wall(s) matched no boundary and were dropped; {before} runs fused into "
        f"{len(walls)} walls"
    )
    return walls, links


def _partition_by_labels(
    face: ShpPolygon,
    points: Sequence[tuple[float, float]],
    targets: Sequence[float | None] = (),
    *,
    cell_mm: float = 110.0,
) -> list[ShpPolygon | None]:
    """Divide a multi-label region into connected cells sized to each label's target.

    Used when one enclosed region holds several room labels — a genuinely open-plan
    living/dining/lobby, or a partition wall that was not recovered.

    **Balanced region growing**, seeded at each label and expanded a cell at a time,
    always advancing whichever label is furthest behind its target area. Two
    properties come out of that, and both were missing from the attempts before it:

    * **Connectivity is structural.** A region only ever grows onto a cell adjacent
      to one it already owns, so it cannot fragment. The previous attempt used a
      multiplicatively-weighted Voronoi, which *does* produce disconnected cells;
      since a room polygon must be a single ring, the stray fragments were discarded
      and LIVING collapsed to 1.1 m² against a printed 17.4 m². Guaranteeing
      connectivity by construction is what makes area-targeting safe to do at all.

    * **Areas track the printed dimensions.** The drawing states each room's size, so
      the split honours it instead of falling where the labels happen to sit. Plain
      nearest-label Voronoi gave LIVING 4.8 m² against the same printed 17.4 m²,
      because the DINING and LOBBY labels sit close to it.

    Growth order is by *fill ratio* (claimed / target) rather than absolute deficit,
    so a small utility room is not starved while a large living room is filled.

    Boundaries are quantised to ``cell_mm`` and simplified; these divide zones inside
    an open plan rather than marking walls, so that costs nothing visible.

    Returns cells in the same order as ``points``; None where a label won nothing.
    """
    n = len(points)
    if n < 2:
        return [face]

    from shapely.geometry import box as shp_box
    from shapely.prepared import prep

    minx, miny, maxx, maxy = face.bounds
    # Guard the raster size rather than risk a pathological grid on a big region.
    while ((maxx - minx) / cell_mm) * ((maxy - miny) / cell_mm) > 240_000:
        cell_mm *= 1.4
    nx = max(1, int((maxx - minx) / cell_mm))
    ny = max(1, int((maxy - miny) / cell_mm))
    dx = (maxx - minx) / nx
    dy = (maxy - miny) / ny

    prepared = prep(face)
    inside: dict[tuple[int, int], tuple[float, float]] = {}
    for j in range(ny):
        cy = miny + (j + 0.5) * dy
        for i in range(nx):
            cx = minx + (i + 0.5) * dx
            if prepared.contains(ShpPoint(cx, cy)):
                inside[(i, j)] = (cx, cy)
    if not inside:
        return [face] + [None] * (n - 1)

    cell_area = face.area / len(inside)

    # Targets, normalised so they sum to the region's actual area. Labels with no
    # printed dimension take the mean of those that have one.
    tgt = [t if (t and t > 0) else None for t in (list(targets) + [None] * n)[:n]]
    known = [t for t in tgt if t is not None]
    fill = (sum(known) / len(known)) if known else face.area / n
    want = [t if t is not None else fill for t in tgt]
    total = sum(want) or 1.0
    want = [w * face.area / total for w in want]

    # Seed each label at the cell containing it, or the nearest interior cell.
    owner: dict[tuple[int, int], int] = {}
    frontier: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    claimed = [0.0] * n
    for li, (px, py) in enumerate(points):
        key = (int((px - minx) / dx), int((py - miny) / dy))
        if key not in inside or key in owner:
            free = [k for k in inside if k not in owner]
            if not free:
                continue
            key = min(free, key=lambda k: math.dist((px, py), inside[k]))
        owner[key] = li
        frontier[li].append(key)
        claimed[li] += cell_area

    active = {li for li in range(n) if frontier[li]}
    while active:
        # Whichever label is proportionally furthest behind grows next.
        li = min(active, key=lambda k: claimed[k] / max(want[k], 1e-9))
        queue = frontier[li]
        grew = False
        while queue:
            i, j = queue.pop(0)
            for ni, nj in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                k = (ni, nj)
                if k in inside and k not in owner:
                    owner[k] = li
                    queue.append(k)
                    claimed[li] += cell_area
                    grew = True
                    break
            if grew:
                queue.insert(0, (i, j))
                break
        if not grew:
            active.discard(li)

    out: list[ShpPolygon | None] = []
    for li in range(n):
        # Boxes are built from exact grid indices, not from a cell centre ± half a
        # cell. The centre form leaves sub-micron float gaps between neighbours, and
        # `unary_union` then returns a MultiPolygon of slivers even though the cells
        # are genuinely 4-connected — taking the largest piece of that discarded 40%
        # of the area and quietly undid the whole point of growing connected regions.
        boxes = [
            shp_box(
                minx + i * dx, miny + j * dy, minx + (i + 1) * dx, miny + (j + 1) * dy
            )
            for (i, j) in inside
            if owner.get((i, j)) == li
        ]
        if not boxes:
            out.append(None)
            continue
        # A hair of dilation makes edge-adjacency unambiguous to GEOS.
        merged = unary_union([b.buffer(dx * 0.02, join_style=2) for b in boxes])
        if isinstance(merged, MultiPolygon):
            # Growth guarantees connectivity, so this is a degenerate-grid fallback
            # rather than an expected path.
            merged = max(merged.geoms, key=lambda g: g.area)
        if not isinstance(merged, ShpPolygon) or merged.area <= 0:
            out.append(None)
            continue
        # Simplify *before* clipping to the face, then clip.
        #
        # Order matters and getting it wrong is subtle. Only the internal divisions
        # between zones are jagged (they follow the grid); the zone's outer edges come
        # from the face itself and are already straight. Simplifying after the clip
        # straightens the jagged parts but also nibbles the clean outer edges inward,
        # and furniture anchored against a wall then fails the room-containment test —
        # which showed up as a living room with no sofa and no coffee table.
        # Simplifying first, then clipping, restores the outer boundary exactly.
        simple = ShpPolygon(merged.exterior.coords).simplify(
            cell_mm * 0.8, preserve_topology=True
        )
        if not isinstance(simple, ShpPolygon) or simple.area <= 0:
            simple = merged
        ring = simple.intersection(face)
        if isinstance(ring, MultiPolygon):
            ring = max(ring.geoms, key=lambda g: g.area) if not ring.is_empty else simple
        out.append(ring if isinstance(ring, ShpPolygon) and ring.area > 0 else simple)
    return out



def _extract_rooms(
    unit: _UnitRegion,
    walls: Sequence[Wall],
    cfg: ExtractConfig,
    prefix: str,
    warnings: list[str],
    barriers: Sequence[LineString] = (),
) -> list[Room]:
    """Turn the wall network's enclosed faces into labelled rooms."""
    clip = unit.polygon.buffer(cfg.unit_clip_margin_mm)
    pieces = _room_faces(walls, clip, cfg, barriers)

    rooms: list[Room] = []
    used_labels: set[int] = set()
    leaked = 0

    # Split any face that holds several room labels. Two situations produce one:
    # a genuinely open-plan living/dining/lobby, which *is* one space with several
    # named zones; and a failure to recover a partition wall. Partitioning by
    # nearest label handles both, and it is safe because it only divides the room
    # polygon — the wall list is untouched, so no wall is ever invented. The
    # alternative, naming a 500 sqft region "LOBBY" and discarding the other
    # labels, loses rooms the drawing plainly declares.
    work: list[tuple[ShpPolygon, list[int]]] = []
    for piece in sorted(pieces, key=lambda p: -p.area):
        if piece.area < cfg.min_room_area_mm2:
            continue
        # Interior rings would be columns or ducts standing inside a room; the
        # outer shell is the floor plate we want.
        poly = ShpPolygon(piece.exterior.coords)
        idxs = [
            i for i, e in enumerate(unit.labels) if poly.contains(ShpPoint(*e.points[0]))
        ]
        if len(idxs) > 1:
            if piece.area > cfg.max_room_area_mm2:
                leaked += 1
            # Feed each label's printed W x D as the target area, so the split
            # reproduces the sizes the drawing declares rather than dividing by
            # position alone.
            targets: list[float | None] = []
            for i in idxs:
                d = parse_dimension_pair(unit.labels[i].text or "")
                targets.append(d.area_mm2 if d else None)
            cells = _partition_by_labels(
                poly, [unit.labels[i].points[0] for i in idxs], targets
            )
            for idx, cell in zip(idxs, cells):
                if cell is not None and cell.area >= cfg.min_room_area_mm2 * 0.5:
                    work.append((cell, [idx]))
        elif len(idxs) == 1 and (dim := parse_dimension_pair(unit.labels[idxs[0]].text or "")) \
                and poly.area > dim.area_mm2 * cfg.oversize_split_ratio:
            # One label, but the face is far larger than that label's printed size.
            # The face has swallowed an adjacent space whose own label fell outside it
            # (clipped away, or never drawn). The drawing still tells us how big the
            # labelled room is, so carve exactly that off and leave the remainder as an
            # unnamed space rather than reporting a 287 sqft bedroom the plan prints as
            # 132 sqft. Honest and geometrically defensible: we know one room's size,
            # so we know the leftover is something else.
            far = max(
                poly.exterior.coords[:-1],
                key=lambda c: math.dist(c, unit.labels[idxs[0]].points[0]),
            )
            inner = poly.buffer(-min(400.0, math.sqrt(poly.area) * 0.1))
            if isinstance(inner, MultiPolygon):
                inner = max(inner.geoms, key=lambda g: g.area)
            seed2 = (
                inner.exterior.interpolate(inner.exterior.project(ShpPoint(*far)))
                if isinstance(inner, ShpPolygon) and not inner.is_empty
                else ShpPoint(*far)
            )
            cells = _partition_by_labels(
                poly,
                [unit.labels[idxs[0]].points[0], (seed2.x, seed2.y)],
                [dim.area_mm2, max(poly.area - dim.area_mm2, cfg.min_room_area_mm2)],
            )
            if cells[0] is not None and cells[0].area >= cfg.min_room_area_mm2:
                work.append((cells[0], [idxs[0]]))
                if cells[1] is not None and cells[1].area >= cfg.min_room_area_mm2:
                    work.append((cells[1], []))
                warnings.append(
                    f"{unit.name}: a {mm_to_sqft(poly.area):.0f} sqft region carried one "
                    f"label printed as {dim.area_sqft:.0f} sqft; split off the stated "
                    "footprint and left the remainder unnamed"
                )
            else:
                work.append((poly, idxs))
        else:
            work.append((poly, idxs))

    for poly, idxs in sorted(work, key=lambda t: -t[0].area):
        inside = [(i, unit.labels[i]) for i in idxs if i not in used_labels]

        if inside:
            # Nearest label to the centroid names the room; the rest indicate the
            # region spans more than one room.
            idx, label_ent = min(
                inside, key=lambda t: math.dist((poly.centroid.x, poly.centroid.y), t[1].points[0])
            )
            used_labels.add(idx)
            raw = label_ent.text or ""
            kind = classify_room_name(raw)
            dim = parse_dimension_pair(raw)
            label = raw.strip().splitlines()[0].strip()
            extra = [f"region also contains {len(inside) - 1} other room label(s)"] if len(inside) > 1 else []
        else:
            kind, dim, label, extra = RoomKind.UNKNOWN, None, "", ["no room label inside region"]

        if kind.is_outdoor:
            floor_level, ceil = -50.0, cfg.parapet_height_mm
        elif kind.is_wet:
            floor_level, ceil = -20.0, cfg.wall_height_mm - 300.0
        else:
            floor_level, ceil = 0.0, cfg.wall_height_mm

        coords = [(float(x), float(y)) for x, y in poly.exterior.coords[:-1]]
        room = Room(
            id=f"{prefix}-r{len(rooms) + 1}",
            kind=kind,
            label=label,
            polygon=coords,
            centroid=(float(poly.centroid.x), float(poly.centroid.y)),
            stated_dimension=dim,
            ceiling_height_mm=ceil,
            floor_level_mm=floor_level,
            source=Source.CAD_VECTOR,
            confidence=0.95 if inside else 0.5,
            notes=extra,
        )
        rooms.append(room)

    if leaked:
        warnings.append(
            f"{unit.name}: {leaked} region(s) span multiple room labels — walls did not fully "
            "close, so those rooms are merged. Check for missing partition line work."
        )

    unmatched = [e for i, e in enumerate(unit.labels) if i not in used_labels]
    if unmatched:
        names = ", ".join(sorted({(e.text or "").splitlines()[0].strip() for e in unmatched})[:6])
        warnings.append(
            f"{unit.name}: {len(unmatched)} labelled room(s) had no enclosed region ({names}) — "
            "their walls were not recovered."
        )
    return rooms


def _openings_from_gaps(
    gap_specs: Sequence[tuple[str, tuple[float, float], tuple[float, float]]],
    walls: Sequence[Wall],
    door_ents: Sequence[FlatEntity],
    window_ents: Sequence[FlatEntity],
    cfg: ExtractConfig,
    prefix: str,
) -> list[Opening]:
    """Turn wall-face gaps into openings, named by whatever symbol sits in them.

    The gap is the *measurement* — it is where the drawing actually cut through both
    faces — so its position and width are trusted. The nearby symbol only supplies
    the *kind*: a swing arc or leaf means a door, window line work means a window,
    and a gap with neither is a cased opening, which is what it visually is. That
    ordering matters: symbol detection is unreliable (three door blocks for sixteen
    rooms) while the gap is always present, so the gap must not depend on the symbol.
    """
    door_pts: list[tuple[float, float]] = []
    for cluster in _cluster_entities(door_ents, cfg.door_cluster_radius_mm):
        xs = [p[0] for e in cluster for p in e.points]
        ys = [p[1] for e in cluster for p in e.points]
        if max(max(xs) - min(xs), max(ys) - min(ys)) >= 300.0:
            door_pts.append(((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0))

    win_pts: list[tuple[float, float]] = []
    for e in window_ents:
        if len(e.points) < 2:
            continue
        xs = [p[0] for p in e.points]
        ys = [p[1] for p in e.points]
        win_pts.append(((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0))

    by_id = {w.id: w for w in walls}
    out: list[Opening] = []
    for wall_id, ga, gb in gap_specs:
        wall = by_id.get(wall_id)
        if wall is None:
            continue
        L = wall.length_mm
        if L < 1e-6:
            continue
        dx, dy = wall.end[0] - wall.start[0], wall.end[1] - wall.start[1]

        def _proj(pt: tuple[float, float]) -> tuple[float, float]:
            t = ((pt[0] - wall.start[0]) * dx + (pt[1] - wall.start[1]) * dy) / (L * L)
            tc = max(0.0, min(1.0, t))
            return t, math.dist(pt, wall.point_at(tc))

        ta, pa = _proj(ga)
        tb, pb = _proj(gb)
        # A gap that no longer lies on this wall is not evidence for an opening in it.
        if max(pa, pb) > 300.0:
            continue
        t0, t1 = sorted((max(0.0, min(1.0, ta)), max(0.0, min(1.0, tb))))
        width = abs(t1 - t0) * L
        if width < cfg.min_gap_opening_mm:
            continue
        t_mid = (t0 + t1) / 2.0
        centre = wall.point_at(t_mid)

        near_door = min((math.dist(centre, p) for p in door_pts), default=1e9)
        near_win = min((math.dist(centre, p) for p in win_pts), default=1e9)

        if near_win <= cfg.symbol_match_mm and near_win <= near_door:
            kind = OpeningKind.WINDOW if width < 2200 else OpeningKind.FRENCH_WINDOW
            height, sill, swing = cfg.default_window_height_mm, cfg.default_window_sill_mm, "none"
        elif near_door <= cfg.symbol_match_mm:
            kind = OpeningKind.DOOR if width < 1100 else OpeningKind.DOUBLE_DOOR
            height, sill, swing = cfg.default_door_height_mm, 0.0, "left"
        else:
            # No symbol: a cased opening. Full door height and no leaf, which is
            # both what it looks like on the drawing and safe to walk through in 3D.
            kind = OpeningKind.OPENING
            height, sill, swing = cfg.default_door_height_mm, 0.0, "none"

        out.append(
            Opening(
                id=f"{prefix}-g{len(out) + 1}",
                wall_id=wall_id,
                kind=kind,
                t_center=t_mid,
                width_mm=min(width, wall.length_mm * 0.97),
                height_mm=height,
                sill_mm=sill,
                swing=swing,  # type: ignore[arg-type]
                source=Source.CAD_VECTOR,
                # Higher than symbol-derived openings: this one was measured from
                # the wall itself, so it wins the overlap filter.
                confidence=0.92,
            )
        )
    return out


def _attach_openings(
    entities_door: Sequence[FlatEntity],
    entities_window: Sequence[FlatEntity],
    walls: Sequence[Wall],
    clip: ShpPolygon,
    cfg: ExtractConfig,
    prefix: str,
    gap_specs: Sequence[tuple[str, tuple[float, float], tuple[float, float]]] = (),
) -> list[Opening]:
    """Place doors and windows onto their host walls.

    Doors are found by their swing arc, which is the one unambiguous door symbol
    in plan: the arc radius *is* the leaf width, and the arc centre sits at the
    hinge. Windows are found from line work running along a wall. Both are
    projected onto the nearest wall centreline and stored as a normalised position
    so they survive later edits to the wall.
    """
    openings: list[Opening] = list(
        _openings_from_gaps(gap_specs, walls, entities_door, entities_window, cfg, prefix)
    )

    def host_wall(
        p: tuple[float, float], along: float | None, max_d: float | None = None
    ) -> tuple[Wall, float] | None:
        """Nearest wall to ``p``, with the normalised position of the projection."""
        best: tuple[float, Wall, float] | None = None
        for w in walls:
            L = w.length_mm
            if L < 1e-6:
                continue
            dx, dy = w.end[0] - w.start[0], w.end[1] - w.start[1]
            t = ((p[0] - w.start[0]) * dx + (p[1] - w.start[1]) * dy) / (L * L)
            t_clamped = max(0.0, min(1.0, t))
            proj = w.point_at(t_clamped)
            d = math.dist(p, proj)
            # A centre that projects past the wall's end is on some other wall: charge
            # the overshoot as distance. Without this a long wall 115 mm to the side
            # beats the short correct wall, the clamp drags the opening to the long
            # wall's end, and the "slide inside" rescue then walks it 843 mm from where
            # the drawing put it.
            if t != t_clamped:
                d += abs(t - t_clamped) * L
            # Prefer a wall the opening's own direction agrees with.
            if along is not None:
                wall_ang = math.degrees(w.angle_rad) % 180.0
                diff = abs(wall_ang - along % 180.0)
                if min(diff, 180.0 - diff) > 25.0:
                    d += 400.0
            if best is None or d < best[0]:
                best = (d, w, t_clamped)
        limit = cfg.opening_attach_tol_mm if max_d is None else max_d
        if best is None or best[0] > limit:
            return None
        return best[1], best[2]

    # --- doors from swing arcs ---
    lo, hi = cfg.door_arc_radius_range
    for e in entities_door:
        if e.dxftype != "ARC" or len(e.points) < 3:
            continue
        xs = [p[0] for p in e.points]
        ys = [p[1] for p in e.points]
        radius = max(max(xs) - min(xs), max(ys) - min(ys))
        if not (lo <= radius <= hi):
            continue
        # The hinge is the arc centre: the point equidistant from the endpoints
        # at `radius`. For a quarter-circle swing it is the corner of the bbox
        # opposite the arc bulge, which the arc's own endpoints identify.
        p0, p1 = e.points[0], e.points[-1]
        # Candidate centres are the two bbox corners the chord does not span.
        corners = [
            (min(xs), min(ys)),
            (max(xs), min(ys)),
            (max(xs), max(ys)),
            (min(xs), max(ys)),
        ]
        hinge = min(
            corners,
            key=lambda c: abs(math.dist(c, p0) - radius) + abs(math.dist(c, p1) - radius),
        )
        if not clip.contains(ShpPoint(*hinge)):
            continue
        # The door sits along the wall from the hinge toward whichever arc
        # endpoint lies on the wall line; take the midpoint of hinge and that end.
        far = p0 if math.dist(hinge, p0) >= math.dist(hinge, p1) else p1
        centre = ((hinge[0] + far[0]) / 2.0, (hinge[1] + far[1]) / 2.0)
        along = math.degrees(math.atan2(far[1] - hinge[1], far[0] - hinge[0]))
        hit = host_wall(centre, along)
        if hit is None:
            hit = host_wall(hinge, None)
        if hit is None:
            continue
        wall, t = hit
        width = min(radius, wall.length_mm * 0.95)
        openings.append(
            Opening(
                id=f"{prefix}-o{len(openings) + 1}",
                wall_id=wall.id,
                kind=OpeningKind.DOOR if width < 1100 else OpeningKind.DOUBLE_DOOR,
                t_center=t,
                width_mm=width,
                height_mm=cfg.default_door_height_mm,
                sill_mm=0.0,
                swing="left",
                source=Source.CAD_VECTOR,
                confidence=0.85,
            )
        )

    # --- doors drawn without a swing arc ---
    # Only 5 of 19 door entities in the demo file are arcs, so an arc-only pass
    # cuts a hole for barely a quarter of the doorways and the rest of the flat
    # renders as sealed boxes. Fall back to the cluster's bounding box: its longer
    # side, projected onto the host wall, is the doorway width.
    for cluster in _cluster_entities(entities_door, cfg.door_cluster_radius_mm):
        if any(e.dxftype == "ARC" for e in cluster):
            continue  # already handled, more precisely, above
        xs = [p[0] for e in cluster for p in e.points]
        ys = [p[1] for e in cluster for p in e.points]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        span = max(w, h)
        if not (lo <= span <= hi * 1.6):
            continue
        centre = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
        if not clip.contains(ShpPoint(*centre)):
            continue
        along = 0.0 if w >= h else 90.0
        hit = host_wall(centre, along)
        if hit is None:
            continue
        wall, t = hit
        width = min(span, wall.length_mm * 0.95)
        openings.append(
            Opening(
                id=f"{prefix}-o{len(openings) + 1}",
                wall_id=wall.id,
                kind=OpeningKind.DOOR if width < 1100 else OpeningKind.DOUBLE_DOOR,
                t_center=t,
                width_mm=width,
                height_mm=cfg.default_door_height_mm,
                sill_mm=0.0,
                swing="left",
                source=Source.CAD_VECTOR,
                confidence=0.6,
            )
        )

    # --- windows from line work parallel to a wall ---
    # Group window line work into runs, then attach each run.
    win_segs: list[Segment] = _segments_from(entities_window, clip)
    runs: list[tuple[tuple[float, float], float, float]] = []  # (centre, length, angle)
    for s in win_segs:
        if s.length < 500.0:
            continue
        runs.append(
            (
                ((s.a[0] + s.b[0]) / 2.0, (s.a[1] + s.b[1]) / 2.0),
                s.length,
                s.angle_mod180,
            )
        )
    # Merge runs that describe the same window (both jamb lines / sill + head).
    merged: list[tuple[tuple[float, float], float, float]] = []
    for centre, length, ang in sorted(runs, key=lambda r: -r[1]):
        if any(
            math.dist(centre, c2) < 600.0 and abs(length - l2) < 900.0
            for c2, l2, _ in merged
        ):
            continue
        merged.append((centre, length, ang))

    for centre, length, ang in merged:
        # Glazing runs along the face of its wall, so its centreline sits at most half
        # a wall thickness plus drafting slack away — 350 mm. The general 900 mm attach
        # tolerance exists for door symbols, whose arcs stand off the wall; applied to
        # windows it planted six of them 500-850 mm from the drawn glazing, on walls
        # the drawing never glazed. No host inside 350 mm means this glazing belongs
        # to no recovered wall, and no window is the honest result.
        hit = host_wall(centre, ang, max_d=350.0)
        if hit is None:
            continue
        wall, t = hit
        width = min(length, wall.length_mm * 0.95)
        if width < 450.0:
            continue
        openings.append(
            Opening(
                id=f"{prefix}-o{len(openings) + 1}",
                wall_id=wall.id,
                kind=OpeningKind.WINDOW if width < 2200 else OpeningKind.FRENCH_WINDOW,
                t_center=t,
                width_mm=width,
                height_mm=cfg.default_window_height_mm,
                sill_mm=cfg.default_window_sill_mm,
                source=Source.CAD_VECTOR,
                confidence=0.7,
            )
        )

    # Reject openings that do not fit inside their host wall.
    #
    # `host_wall` clamps the projected position to [0, 1], so an opening whose centre
    # projects beyond the wall's end lands at t=0 or t=1 with half its width hanging
    # off. That produced a 1472 mm window on a 1550 mm wall at t=0 — geometrically it
    # removes the whole wall, and it means the opening belonged to a different wall.
    kept: list[Opening] = []
    for o in openings:
        w = next((x for x in walls if x.id == o.wall_id), None)
        if w is None or w.length_mm <= 0:
            continue
        half = (o.width_mm / 2.0) / w.length_mm
        if o.t_center - half < -1e-6 or o.t_center + half > 1.0 + 1e-6:
            # Try to slide it inside; if it cannot fit at all, drop it.
            if o.width_mm > w.length_mm * 0.92:
                continue
            slid = min(max(o.t_center, half), 1.0 - half)
            if abs(slid - o.t_center) * w.length_mm > 300.0:
                # Sliding further than a jamb's width means the opening never belonged
                # to this wall — placing it is inventing an opening the drawing shows
                # somewhere else.
                continue
            o.t_center = slid
        kept.append(o)
    openings = kept

    # Drop openings that overlap on the same wall, keeping the more confident.
    openings.sort(key=lambda o: (-o.confidence, o.wall_id, o.t_center))
    accepted: list[Opening] = []
    for o in openings:
        w = next((x for x in walls if x.id == o.wall_id), None)
        if w is None or w.length_mm <= 0:
            continue
        half = (o.width_mm / 2.0) / w.length_mm
        clash = False
        for a in accepted:
            if a.wall_id != o.wall_id:
                continue
            a_w = next((x for x in walls if x.id == a.wall_id), None)
            if a_w is None:
                continue
            a_half = (a.width_mm / 2.0) / a_w.length_mm
            if abs(a.t_center - o.t_center) < (half + a_half) * 0.85:
                clash = True
                break
        if not clash:
            accepted.append(o)
    for i, o in enumerate(accepted, 1):
        o.id = f"{prefix}-o{i}"
    return accepted


_FIXTURE_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("wc", ("wc", "closet", "commode", "ewc", "toilet")),
    ("basin", ("basin", "washbasin", "lavatory", "sink")),
    ("shower", ("shower", "cubicle")),
    ("bathtub", ("tub", "bath")),
    ("kitchen_sink", ("ksink", "kitchen sink")),
    ("hob", ("hob", "cooktop", "stove", "burner")),
    ("counter", ("counter", "platform", "slab")),
    ("bed", ("bed", "cot", "mattress")),
    ("sofa", ("sofa", "couch", "settee", "lounge")),
    ("dining_table", ("dining", "dtable")),
    ("wardrobe", ("wardrobe", "cupboard", "closet unit", "almirah")),
    ("washing_machine", ("washing", "wm", "washer")),
    ("fridge", ("fridge", "refrig")),
)


def _categorise_fixture(label: str, block: str | None) -> str:
    hay = f"{label} {block or ''}".lower()
    for cat, keys in _FIXTURE_CATEGORIES:
        if any(k in hay for k in keys):
            return cat
    return "unknown"


def _extract_fixtures(
    buckets: _Buckets, clip: ShpPolygon, rooms: Sequence[Room], prefix: str
) -> list[FixtureHint]:
    """Capture drawn fixtures as placement hints.

    The architect's sanitaryware and kitchen layout encode plumbing runs and
    service constraints. Honouring them is the difference between a model that
    matches the drawing and a model that merely resembles it.
    """
    out: list[FixtureHint] = []
    role_cat = {
        LayerRole.SANITARY: "sanitary",
        LayerRole.KITCHEN_FIT: "kitchen",
        LayerRole.FURNITURE: "furniture",
    }
    room_shapes = [(r, ShpPolygon(r.polygon)) for r in rooms if len(r.polygon) >= 3]

    for role, default_cat in role_cat.items():
        entities = list(buckets.get(role))
        # Fittings are not always drawn as closed polylines. A counter run is often a
        # set of separate line segments that happen to enclose a shape, and requiring
        # three or more points in a single entity threw all of those away — on the demo
        # drawing that discarded 80 of the 86 entities on the fixed-equipment layer.
        # Noding the segments and taking the closed faces recovers them.
        loose = [
            LineString(e.points)
            for e in entities
            if len(e.points) == 2 and math.dist(e.points[0], e.points[1]) > 1e-6
        ]
        if loose:
            template = next((e for e in entities if len(e.points) == 2), None)
            for face in polygonize(unary_union(loose)):
                if not isinstance(face, ShpPolygon) or face.area < 40_000.0:
                    continue
                entities.append(
                    FlatEntity(
                        dxftype="LWPOLYLINE",
                        layer=template.layer if template else "",
                        points=[(float(x), float(y)) for x, y in face.exterior.coords[:-1]],
                        closed=True,
                        block_path=template.block_path if template else (),
                    )
                )

        for e in entities:
            if len(e.points) < 3:
                continue
            cx = sum(p[0] for p in e.points) / len(e.points)
            cy = sum(p[1] for p in e.points) / len(e.points)
            if not clip.contains(ShpPoint(cx, cy)):
                continue
            if _shoelace(e.points) < 40_000.0:  # < 0.04 m², a symbol not an object
                continue
            room_id = next(
                (r.id for r, shp in room_shapes if shp.contains(ShpPoint(cx, cy))), None
            )
            # A fixture outside every room is building services, not a fitting.
            #
            # Assembling loose line segments recovered eight shapes on the demo drawing
            # and every one of them was junk: two 1.71 x 2.21 m and two 1.61 x 2.10 m
            # boxes — lift cars in the core — plus four 1.05 x 0.16 m slivers, all in no
            # room at all and all labelled "kitchen" because the layer is A-EQPM-FIXD.
            # Emitting those is worse than emitting nothing, because the layout solver
            # *locks* fixtures on the grounds that they encode plumbing runs, so one
            # stray box becomes an immovable 3.8 m² counter.
            if room_id is None:
                continue
            cat = _categorise_fixture(e.layer, e.block_name)
            # Size sanity for what the category claims to be. A kitchen counter is about
            # 600 mm deep and a WC or basin under a metre; anything whose *smaller*
            # dimension is over 1.2 m is a shaft, a room outline or a hatch boundary.
            xs = [p[0] for p in e.points]
            ys = [p[1] for p in e.points]
            minor = min(max(xs) - min(xs), max(ys) - min(ys))
            if minor > 1200.0:
                continue
            out.append(
                FixtureHint(
                    id=f"{prefix}-f{len(out) + 1}",
                    label=e.layer,
                    footprint=[(float(x), float(y)) for x, y in e.points],
                    room_id=room_id,
                    category=cat if cat != "unknown" else default_cat,
                    source=Source.CAD_VECTOR,
                    confidence=0.7,
                    block_name=e.block_name,
                )
            )
    return out


def _report_fixture_absence(buckets: _Buckets, notes: list[str]) -> None:
    """Say so when a drawing carries no fittings at all.

    ``fixtures: 0`` reads like a silent success when it is really "the architect did not
    draw any". This drawing has 28 layers and not one of them is sanitaryware or
    furniture, so every bathroom and kitchen fitting in the model is invented by the
    layout solver rather than read. That is worth knowing when the kitchen comes out with
    a fridge and nothing else.
    """
    missing = [
        role.value
        for role in (LayerRole.SANITARY, LayerRole.FURNITURE, LayerRole.KITCHEN_FIT)
        if not buckets.get(role)
    ]
    if missing:
        notes.append(
            "no "
            + ", ".join(missing)
            + " layer content in this drawing — fittings in those rooms are placed by "
            "the layout solver, not read from the file"
        )


_CIRCULATION_KINDS = (RoomKind.ENTRANCE, RoomKind.LOBBY, RoomKind.CORRIDOR)


def _door_preference(room: Room, neighbour: Room | None) -> float:
    """How appropriate is a door between these two spaces?

    Length alone picked the longest shared wall, which put a door directly between two
    bedrooms. Every room in a flat is entered from circulation unless there is a reason
    otherwise, so the graph search is weighted to reflect how a plan is actually
    organised rather than which wall happens to be longest.
    """
    if neighbour is None:
        return 1.0
    if neighbour.kind in _CIRCULATION_KINDS:
        return 2.0  # rooms are entered from the hall
    if room.kind == RoomKind.BATHROOM and neighbour.kind in (
        RoomKind.BEDROOM,
        RoomKind.MASTER_BEDROOM,
        RoomKind.WARDROBE,
    ):
        return 1.5  # an ensuite is entered from its bedroom or dressing room
    if room.kind.is_wet and neighbour.kind in (RoomKind.KITCHEN, RoomKind.DINING):
        return 0.25  # a WC opening onto where people eat
    both_private = {room.kind, neighbour.kind} <= {
        RoomKind.BEDROOM,
        RoomKind.MASTER_BEDROOM,
        RoomKind.KIDS_BEDROOM,
    }
    return 0.2 if both_private else 1.0


def _doors_from_topology(
    smap: SpaceMap,
    links: Sequence[tuple[Any, Wall | None]],
    id_map: dict[str, str],
    rooms: Sequence[Room],
    walls: Sequence[Wall],
    openings: list[Opening],
    cfg: ExtractConfig,
    prefix: str,
    warnings: list[str],
    door_symbols: Sequence[tuple[float, float]] = (),
) -> int:
    """Stage 3: place doors on the mapped boundaries, so the flat is connected.

    The old pass searched outward from each room for a nearby wall long enough to hold
    a door. That is why doors kept landing wrongly: "near this room" is not the same as
    "between this room and that one", so a door could end up on a wall the room merely
    touches at a corner, or on an outside wall, or duplicating one that already existed
    a metre away.

    With the house mapped first, the question is no longer geometric but topological.
    Every wall is a *boundary between two named spaces*, so the door goes on the
    boundary the two spaces actually share, centred along it. And it is only placed
    where one is genuinely needed: this walks the connectivity graph out from the front
    door and adds exactly the doors required to reach every room, which is the same
    thing as building a spanning tree over the adjacency graph.

    Open thresholds are already edges in that graph. Living into dining needs no door
    and gets none.
    """
    by_boundary: dict[int, Wall] = {}
    for i, (_b, w) in enumerate(links):
        if w is not None:
            by_boundary[i] = w
    wall_map = {w.id: w for w in walls}

    room_of = {sid: rid for sid, rid in id_map.items()}
    by_room = {r.id: r for r in rooms}

    # Resolve which rooms each *drawn* door connects, by probing both sides of its
    # wall. Drawn doors arrive with connects=(None, None), so the reachability walk
    # could not see that a doorway already links two rooms and would add its own an
    # arm's length away — a drawn 590 mm door and an inferred 900 mm door 458 mm
    # apart on parallel walls.
    _shapes_conn = [
        (r, ShpPolygon(r.polygon)) for r in rooms if len(r.polygon) >= 3
    ]
    for o in openings:
        if not o.kind.is_door or (o.connects and any(o.connects)):
            continue
        w = wall_map.get(o.wall_id)
        if w is None:
            continue
        cx, cy = w.point_at(o.t_center)
        nx, ny = w.normal()
        off = w.thickness_mm / 2.0 + 200.0
        found: list[str] = []
        for sgn in (1.0, -1.0):
            probe = ShpPoint(cx + sgn * nx * off, cy + sgn * ny * off)
            hit = next((r.id for r, q in _shapes_conn if q.contains(probe)), None)
            if hit is not None:
                found.append(hit)
        if len(found) == 2 and found[0] != found[1]:
            o.connects = (found[0], found[1])

    reachable: set[str] = set()
    OUT = "\x00outside"

    # Edges the plan already provides: open thresholds, and doors that were drawn.
    edges: list[tuple[str, str, int | None]] = []
    for i, (b, wall) in enumerate(links):
        a = room_of.get(b.space_a)
        if a is None:
            continue
        other = room_of.get(b.space_b) if b.space_b else OUT
        if other is None:
            continue
        if not b.walled and b.space_b is not None:
            edges.append((a, other, None))  # continuous space; no door required
            continue
        if wall is None:
            continue
        served = any(
            o.kind.is_door
            and o.wall_id == wall.id
            and LineString([wall.start, wall.end])
            .interpolate(o.t_center, normalized=True)
            .distance(b.line)
            <= cfg.opening_serves_tol_mm
            for o in openings
        )
        if served:
            edges.append((a, other, None))

    def spread() -> None:
        changed = True
        while changed:
            changed = False
            for a, b2, _ in edges:
                for x, y in ((a, b2), (b2, a)):
                    if x in reachable and y not in reachable:
                        reachable.add(y)
                        changed = True

    # Start from outside. A flat with no recovered front door still has one, so fall
    # back to the entrance or lobby, and failing that the largest circulation space.
    reachable.add(OUT)
    spread()
    if len(reachable) == 1:
        seed = next(
            (
                r
                for k in (RoomKind.ENTRANCE, RoomKind.LOBBY, RoomKind.CORRIDOR)
                for r in rooms
                if r.kind == k
            ),
            max(rooms, key=lambda r: ShpPolygon(r.polygon).area) if rooms else None,
        )
        if seed is not None:
            reachable.add(seed.id)
            spread()

    needed = [
        r
        for r in rooms
        if r.id not in reachable
        and r.kind not in (RoomKind.DUCT, RoomKind.LIFT, RoomKind.STAIRS)
        and not r.kind.is_outdoor
    ]
    added = 0
    guard = 0
    while needed and guard < len(rooms) * 2:
        guard += 1
        # Take the longest boundary from anything still cut off to anything already
        # reached: the longest shared wall is where a door most plausibly goes, and
        # working longest-first keeps the doors on the main circulation route.
        best: tuple[float, int, Wall, Room, str] | None = None
        for i, (b, wall) in enumerate(links):
            if wall is None or b.space_b is None or not b.walled:
                continue
            a, c = room_of.get(b.space_a), room_of.get(b.space_b)
            if a is None or c is None:
                continue
            for inside, out in ((a, c), (c, a)):
                if inside in reachable or out not in reachable:
                    continue
                room = next((r for r in rooms if r.id == inside), None)
                if room is None or room not in needed:
                    continue
                if b.length < cfg.min_gap_opening_mm * 1.6:
                    continue
                score = b.length * _door_preference(room, by_room.get(out))
                if best is None or score > best[0]:
                    best = (score, i, wall, room, out)
        if best is None:
            break

        _, _, wall, room, other_id = best
        b = links[best[1]][0]
        line = LineString([wall.start, wall.end])
        # Position at a drawn door symbol when one sits on this boundary. An inferred
        # door's *presence* is topology, but the drawing frequently marks where the
        # doorway actually is — an arc or leaf on the door layer — and centring on the
        # boundary while a symbol sits 800 mm away is a wrong placement the file could
        # have prevented.
        anchor = b.line.interpolate(0.5, normalized=True)
        near_syms = [
            pxy for pxy in door_symbols if b.line.distance(ShpPoint(*pxy)) < 1200.0
        ]
        if near_syms:
            sx, sy = min(near_syms, key=lambda pxy: b.line.distance(ShpPoint(*pxy)))
            anchor = ShpPoint(sx, sy)
        t = line.project(anchor, normalized=True)
        # Two rooms' walks can pick the same jamb (a corner where both boundaries end).
        # Two leaves in one reveal is always wrong: nudge along the wall until clear.
        pos = line.interpolate(max(0.1, min(0.9, t)), normalized=True)
        for o in openings:
            if not o.kind.is_door:
                continue
            ow = wall_map.get(o.wall_id)
            if ow is None:
                continue
            opt = LineString([ow.start, ow.end]).interpolate(o.t_center, normalized=True)
            if opt.distance(pos) < 700.0:
                shift = (700.0 + o.width_mm / 2.0) / max(line.length, 1.0)
                for cand in (t + shift, t - shift):
                    if 0.1 <= cand <= 0.9:
                        cpt = line.interpolate(cand, normalized=True)
                        if cpt.distance(opt) >= 650.0:
                            t = cand
                            break
                break
        added += 1
        openings.append(
            Opening(
                id=f"{prefix}-t{added}",
                wall_id=wall.id,
                kind=OpeningKind.DOOR,
                t_center=max(0.1, min(0.9, t)),
                width_mm=min(900.0, max(750.0, b.length * 0.5)),
                height_mm=cfg.default_door_height_mm,
                sill_mm=0.0,
                swing="left",
                source=Source.INFERRED,
                confidence=0.4,
                connects=(room.id, other_id),
            )
        )
        reachable.add(room.id)
        spread()
        needed = [r for r in needed if r.id not in reachable]

    # Balcony thresholds, from the topology rather than from the symbol. A balcony that
    # adjoins a habitable room is reached through a full-height sliding door — that is
    # what a balcony *is*. Deciding it here rather than by reclassifying whichever piece
    # of glazing line work happened to land nearby is more robust: the drawing renders a
    # sliding door as two parallel lines, indistinguishable from a window, so the rooms
    # either side are the only reliable evidence.
    # One threshold per room-and-balcony pair, on their longest shared run. Taking every
    # boundary fragment gave a single balcony four sub-metre sliding doors.
    pairs: dict[tuple[str, str], tuple[Any, Wall]] = {}
    for b, wall in links:
        if wall is None or b.space_b is None:
            continue
        a, c = by_room.get(room_of.get(b.space_a, "")), by_room.get(room_of.get(b.space_b, ""))
        if a is None or c is None:
            continue
        outside_room, inside_room = (a, c) if a.kind.is_outdoor else (c, a)
        if not outside_room.kind.is_outdoor or inside_room.kind.is_outdoor:
            continue
        if inside_room.kind in (RoomKind.DUCT, RoomKind.LIFT, RoomKind.STAIRS):
            continue
        key = (inside_room.id, outside_room.id)
        if key not in pairs or b.length > pairs[key][0].length:
            pairs[key] = (b, wall)

    slid = 0
    for (inside_id, outside_id), (b, wall) in pairs.items():
        inside_room, outside_room = by_room[inside_id], by_room[outside_id]
        line = LineString([wall.start, wall.end])
        # A door already at this threshold makes the synthesised one a duplicate —
        # measured by position against the boundary, on *any* wall. Testing only the
        # same wall id missed every drawn slider that attached to a parallel fragment
        # 76-450 mm away, so five balconies got two overlapping sliding doors each.
        occupied = False
        for o in openings:
            if not o.kind.is_door:
                continue
            ow = wall_map.get(o.wall_id)
            if ow is None:
                continue
            pt = LineString([ow.start, ow.end]).interpolate(o.t_center, normalized=True)
            if pt.distance(b.line) <= max(cfg.opening_serves_tol_mm, 900.0):
                occupied = True
                break
        if occupied:
            continue
        if b.length < cfg.min_gap_opening_mm * 1.6:
            continue
        slid += 1
        openings.append(
            Opening(
                id=f"{prefix}-b{slid}",
                wall_id=wall.id,
                kind=OpeningKind.SLIDING_DOOR,
                t_center=max(0.1, min(0.9, line.project(
                    b.line.interpolate(0.5, normalized=True), normalized=True
                ))),
                width_mm=min(b.length * 0.85, cfg.sliding_door_min_mm * 2.6),
                height_mm=cfg.balcony_door_height_mm,
                sill_mm=0.0,
                swing=None,
                source=Source.INFERRED,
                confidence=0.5,
                connects=(inside_room.id, outside_room.id),
            )
        )
    if slid:
        warnings.append(
            f"{slid} balcony threshold(s) placed as full-height sliding doors between the "
            "room and the outdoor space it opens onto"
        )

    if added:
        warnings.append(
            f"{added} doorway(s) were not drawn in the file and have been placed on the "
            "boundary each room shares with the space next to it, so every room is "
            "reachable — marked source=inferred, position is a reasoned placement "
            "rather than a measurement"
        )
    if needed:
        warnings.append(
            f"{len(needed)} room(s) could not be connected to the rest of the flat: "
            + ", ".join(sorted({r.label or r.id for r in needed})[:6])
        )
    return added


def _infer_missing_doorways(
    plan_rooms: Sequence[Room],
    walls: Sequence[Wall],
    openings: list[Opening],
    cfg: ExtractConfig,
    prefix: str,
    warnings: list[str],
) -> int:
    """Add a doorway to any room that has none, on its longest shared wall.

    A room you cannot enter is not a room. When a drawing neither cuts its wall faces
    at a doorway nor draws a door block — this one does both for most partitions, so
    only 12 openings were recovered for 18 rooms — the doorway is still *certainly*
    there, because the room is habitable and someone has to get into it. Leaving it out
    renders a sealed box you cannot walk through and cannot see into, which is a worse
    error than placing a standard door on the most plausible wall.

    So this is an architectural inference, not a measurement, and it is labelled as
    one: ``Source.INFERRED`` at low confidence, and every instance is reported in the
    plan's warnings. The choice is constrained rather than free — the door must go on a
    wall the room actually shares with a neighbour, centred on the overlap where both
    rooms touch it, so it always connects two real spaces.
    """
    from shapely.geometry import LineString as ShpLine

    rings = {r.id: ShpPolygon(r.polygon).exterior for r in plan_rooms if len(r.polygon) >= 3}
    shapes = {r.id: ShpPolygon(r.polygon) for r in plan_rooms if len(r.polygon) >= 3}
    by_id = {w.id: w for w in walls}

    def bounding(room: Room) -> list[Wall]:
        ring = rings.get(room.id)
        if ring is None:
            return []
        return [
            w
            for w in walls
            if w.kind not in ("parapet", "railing")
            and ShpLine([w.start, w.end]).distance(ring) < 300.0
        ]

    def serves(room: Room, opening: Opening) -> bool:
        """Does this opening let you *walk into* this room?

        Two conditions, and the second one is the whole point:

        1. The opening's own position must land on this room's stretch of the wall. A
           6 m wall can bound three rooms in series while its single door serves one.

        2. It must be a **door**. Counting any opening as access was the bug behind
           "no doors": a bedroom with a window was treated as reachable, so the
           inference pass skipped it, and the finished flat had five doors for
           eighteen rooms. You cannot enter a bedroom through its window.
        """
        if not opening.kind.is_door:
            return False
        # A resolved connects pair is the direct answer; geometry is the fallback for
        # openings that never got one.
        if opening.connects and room.id in opening.connects:
            return True
        wall = by_id.get(opening.wall_id)
        shape = shapes.get(room.id)
        if wall is None or shape is None:
            return False
        c = wall.point_at(opening.t_center)
        return shape.exterior.distance(ShpPoint(*c)) <= cfg.opening_serves_tol_mm

    added = 0
    for room in plan_rooms:
        # Ducts and lift shafts are not entered. Balconies get their threshold from the
        # outdoor reclassifier, which makes it a sliding door rather than a hinged one.
        if room.kind in (RoomKind.DUCT, RoomKind.LIFT, RoomKind.STAIRS) or room.kind.is_outdoor:
            continue
        mine = bounding(room)
        if any(serves(room, o) for o in openings):
            continue  # already reachable

        # Prefer a wall this room shares with another room, longest first: that is
        # both the most likely place for a door and the one that connects two spaces.
        best: tuple[float, Wall, Room] | None = None
        for w in mine:
            line = ShpLine([w.start, w.end])
            for other in plan_rooms:
                if other.id == room.id:
                    continue
                oring = rings.get(other.id)
                if oring is None or line.distance(oring) >= 300.0:
                    continue
                shared = min(
                    line.length,
                    ShpPolygon(room.polygon).intersection(line.buffer(400.0)).length or 0.0,
                )
                score = w.length_mm + shared
                if best is None or score > best[0]:
                    best = (score, w, other)
        if best is None:
            continue

        _, wall, other = best
        if wall.length_mm < cfg.min_gap_opening_mm * 1.4:
            continue
        # Centre the door on the stretch where both rooms meet the wall.
        line = ShpLine([wall.start, wall.end])
        a = line.project(ShpPolygon(room.polygon).centroid, normalized=True)
        b = line.project(ShpPolygon(other.polygon).centroid, normalized=True)
        t = max(0.12, min(0.88, (a + b) / 2.0))
        width = min(900.0, wall.length_mm * 0.6)
        openings.append(
            Opening(
                id=f"{prefix}-i{added + 1}",
                wall_id=wall.id,
                kind=OpeningKind.DOOR,
                t_center=t,
                width_mm=width,
                height_mm=cfg.default_door_height_mm,
                sill_mm=0.0,
                swing="left",
                source=Source.INFERRED,
                confidence=0.35,
                connects=(room.id, other.id),
            )
        )
        added += 1

    if added:
        warnings.append(
            f"{added} doorway(s) were not drawn in the file and have been inferred so "
            "every room is reachable — they are marked source=inferred at low "
            "confidence and their position is a best guess, not a measurement"
        )
    return added


def _reclassify_outdoor_openings(
    rooms: Sequence[Room], walls: Sequence[Wall], openings: list[Opening], cfg: ExtractConfig
) -> int:
    """Any opening between an inside room and a balcony becomes a sliding door.

    A bedroom does not have a 900 mm-sill *window* onto its own balcony — it has a
    full-height sliding or French door you walk through. Classifying those as windows
    (which is what the drawing's line work looks like, since a sliding door is drawn as
    parallel lines just like glazing) walls the balcony off behind a waist-high
    parapet, and the balcony then reads as inaccessible in every view.

    So the decision is made from the *rooms either side*, not from the symbol:
    interior on one side and outdoor on the other means a door, full height, no sill.
    """
    from shapely.geometry import LineString as SLine, Point as SPt, Polygon as SP

    shapes = [(r, SP(r.polygon)) for r in rooms if len(r.polygon) >= 3]
    by_id = {w.id: w for w in walls}
    changed = 0

    for o in openings:
        wall = by_id.get(o.wall_id)
        if wall is None:
            continue

        # Which rooms does this *wall* bound? Asked of the wall rather than of a probe
        # beside the opening, because a balcony's recovered polygon frequently covers
        # only part of the wall it sits on — the demo flat's 3.35 m patio opening has a
        # bedroom and an unlabelled fragment beside it, with the balcony further along
        # the same 12.3 m wall. A two-sided probe correctly finds no balcony there and
        # skips, which is how every balcony ended up sealed behind a parapet.
        line = SLine([wall.start, wall.end])
        near = [r for r, shp in shapes if line.distance(shp.exterior) < 450.0]
        outdoor = [r for r in near if r.kind.is_outdoor]
        # "Indoor" is anything not outdoor and not a void. Testing for *habitable or
        # circulation* was too narrow: the region beside the demo flat's patio opening
        # is an unlabelled fragment, so the indoor list came back empty and every
        # balcony stayed sealed. An unlabelled interior region is still interior.
        indoor = [
            r
            for r in near
            if not r.kind.is_outdoor
            and r.kind not in (RoomKind.DUCT, RoomKind.LIFT, RoomKind.STAIRS)
        ]
        if not outdoor or not indoor:
            continue

        # A threshold, not a window: either it is already door-height, or it is a wide
        # glazed opening, which at this width physically has to be something you walk
        # through.
        wide_glazed = o.width_mm >= cfg.sliding_door_min_mm and not o.kind.is_door
        if not (wide_glazed or o.kind.is_door):
            continue

        inside = max(indoor, key=lambda r: r.area_mm2)
        outside = max(outdoor, key=lambda r: r.area_mm2)
        o.kind = (
            OpeningKind.SLIDING_DOOR if o.width_mm >= cfg.sliding_door_min_mm else OpeningKind.DOOR
        )
        o.height_mm = cfg.balcony_door_height_mm
        o.sill_mm = 0.0
        o.swing = "none" if o.kind is OpeningKind.SLIDING_DOOR else "left"
        o.connects = (inside.id, outside.id)
        changed += 1
    return changed


_SOCIAL_OPEN = frozenset(
    {RoomKind.LIVING, RoomKind.DINING, RoomKind.LOBBY, RoomKind.CORRIDOR, RoomKind.ENTRANCE}
)


def _drop_selfsided_walls(
    walls: list[Wall],
    rooms: Sequence[Room],
    openings: Sequence[Opening],
    drawn_segments: Sequence[Any] = (),
) -> tuple[list[Wall], list[Wall]]:
    """Remove interior walls the final floors prove wrong.

    Two deterministic cases, both artefacts of the partition rather than the drawing:

    * A short stub with the same room on both sides — a 746 mm stub was left standing
      in a dining room by a boundary the sliver-heal later re-claimed on both sides.
      Capped at stub length: a *long* self-sided wall is a real partition the merged
      floor legitimately wraps around (a toilet's dividing wall), and dropping those
      unsealed four bathrooms.
    * A wall between two open-plan social spaces with no drawn line work along it —
      the invented lobby/living wall the user caught by eye, reproduced by remainder
      synthesis at up to 2.4 m. Wet rooms and bedrooms keep their unsupported walls
      by policy; between social spaces the drawing's silence is the statement.

    Walls carrying an opening are kept regardless, so no door is orphaned; exterior
    walls and guards are out of scope by definition.
    """
    used = {o.wall_id for o in openings}
    shapes = [(r, ShpPolygon(r.polygon)) for r in rooms if len(r.polygon) >= 3]
    drawn_u = (
        unary_union(
            [
                LineString([sg.a, sg.b])
                for sg in drawn_segments
                if math.dist(sg.a, sg.b) > 250.0
            ]
        )
        if drawn_segments
        else None
    )
    kept: list[Wall] = []
    dropped: list[Wall] = []
    for w in walls:
        if w.exterior or w.kind in ("parapet", "railing"):
            kept.append(w)
            continue
        line = LineString([w.start, w.end])
        mid = line.interpolate(0.5, normalized=True)
        nx, ny = w.normal()
        off = w.thickness_mm / 2.0 + 200.0
        sides: list[Room | None] = []
        for sgn in (1.0, -1.0):
            probe = ShpPoint(mid.x + sgn * nx * off, mid.y + sgn * ny * off)
            sides.append(next((r for r, q in shapes if q.contains(probe)), None))
        a, b = sides
        if (
            a is not None
            and b is not None
            and a.id == b.id
            and w.length_mm < 1200.0
            and w.id not in used
        ):
            dropped.append(w)
            continue
        if (
            a is not None
            and b is not None
            and a.id != b.id
            and a.kind in _SOCIAL_OPEN
            and b.kind in _SOCIAL_OPEN
        ):
            support = (
                drawn_u.intersection(line.buffer(150.0)).length if drawn_u is not None else 0.0
            )
            if support < w.length_mm * 0.25:
                # An opening does not defend this wall: hosted on an invention, it is
                # one too — a drawn door arc that anchored here was mis-anchored, and
                # between two open-plan spaces a doorway means nothing anyway.
                dropped.append(w)
                continue
        kept.append(w)
    return kept, dropped


def _reclaim_wall_bodies(dropped: Sequence[Wall], rooms: Sequence[Room]) -> None:
    """Give a dropped wall's body back to the floor beside it.

    Floors are cut around every wall body before the drop runs, so removing a wall
    leaves a wall-thick channel of no-man's floor between two rooms — 230 mm, which
    is wider than the 150 mm the audit accepts as "facing a room". The body strip
    merges into whichever adjacent room shares the longest frontier with it.
    """
    for w in dropped:
        body = LineString([w.start, w.end]).buffer(w.thickness_mm / 2.0, cap_style=2)
        best: tuple[float, Room, ShpPolygon] | None = None
        for r in rooms:
            if len(r.polygon) < 3:
                continue
            q = ShpPolygon(r.polygon)
            if q.distance(body) > 30.0:
                continue
            shared = q.buffer(40.0).intersection(body).area
            if shared > 0.0 and (best is None or shared > best[0]):
                best = (shared, r, q)
        if best is None:
            continue
        _, room, q = best
        merged = unary_union([q, body.buffer(10.0)])
        if isinstance(merged, MultiPolygon) and not merged.is_empty:
            mine = [g for g in merged.geoms if g.contains(q.representative_point())]
            merged = mine[0] if mine else max(merged.geoms, key=lambda g: g.area)
        if not isinstance(merged, ShpPolygon) or merged.is_empty or merged.interiors:
            continue
        room.polygon = [(float(x), float(y)) for x, y in merged.exterior.coords[:-1]]
        room.centroid = (float(merged.centroid.x), float(merged.centroid.y))


def _mark_exterior(walls: list[Wall], rooms: Sequence[Room]) -> None:
    """Flag walls with habitable space on one side only as exterior.

    Drives exterior finishes and tells the furnishing stage which walls can carry
    a window rather than a door.
    """
    shapes = [ShpPolygon(r.polygon) for r in rooms if len(r.polygon) >= 3]
    if not shapes:
        return
    interior = unary_union([s.buffer(1.0) for s in shapes])
    for w in walls:
        n = w.normal()
        mid = w.midpoint
        probe = w.thickness_mm / 2.0 + 150.0
        left = ShpPoint(mid[0] + n[0] * probe, mid[1] + n[1] * probe)
        right = ShpPoint(mid[0] - n[0] * probe, mid[1] - n[1] * probe)
        w.exterior = interior.contains(left) != interior.contains(right)
        # Keep `kind` in step. This pass is the authority on which walls face outside —
        # it probes the actual rooms — but it used to rewrite the flag and leave `kind`
        # behind, so seven walls ended up "structural" while flagged interior, or
        # "partition" while flagged exterior. The material assignment reads one and the
        # rest of the pipeline reads the other.
        if w.kind not in ("parapet", "railing"):
            w.kind = "structural" if w.exterior else "partition"


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def extract_plan(
    path: Path | str,
    *,
    config: ExtractConfig | None = None,
    work_dir: Path | str | None = None,
    max_units: int | None = None,
) -> PlanIR:
    """Extract a :class:`PlanIR` from a DWG or DXF file."""
    cfg = config or ExtractConfig()
    path = Path(path)
    work_dir = Path(work_dir) if work_dir else path.parent
    notes: list[str] = []
    warnings: list[str] = []

    if path.suffix.lower() == ".dwg":
        dxf_path, conv_notes = dwg_to_dxf(path, work_dir)
        notes.extend(conv_notes)
    else:
        dxf_path = path

    doc, load_notes = load_dxf(dxf_path, work_dir=work_dir)
    notes.extend(load_notes)

    raw_entities, fstats = flatten_layout(doc.modelspace())
    notes.append(fstats.summary())
    if fstats.errors:
        notes.append(f"flatten errors (first): {fstats.error_samples[:2]}")

    mm_per_unit, scale_why = _resolve_scale(doc, raw_entities)
    notes.append(f"scale: {scale_why}")
    entities = [_scale_entity(e, mm_per_unit) for e in raw_entities]

    # When the rule table cannot read this office's layer vocabulary, ask a
    # language model to map the names — names only, never geometry. Dormant on
    # drawings whose conventions the rules already know, and on machines with no
    # key; either way the notes say what happened.
    from .conventions import infer_layer_conventions

    overrides, conv_notes = infer_layer_conventions(entities)
    notes.extend(conv_notes)

    buckets = _bucket(entities, overrides)
    role_summary = ", ".join(
        f"{r.value}:{len(v)}" for r, v in sorted(buckets.by_role.items(), key=lambda kv: -len(kv[1]))[:10]
    )
    notes.append(f"layer roles: {role_summary}")
    if buckets.unknown_layers:
        top = sorted(buckets.unknown_layers.items(), key=lambda kv: -kv[1])[:5]
        notes.append(
            f"{len(buckets.unknown_layers)} unclassified layer(s), largest: "
            + ", ".join(f"{n}×{c}" for n, c in top)
        )

    _report_fixture_absence(buckets, notes)
    regions = _find_units(buckets, cfg, notes)
    if not regions:
        warnings.append(
            "No dwellings could be identified. The drawing may lack room labels and "
            "area boundaries, or may not be a floor plan."
        )
        return PlanIR(
            source_file=str(path),
            source_kind="dwg" if path.suffix.lower() == ".dwg" else "dxf",
            ingest_notes=notes,
            warnings=warnings,
        )

    # Largest first so `max_units` keeps the most substantial dwellings.
    regions.sort(key=lambda r: -r.polygon.area)
    if max_units is not None:
        if len(regions) > max_units:
            notes.append(f"limiting output to the {max_units} largest of {len(regions)} dwellings")
        regions = regions[:max_units]

    wall_ents = buckets.get(LayerRole.WALL)
    low_ents = buckets.get(LayerRole.WALL_LOW)
    col_ents = buckets.get(LayerRole.COLUMN)
    door_ents = buckets.get(LayerRole.DOOR)
    win_ents = buckets.get(LayerRole.WINDOW, LayerRole.WINDOW_SILL)

    plan = PlanIR(
        source_file=str(path),
        source_kind="dwg" if path.suffix.lower() == ".dwg" else "dxf",
        levels=[Level("L0", "Ground Floor", 0.0, cfg.wall_height_mm, cfg.slab_thickness_mm)],
        ingest_notes=notes,
        warnings=warnings,
    )
    imperial_votes = 0

    for ui, region in enumerate(regions, 1):
        prefix = f"u{ui}"
        clip = region.polygon.buffer(cfg.unit_clip_margin_mm)

        segs = _segments_from(wall_ents, clip)
        low_segs = _segments_from(low_ents, clip)
        cand, wreport = solve_walls(segs, SolverConfig())
        low_cand, _ = solve_walls(low_segs, SolverConfig(keep_unpaired_min_length=1200.0))

        walls: list[Wall] = []
        gap_specs: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
        for w in cand:
            wid = f"{prefix}-w{len(walls) + 1}"
            walls.append(
                Wall(
                    id=wid,
                    start=w.start,
                    end=w.end,
                    thickness_mm=w.thickness,
                    height_mm=cfg.wall_height_mm,
                    kind=w.kind,  # type: ignore[arg-type]
                    source=Source.CAD_VECTOR,
                    confidence=w.confidence,
                )
            )
            for t0, t1 in w.openings:
                # Absolute endpoints, not normalised positions. The final wall with
                # this id may be trimmed, snapped or re-extended by the boundary pass,
                # and a stale ``t`` on new geometry lands the opening on a stretch the
                # drawing draws as solid — two sliding doors and three windows were
                # painted onto unbroken wall bands this way.
                ax = w.start[0] + (w.end[0] - w.start[0]) * t0
                ay = w.start[1] + (w.end[1] - w.start[1]) * t0
                bx = w.start[0] + (w.end[0] - w.start[0]) * t1
                by = w.start[1] + (w.end[1] - w.start[1]) * t1
                gap_specs.append((wid, (ax, ay), (bx, by)))
        for w in low_cand:
            walls.append(
                Wall(
                    id=f"{prefix}-w{len(walls) + 1}",
                    start=w.start,
                    end=w.end,
                    thickness_mm=w.thickness,
                    height_mm=cfg.parapet_height_mm,
                    kind="parapet",
                    source=Source.CAD_VECTOR,
                    confidence=w.confidence,
                )
            )
        if ui == 1:
            plan.ingest_notes.append(f"{region.name}: {wreport.summary()}")

        # Promote drawn walls the solver could not pair into real walls.
        #
        # They were already excluded from the partition as wall *body*, which is why no
        # room contains one. But they were never emitted as walls, so the cells they
        # excluded had nothing built in them: every dwelling came back with unsealed rooms
        # and nine had rooms with no way in. A wall we can see but could not measure is
        # still a wall; it just gets a nominal thickness instead of a measured one.
        solved_union = (
            unary_union([LineString([w.start, w.end]) for w in walls if w.length_mm > 1e-6])
            if walls
            else None
        )
        promoted = 0
        for seg in segs:
            if math.dist(seg.a, seg.b) < cfg.raw_barrier_min_mm:
                continue
            line = LineString([seg.a, seg.b])
            if solved_union is not None and solved_union.distance(
                line.interpolate(0.5, normalized=True)
            ) <= cfg.raw_barrier_explained_mm:
                continue
            promoted += 1
            walls.append(
                Wall(
                    id=f"{prefix}-d{promoted}",
                    start=(float(seg.a[0]), float(seg.a[1])),
                    end=(float(seg.b[0]), float(seg.b[1])),
                    thickness_mm=cfg.interior_wall_mm,
                    height_mm=cfg.wall_height_mm,
                    kind="partition",
                    source=Source.CAD_VECTOR,
                    confidence=0.7,
                )
            )
        if promoted:
            plan.ingest_notes.append(
                f"{region.name}: {promoted} drawn wall(s) the solver could not pair were "
                "promoted to walls at nominal thickness"
            )

        # Rooms are bounded by full-height walls only; a 1 m parapet does not
        # enclose a balcony, and treating it as a room boundary would put a
        # ceiling over the terrace.
        full_height = [w for w in walls if w.kind not in ("parapet", "railing")]
        barriers = _enclosure_barriers(door_ents, win_ents, clip, cfg)

        # --- Stage 1: map the complete house ------------------------------------
        # The drawing marks its balconies itself — a hatch layer named for them. That is
        # direct evidence of which leftover regions are outdoor space, and it replaces the
        # building-edge heuristic wherever it exists: the heuristic had called a 182 sqft
        # interior region a balcony when only 10% of it carried the hatch.
        balcony_evidence = None
        bal_polys = [
            ShpPolygon(e.points)
            for e in entities
            if "BALCON" in e.layer.upper()
            and len(e.points) >= 3
            and ShpPolygon(e.points).is_valid
            and clip.intersects(ShpPolygon(e.points))
        ]
        if bal_polys:
            balcony_evidence = unary_union(bal_polys)
        smap = _map_house(
            region,
            walls,
            cfg,
            clip,
            barriers,
            plan.ingest_notes,
            plan.warnings,
            raw_wall_segments=segs,
            outdoor_evidence=balcony_evidence,
        )
        if smap is None:
            # Fall back to the old wall-first recovery rather than dropping the
            # dwelling: a plan with no usable labels still has walls worth showing.
            rooms = _extract_rooms(region, full_height, cfg, prefix, plan.warnings, barriers)
            plan.warnings.append(
                f"{region.name}: could not map spaces from labels; fell back to "
                "wall-derived rooms, which may leave unassigned regions"
            )
        else:
            # --- Stage 2: extrude walls from the mapped boundaries -------------
            # Evidence that a boundary is walled is more than the wall layer. A parapet
            # bounds a balcony, a window band sits in a wall, a door leaf hangs in one —
            # all drawn on their own layers. Testing against the wall layer alone called
            # balcony edges and doorway stretches "undrawn" and dropped real walls, which
            # unsealed half the drawing.
            evidence_segs = (
                list(segs)
                + list(low_segs)
                + _segments_from(door_ents, clip)
                + _segments_from(win_ents, clip)
            )
            walls, links = _walls_from_boundaries(
                smap, walls, cfg, prefix, plan.ingest_notes, drawn_segments=evidence_segs
            )
            gap_specs = [g for g in gap_specs if any(w.id == g[0] for w in walls)]
            # Floors follow the walls, so they meet them exactly.
            snapped_fp = _house_footprint(
                [w for w in walls if w.kind not in ("parapet", "railing")],
                region,
                clip,
                cfg,
                barriers,
            )
            rooms, space_to_room = _rooms_from_spaces(
                smap,
                walls,
                cfg,
                prefix,
                drawn_segments=segs,
                clip_footprint=snapped_fp,
                declared=region.polygon,
            )

        if not rooms:
            plan.warnings.append(f"{region.name}: no rooms recovered; skipping this dwelling")
            continue

        # --- Stage 3: openings, positioned on those walls ----------------------
        openings = _attach_openings(
            door_ents, win_ents, walls, clip, cfg, prefix, gap_specs
        )
        # Exterior faces first: the outdoor reclassifier needs to know which rooms are
        # balconies relative to the walls between them — and the self-sided drop must
        # run after it, because until then a stub can wear a stale exterior flag and
        # hide behind it.
        _mark_exterior(walls, rooms)
        walls, dropped_walls = _drop_selfsided_walls(walls, rooms, openings, segs)
        if dropped_walls:
            dropped_ids = {w.id for w in dropped_walls}
            openings = [o for o in openings if o.wall_id not in dropped_ids]
            _reclaim_wall_bodies(dropped_walls, rooms)
        slid = _reclassify_outdoor_openings(rooms, walls, openings, cfg)
        if slid:
            plan.ingest_notes.append(
                f"{region.name}: {slid} threshold(s) onto outdoor space reclassified as "
                "full-height sliding/patio doors"
            )
        # Doors last, so they can see which rooms the sliding thresholds already serve.
        if smap is not None:
            _doors_from_topology(
                smap,
                links,
                space_to_room,
                rooms,
                walls,
                openings,
                cfg,
                prefix,
                plan.warnings,
                door_symbols=[e.points[0] for e in door_ents if e.points],
            )
        else:
            _infer_missing_doorways(rooms, walls, openings, cfg, prefix, plan.warnings)
        fixtures = _extract_fixtures(buckets, clip, rooms, prefix)

        for r in rooms:
            if r.stated_dimension and r.stated_dimension.imperial:
                imperial_votes += 1
            agree = r.dimension_agreement()
            if agree is not None and agree < cfg.dimension_agreement_warn:
                plan.warnings.append(
                    f"{region.name}/{r.label or r.id}: measured {r.area_sqft:.0f} sqft vs printed "
                    f"{r.stated_dimension.area_sqft:.0f} sqft "  # type: ignore[union-attr]
                    f"({agree:.0%} agreement) — check wall recovery for this room"
                )

        measured_sqft = sum(r.area_sqft for r in rooms)
        if region.stated_area_sqft:
            ratio = min(measured_sqft, region.stated_area_sqft) / max(
                measured_sqft, region.stated_area_sqft
            )
            if ratio < 0.7:
                plan.warnings.append(
                    f"{region.name}: rooms total {measured_sqft:.0f} sqft against a declared "
                    f"{region.stated_area_sqft:.0f} sqft ({ratio:.0%}) — extraction likely incomplete"
                )

        plan.walls.extend(walls)
        plan.rooms.extend(rooms)
        plan.openings.extend(openings)
        plan.fixtures.extend(fixtures)
        b = region.polygon.bounds
        plan.units.append(
            PlanUnit(
                id=prefix,
                name=region.name,
                room_ids=[r.id for r in rooms],
                wall_ids=[w.id for w in walls],
                stated_area_sqft=region.stated_area_sqft,
                boundary=[
                    (float(x), float(y)) for x, y in region.polygon.exterior.coords[:-1]
                ],
                bounds=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                confidence=0.9 if region.source == "area_boundary" else 0.6,
            )
        )

    plan.imperial_display = imperial_votes > len(plan.rooms) * 0.4
    plan.ingest_notes.append(
        f"extracted {len(plan.units)} dwelling(s): {len(plan.rooms)} rooms, "
        f"{len(plan.walls)} walls, {len(plan.openings)} openings, {len(plan.fixtures)} fixtures"
    )
    return plan
