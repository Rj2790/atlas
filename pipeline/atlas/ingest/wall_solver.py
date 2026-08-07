"""Recover wall centrelines and thicknesses from CAD line work.

A draughtsman does not draw walls. They draw the two *faces* of each wall as
separate lines, and the wall itself exists only in the reader's head. The demo
file's wall layer is 141 bare ``LINE`` entities with no grouping, no thickness
attribute, and no indication of which line pairs with which.

To build 3D we need the opposite representation: a centreline with a thickness,
because only that joins cleanly at corners and T-junctions and only that lets an
opening be positioned along a wall. So the job is to invert the draughting:

1. **Cluster by direction.** Two lines can only be faces of one wall if they are
   parallel. Angles are taken mod 180° (a line has no direction) and clustered
   with a tolerance, which also recovers non-orthogonal walls — the reason we
   cluster rather than assume the Manhattan grid that most tools hard-code.

2. **Rotate into wall-local coordinates.** Within a direction cluster, rotating so
   the walls run horizontal turns "parallel and a plausible distance apart" into
   the 1D question "do these two ``v`` offsets differ by a wall thickness?", and
   "do they overlap?" into a 1D interval intersection. Much easier to get right
   than general 2D geometry.

3. **Pair opposing faces.** Candidate pairs are scored and consumed greedily,
   thinnest first. Thinnest-first matters: where a 150 mm partition abuts a 230 mm
   structural wall, three parallel rails exist and the greedy choice must not
   marry the outer two into one 380 mm wall that does not exist.

4. **Merge and heal.** Collinear fragments of the same wall are merged, and
   endpoints near a neighbouring wall are snapped and extended to meet it, so
   corners close instead of leaving a hairline gap that would later leak light
   into the render.

Everything is millimetres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["Segment", "WallCandidate", "SolverConfig", "SolveReport", "solve_walls"]

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Segment:
    """A straight piece of line work from the drawing."""

    a: Point
    b: Point
    layer: str = ""

    @property
    def length(self) -> float:
        return math.dist(self.a, self.b)

    @property
    def angle_mod180(self) -> float:
        """Orientation in degrees, folded to [0, 180) since a line is undirected."""
        return math.degrees(math.atan2(self.b[1] - self.a[1], self.b[0] - self.a[0])) % 180.0


@dataclass(slots=True)
class WallCandidate:
    """A recovered wall: centreline, thickness, and how much to trust it."""

    start: Point
    end: Point
    thickness: float
    confidence: float
    """1.0 for a clean face pair; lower when inferred from a single face."""

    kind: str = "partition"
    layers: tuple[str, ...] = ()
    paired: bool = True
    """False when derived from an unpaired face using an assumed thickness."""

    openings: tuple[tuple[float, float], ...] = ()
    """Normalised ``(t0, t1)`` spans along this wall where *both* faces were absent.

    This is the drawing telling us where its openings are by omission. Most doorways
    in a real plan carry no door symbol at all — the demo file has three door blocks
    for a sixteen-room flat — yet every doorway is cut through both wall faces, and
    the solver already locates those cuts in order to bridge them. Reporting them
    instead of discarding them recovers openings that nothing else can see.

    Both faces must be absent. A gap in one face only is a recess, a service chase
    or a change of finish, not a hole through the wall."""

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)


@dataclass(slots=True)
class SolverConfig:
    """Tolerances. Defaults suit residential drawings in millimetres."""

    angle_tol_deg: float = 2.0
    """Two faces this close in orientation are treated as parallel."""

    min_segment_length: float = 60.0
    """Shorter line work is a wall *end cap*, a hatch tick or a symbol, not a face."""

    min_thickness: float = 60.0
    max_thickness: float = 500.0
    """Plausible wall thickness band. 100/150/230 mm are the common Indian sizes;
    the band is wider to admit cavity walls and clad columns."""

    min_overlap: float = 250.0
    """Two faces must run alongside each other at least this far to be one wall.
    Below this they are probably crossing at a junction."""

    rail_merge_tol: float = 12.0
    """Faces within this perpendicular distance are the same physical rail."""

    collinear_merge_gap: float = 1400.0
    """Bridge collinear fragments separated by up to this much.

    Draughting cuts the face lines where a door or window sits, so one physical
    wall arrives as two or three fragments with a hole between them. A wall with a
    door in it is still *one wall with an opening*, and modelling it that way is
    what lets the opening be cut properly later. Sized to span the widest single
    leaf plus frame (a 1200 mm door reads as a ~1300 mm gap); a genuinely separate
    wall starting within 1.4 m along the same line, at the same thickness, is a
    cased opening in practice and is better treated as one wall too.

    This value is load-bearing for room detection: too small and rooms leak into
    each other through every doorway, which is the failure mode that makes naive
    floor-plan pipelines produce one giant undivided space."""

    junction_snap: float = 180.0
    """Snap and extend a wall end this far to close a corner."""

    structural_thickness: float = 200.0
    """At or above this, classify the wall as structural rather than partition."""

    min_opening_width: float = 550.0
    """Narrowest gap through both wall faces to treat as an opening. Below a normal
    door leaf; above the drafting slop that produces spurious hairline gaps."""

    single_face_thickness: float = 150.0
    """Assumed thickness when a face has no partner but is long enough to matter."""

    keep_unpaired_min_length: float = 1500.0
    """Only rescue unpaired faces at least this long; shorter ones are noise."""


@dataclass
class SolveReport:
    input_segments: int = 0
    used_segments: int = 0
    direction_clusters: int = 0
    paired_walls: int = 0
    unpaired_walls: int = 0
    recovered_openings: int = 0
    merged: int = 0
    snapped_ends: int = 0
    thickness_histogram: dict[int, int] = field(default_factory=dict)

    def summary(self) -> str:
        hist = ", ".join(
            f"{t}mm×{n}" for t, n in sorted(self.thickness_histogram.items(), key=lambda kv: -kv[1])[:6]
        )
        return (
            f"walls: {self.paired_walls} paired + {self.unpaired_walls} single-face "
            f"from {self.input_segments} segments in {self.direction_clusters} directions "
            f"({self.merged} merged, {self.snapped_ends} ends snapped, "
            f"{self.recovered_openings} openings from face gaps) [{hist}]"
        )


def _cluster_directions(segs: list[Segment], tol: float) -> list[list[Segment]]:
    """Group segments by orientation, wrapping across the 0/180 seam.

    Wrapping matters: a wall at 179.5° and one at 0.5° are parallel, and a naive
    sort-and-split would put them in different clusters and never pair them.
    """
    if not segs:
        return []
    ordered = sorted(segs, key=lambda s: s.angle_mod180)
    clusters: list[list[Segment]] = [[ordered[0]]]
    for s in ordered[1:]:
        if abs(s.angle_mod180 - clusters[-1][-1].angle_mod180) <= tol:
            clusters[-1].append(s)
        else:
            clusters.append([s])
    # Stitch the seam: the last cluster may continue into the first.
    if len(clusters) > 1:
        first, last = clusters[0], clusters[-1]
        if (first[0].angle_mod180 + 180.0) - last[-1].angle_mod180 <= tol:
            clusters[0] = last + first
            clusters.pop()
    return clusters


def _cluster_angle(cluster: list[Segment]) -> float:
    """Length-weighted mean orientation, in radians.

    Weighting by length lets a long true wall face set the direction instead of
    being dragged off by a short, slightly-off fragment.
    """
    sx = sy = 0.0
    for s in cluster:
        # Double the angle so opposite directions reinforce rather than cancel.
        a = 2.0 * math.radians(s.angle_mod180)
        sx += s.length * math.cos(a)
        sy += s.length * math.sin(a)
    if sx == 0.0 and sy == 0.0:
        return math.radians(cluster[0].angle_mod180)
    return math.atan2(sy, sx) / 2.0


@dataclass(slots=True)
class _Rail:
    """One face line in wall-local coordinates: a ``v`` offset and a ``u`` span."""

    v: float
    u0: float
    u1: float
    layers: set[str] = field(default_factory=set)
    weight: float = 0.0
    gaps: list[tuple[float, float]] = field(default_factory=list)
    """``u`` intervals where this face line was absent but was bridged."""

    def overlap(self, other: "_Rail") -> tuple[float, float] | None:
        lo, hi = max(self.u0, other.u0), min(self.u1, other.u1)
        return (lo, hi) if hi > lo else None


def _build_rails(cluster: list[Segment], theta: float, cfg: SolverConfig) -> list[_Rail]:
    """Project a direction cluster into local ``(u, v)`` and merge coincident faces."""
    cos_t, sin_t = math.cos(-theta), math.sin(-theta)
    raw: list[_Rail] = []
    for s in cluster:
        ua = s.a[0] * cos_t - s.a[1] * sin_t
        va = s.a[0] * sin_t + s.a[1] * cos_t
        ub = s.b[0] * cos_t - s.b[1] * sin_t
        vb = s.b[0] * sin_t + s.b[1] * cos_t
        raw.append(
            _Rail(
                v=(va + vb) / 2.0,
                u0=min(ua, ub),
                u1=max(ua, ub),
                layers={s.layer} if s.layer else set(),
                weight=s.length,
            )
        )

    # Merge faces at the same offset into one rail, unioning their u-spans only
    # where they actually touch, so two distant walls on one line stay separate.
    raw.sort(key=lambda r: (r.v, r.u0))
    rails: list[_Rail] = []
    for r in raw:
        placed = False
        for q in rails:
            if abs(q.v - r.v) <= cfg.rail_merge_tol and r.u0 <= q.u1 + cfg.collinear_merge_gap:
                total = q.weight + r.weight
                q.v = (q.v * q.weight + r.v * r.weight) / total if total else q.v
                # A genuine hole in this face, about to be bridged. Record it before
                # the spans are unioned and the evidence is gone.
                if r.u0 > q.u1 + 1.0:
                    q.gaps.append((q.u1, r.u0))
                q.u0, q.u1 = min(q.u0, r.u0), max(q.u1, r.u1)
                q.layers |= r.layers
                q.gaps.extend(r.gaps)
                q.weight = total
                placed = True
                break
        if not placed:
            rails.append(r)
    return rails


def _to_world(u: float, v: float, theta: float) -> Point:
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return (u * cos_t - v * sin_t, u * sin_t + v * cos_t)


def _pair_rails(rails: list[_Rail], theta: float, cfg: SolverConfig) -> list[WallCandidate]:
    """Marry opposing faces into walls, thinnest plausible pair first."""
    candidates: list[tuple[float, _Rail, _Rail, tuple[float, float]]] = []
    for i in range(len(rails)):
        for j in range(i + 1, len(rails)):
            ri, rj = rails[i], rails[j]
            thickness = abs(ri.v - rj.v)
            if not (cfg.min_thickness <= thickness <= cfg.max_thickness):
                continue
            ov = ri.overlap(rj)
            if ov is None or (ov[1] - ov[0]) < cfg.min_overlap:
                continue
            candidates.append((thickness, ri, rj, ov))

    # Thinnest first: with three parallel rails, the true wall is the near pair.
    # Longer overlap breaks ties, favouring the more strongly evidenced wall.
    candidates.sort(key=lambda c: (c[0], -(c[3][1] - c[3][0])))

    consumed: dict[int, list[tuple[float, float]]] = {}

    def free_span(rail: _Rail, span: tuple[float, float]) -> tuple[float, float] | None:
        """Trim ``span`` by whatever of ``rail`` is already used by a thinner wall."""
        lo, hi = span
        for c0, c1 in consumed.get(id(rail), []):
            if c0 <= lo and c1 >= hi:
                return None
            if c0 <= lo < c1:
                lo = c1
            elif c0 < hi <= c1:
                hi = c0
        return (lo, hi) if hi - lo >= cfg.min_overlap else None

    walls: list[WallCandidate] = []
    for thickness, ri, rj, ov in candidates:
        span = free_span(ri, ov)
        if span is None:
            continue
        span = free_span(rj, span)
        if span is None:
            continue
        consumed.setdefault(id(ri), []).append(span)
        consumed.setdefault(id(rj), []).append(span)

        v_mid = (ri.v + rj.v) / 2.0
        length = span[1] - span[0]
        openings: list[tuple[float, float]] = []
        if length > 1e-6:
            for g0, g1 in ri.gaps:
                for h0, h1 in rj.gaps:
                    lo, hi = max(g0, h0, span[0]), min(g1, h1, span[1])
                    if hi - lo >= cfg.min_opening_width:
                        openings.append(((lo - span[0]) / length, (hi - span[0]) / length))
        walls.append(
            WallCandidate(
                start=_to_world(span[0], v_mid, theta),
                end=_to_world(span[1], v_mid, theta),
                thickness=thickness,
                confidence=1.0,
                kind="structural" if thickness >= cfg.structural_thickness else "partition",
                layers=tuple(sorted(ri.layers | rj.layers)),
                paired=True,
                openings=tuple(_merge_spans(openings)),
            )
        )

    # Rescue long faces that never found a partner — commonly the outer face of a
    # party wall whose inner face belongs to the neighbouring unit and so was
    # clipped away. Better a wall of assumed thickness than a hole in the model,
    # but flagged with lower confidence and paired=False so the UI can say so.
    for rail in rails:
        used = consumed.get(id(rail), [])
        covered = sum(hi - lo for lo, hi in used)
        span_len = rail.u1 - rail.u0
        if span_len < cfg.keep_unpaired_min_length or covered > 0.5 * span_len:
            continue
        t = cfg.single_face_thickness
        walls.append(
            WallCandidate(
                start=_to_world(rail.u0, rail.v, theta),
                end=_to_world(rail.u1, rail.v, theta),
                thickness=t,
                confidence=0.45,
                kind="partition",
                layers=tuple(sorted(rail.layers)),
                paired=False,
            )
        )
    return walls


def _merge_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union overlapping normalised spans."""
    if not spans:
        return []
    spans = sorted(spans)
    out = [spans[0]]
    for lo, hi in spans[1:]:
        if lo <= out[-1][1] + 1e-6:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _merge_collinear(walls: list[WallCandidate], cfg: SolverConfig) -> tuple[list[WallCandidate], int]:
    """Join collinear, same-thickness walls separated by a small gap.

    Face lines are cut where a door sits, which fragments one wall into two. The
    3D build wants the whole wall with the door as an *opening in it*, not two
    stubs, so we re-join and let the opening stage cut the hole properly.
    """
    merged = 0
    out = list(walls)
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                if abs(a.thickness - b.thickness) > cfg.rail_merge_tol:
                    continue
                ang = abs((math.degrees(math.atan2(a.end[1] - a.start[1], a.end[0] - a.start[0])) % 180.0)
                          - (math.degrees(math.atan2(b.end[1] - b.start[1], b.end[0] - b.start[0])) % 180.0))
                if min(ang, 180.0 - ang) > cfg.angle_tol_deg:
                    continue
                # Perpendicular offset between the two centrelines must be ~0.
                dx, dy = a.end[0] - a.start[0], a.end[1] - a.start[1]
                n = math.hypot(dx, dy)
                if n < 1e-9:
                    continue
                nx, ny = -dy / n, dx / n
                perp = abs((b.start[0] - a.start[0]) * nx + (b.start[1] - a.start[1]) * ny)
                if perp > cfg.rail_merge_tol:
                    continue
                # Nearest endpoints must be within the bridging gap.
                pairs = [
                    (math.dist(a.end, b.start), a.start, b.end),
                    (math.dist(a.start, b.end), b.start, a.end),
                    (math.dist(a.end, b.end), a.start, b.start),
                    (math.dist(a.start, b.start), a.end, b.end),
                ]
                gap, s, e = min(pairs, key=lambda p: p[0])
                if gap > cfg.collinear_merge_gap:
                    continue
                # Re-normalise each wall's openings onto the merged centreline, or
                # they would silently slide to the wrong place along the new span.
                new_len = math.dist(s, e) or 1.0
                merged_ops: list[tuple[float, float]] = []
                for w, in ((a,), (b,)):
                    wl = w.length
                    for t0, t1 in w.openings:
                        p0 = (w.start[0] + (w.end[0] - w.start[0]) * t0,
                              w.start[1] + (w.end[1] - w.start[1]) * t0)
                        p1 = (w.start[0] + (w.end[0] - w.start[0]) * t1,
                              w.start[1] + (w.end[1] - w.start[1]) * t1)
                        merged_ops.append(
                            (math.dist(s, p0) / new_len, math.dist(s, p1) / new_len)
                        )
                out[i] = WallCandidate(
                    start=s,
                    end=e,
                    thickness=(a.thickness + b.thickness) / 2.0,
                    confidence=min(a.confidence, b.confidence),
                    kind=a.kind if a.length >= b.length else b.kind,
                    layers=tuple(sorted(set(a.layers) | set(b.layers))),
                    paired=a.paired and b.paired,
                    openings=tuple(_merge_spans(merged_ops)),
                )
                out.pop(j)
                merged += 1
                changed = True
                break
            if changed:
                break
    return out, merged


def _snap_junctions(walls: list[WallCandidate], cfg: SolverConfig) -> int:
    """Pull near-coincident wall ends onto a shared point so corners close.

    Averaging the cluster (rather than snapping to whichever wall came first)
    avoids biasing the geometry by input order.
    """
    ends: list[tuple[int, str]] = []
    for i, w in enumerate(walls):
        ends.append((i, "start"))
        ends.append((i, "end"))

    def pos(ref: tuple[int, str]) -> Point:
        w = walls[ref[0]]
        return w.start if ref[1] == "start" else w.end

    snapped = 0
    visited: set[int] = set()
    for idx, ref in enumerate(ends):
        if idx in visited:
            continue
        group = [idx]
        p = pos(ref)
        for jdx in range(idx + 1, len(ends)):
            if jdx in visited or ends[jdx][0] == ref[0]:
                continue
            if math.dist(p, pos(ends[jdx])) <= cfg.junction_snap:
                group.append(jdx)
        if len(group) < 2:
            continue
        cx = sum(pos(ends[g])[0] for g in group) / len(group)
        cy = sum(pos(ends[g])[1] for g in group) / len(group)
        for g in group:
            visited.add(g)
            wi, which = ends[g]
            w = walls[wi]
            if which == "start":
                w.start = (cx, cy)
            else:
                w.end = (cx, cy)
            snapped += 1
    return snapped


def solve_walls(
    segments: list[Segment], cfg: SolverConfig | None = None
) -> tuple[list[WallCandidate], SolveReport]:
    """Recover walls from face line work. See module docstring for the method."""
    cfg = cfg or SolverConfig()
    report = SolveReport(input_segments=len(segments))

    usable = [s for s in segments if s.length >= cfg.min_segment_length]
    report.used_segments = len(usable)
    if not usable:
        return [], report

    clusters = _cluster_directions(usable, cfg.angle_tol_deg)
    report.direction_clusters = len(clusters)

    walls: list[WallCandidate] = []
    for cluster in clusters:
        theta = _cluster_angle(cluster)
        rails = _build_rails(cluster, theta, cfg)
        walls.extend(_pair_rails(rails, theta, cfg))

    walls, merged = _merge_collinear(walls, cfg)
    report.merged = merged
    report.snapped_ends = _snap_junctions(walls, cfg)

    walls = [w for w in walls if w.length >= cfg.min_overlap]
    report.paired_walls = sum(1 for w in walls if w.paired)
    report.unpaired_walls = sum(1 for w in walls if not w.paired)
    for w in walls:
        bucket = int(round(w.thickness / 10.0) * 10)
        report.thickness_histogram[bucket] = report.thickness_histogram.get(bucket, 0) + 1
        report.recovered_openings += len(w.openings)
    return walls, report
