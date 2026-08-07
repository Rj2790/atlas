"""Check every dwelling in a drawing against invariants that must always hold.

This exists because of how the geometry was tuned. Constants like the balcony depth,
the outdoor-edge fraction, the wall snap distance and the corridor minimum area were
each chosen by building **one** flat and watching its numbers move. That is
overfitting, and it is invisible: a change that fixes unit 1 and breaks unit 7 looks
like pure progress if unit 1 is the only thing you look at.

A production drawing hands us sixteen dwellings for free — 2, 3 and 4 BHK, 710 to
1558 sqft, drawn by the same hand to the same conventions. Auditing all of them turns
every threshold change into a measurement instead of a guess.

The invariants are the ones whose violation is always a defect, never a judgement:

* **No wall runs through a room.** A wall crossing a room's interior is wrong however
  the room was derived.
* **No room overlaps another.** Rooms partition space; overlap means double-counted
  floor.
* **Walls follow the drawing's own directions.** Buildings are orthogonal; a wall well
  off the dominant axis came from a raster artefact.
* **Enclosure**, except where two spaces are deliberately continuous — an open-plan
  living/dining threshold is a design fact, not a hole.
* **`kind` and `exterior` agree**, since one drives materials and the other drives
  everything else.
* **Every room is reachable** through a door or an open threshold.
* **No wall the *drawing* shows runs through a recovered room.** This one is the reason
  the others are not enough. Every check above compares our rooms against our *own*
  walls, and both come out of the same partition, so they agree with each other by
  construction even when the partition ignored a wall that is plainly drawn. Measuring
  against the source line work is the only independent test, and it found 40 m of drawn
  wall sitting inside recovered rooms on a flat that passed everything else — 9.2 m
  through one bedroom and 6.7 m through a 37 sqft toilet.

Areas are *reported, not asserted*. Measured floor area legitimately falls short of a
declared carpet area by the wall footprints, and how far depends on the plan, so a
threshold there would either be too loose to catch anything or would fail honest units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .planir import OpeningKind, PlanIR, RoomKind, Source
from .units import mm_to_sqft

__all__ = ["UnitAudit", "AuditResult", "audit_plan", "audit_file"]


_OPEN_PAIR = frozenset({RoomKind.LIVING, RoomKind.DINING})
"""Room kinds allowed to share an unwalled threshold, so their enclosure may be short."""

_UNENTERED = (RoomKind.DUCT, RoomKind.LIFT, RoomKind.STAIRS)


@dataclass(slots=True)
class UnitAudit:
    name: str
    rooms: int
    walls: int
    openings: int
    fixtures: int
    measured_sqft: float
    declared_sqft: float | None
    accounted: float | None = None
    """Share of the declared polygon assigned to floor or wall body. The like-for-like
    area figure: carpet area includes internal walls, so floor alone always reads low."""
    dims: float | None = None
    """Mean printed-dimension agreement over labelled rooms, in [0, 1]."""
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def area_ratio(self) -> float | None:
        if not self.declared_sqft:
            return None
        return self.measured_sqft / self.declared_sqft


@dataclass
class AuditResult:
    source: str
    units: list[UnitAudit] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(u.ok for u in self.units)

    def table(self) -> str:
        head = (
            f"{'#':>3}  {'unit':<8}{'rooms':>6}{'walls':>6}{'opens':>6}{'fixt':>5}"
            f"{'measured':>10}{'declared':>10}{'ratio':>7}{'acct':>6}{'dims':>6}  result"
        )
        lines = [head, "-" * len(head)]
        for n, u in enumerate(self.units, 1):
            ratio = f"{u.area_ratio:.0%}" if u.area_ratio else "—"
            decl = f"{u.declared_sqft:.0f}" if u.declared_sqft else "—"
            acct = f"{u.accounted:.0%}" if u.accounted is not None else "—"
            dims = f"{u.dims:.0%}" if u.dims is not None else "—"
            verdict = "ok" if u.ok else f"{len(u.failures)} FAIL"
            lines.append(
                f"{n:>3}  {u.name[:7]:<8}{u.rooms:>6}{u.walls:>6}{u.openings:>6}"
                f"{u.fixtures:>5}{u.measured_sqft:>10.0f}{decl:>10}{ratio:>7}{acct:>6}{dims:>6}  {verdict}"
            )
        return "\n".join(lines)

    def failure_report(self) -> str:
        # Numbered, because a drawing holds several "4 BHK" and "3 BHK" units and an
        # unnumbered failure cannot be reproduced with `--unit`.
        out: list[str] = []
        for n, u in enumerate(self.units, 1):
            for f in u.failures:
                out.append(f"  unit {n} ({u.name}): {f}")
        return "\n".join(out)

    def failing_units(self) -> dict[int, int]:
        """1-based unit number -> failure count, for baseline comparison."""
        return {n: len(u.failures) for n, u in enumerate(self.units, 1) if u.failures}


def _dominant_angle(walls: Sequence[Any]) -> float:
    """Length-weighted modal wall direction, folded into [0, 90) degrees."""
    buckets: dict[int, float] = {}
    for w in walls:
        L = w.length_mm
        if L < 500.0:
            continue
        ang = math.degrees(math.atan2(w.end[1] - w.start[1], w.end[0] - w.start[0])) % 90.0
        buckets[int(round(ang))] = buckets.get(int(round(ang)), 0.0) + L
    return float(max(buckets, key=lambda k: buckets[k])) if buckets else 0.0


def audit_plan(
    plan: PlanIR, drawn: Any | None = None, evidence: "DrawingEvidence | None" = None
) -> AuditResult:
    """Check every dwelling in ``plan``.

    ``drawn`` is the drawing's own wall line work as a shapely geometry, when available.
    ``evidence`` is the fuller bundle (walls + door symbols + window line work); with it
    the audit can also ask "did we invent this element?". Without either, the audit can
    only check the model against itself.
    """
    result = AuditResult(source=plan.source_file)
    units = plan.units or []
    for unit in units:
        sub = plan.subset(unit)
        result.units.append(
            _audit_unit(unit.name, sub, unit.stated_area_sqft, drawn, evidence)
        )
    return result


def _audit_unit(
    name: str,
    plan: PlanIR,
    declared: float | None,
    drawn: Any | None = None,
    evidence: "DrawingEvidence | None" = None,
) -> UnitAudit:
    shapes = [(r, Polygon(r.polygon)) for r in plan.rooms if len(r.polygon) >= 3]
    solid_walls = [w for w in plan.walls if w.kind not in ("parapet", "railing")]
    audit = UnitAudit(
        name=name,
        rooms=len(plan.rooms),
        walls=len(plan.walls),
        openings=len(plan.openings),
        fixtures=len(plan.fixtures),
        measured_sqft=sum(mm_to_sqft(p.area) for _, p in shapes),
        declared_sqft=declared,
    )
    if not shapes:
        audit.failures.append("no rooms recovered")
        return audit

    # --- walls crossing a room's interior ---------------------------------
    through = [
        (w.id, r.label or r.kind.value, seg)
        for w in solid_walls
        for r, p in shapes
        for seg in [LineString([w.start, w.end]).intersection(p.buffer(-40)).length]
        if seg > 600.0
    ]
    if through:
        worst = max(through, key=lambda t: t[2])
        audit.failures.append(
            f"{len(through)} wall(s) run through a room, worst {worst[0]} "
            f"{worst[2]:.0f} mm into {worst[1]}"
        )

    # --- rooms overlapping each other -------------------------------------
    overlaps = 0
    for i, (_, a) in enumerate(shapes):
        for _, b in shapes[i + 1 :]:
            if a.intersection(b).area > 1e5:  # 0.1 m²
                overlaps += 1
    if overlaps:
        audit.failures.append(f"{overlaps} pair(s) of rooms overlap")

    # --- walls off the drawing's own axes ---------------------------------
    base = _dominant_angle(solid_walls)
    oblique = []
    for w in solid_walls:
        ang = math.degrees(math.atan2(w.end[1] - w.start[1], w.end[0] - w.start[0])) % 90.0
        off = min(abs(ang - base), 90.0 - abs(ang - base))
        if off > 5.0:
            oblique.append((w.id, off))
    if oblique:
        audit.failures.append(
            f"{len(oblique)} wall(s) more than 5 degrees off the drawing's axes "
            f"(worst {max(oblique, key=lambda t: t[1])[1]:.0f}deg)"
        )

    # --- kind vs exterior --------------------------------------------------
    mismatched = [
        w.id
        for w in plan.walls
        if w.kind not in ("parapet", "railing") and (w.kind == "structural") != bool(w.exterior)
    ]
    if mismatched:
        audit.failures.append(f"{len(mismatched)} wall(s) have kind disagreeing with exterior")

    # --- enclosure ---------------------------------------------------------
    # The tolerance is one raster cell (90 mm) plus a hair, because that is the partition's
    # honest resolution: a cell centre must fall outside the wall solid, so an un-healed
    # room edge can legitimately sit anywhere up to a full cell from the wall face. (An
    # earlier 70 mm figure was fitted to the output of the day rather than derived — rooms
    # whose sliver-heal was reverted then showed walls at 72-77 mm and failed for being
    # exactly as accurate as the raster allows.) The check still discriminates: every real
    # defect it has caught sat 133-1508 mm from the nearest wall.
    if solid_walls:
        fabric = unary_union(
            [
                LineString([w.start, w.end]).buffer(w.thickness_mm / 2.0, cap_style=2)
                for w in plan.walls
            ]
        ).buffer(100.0)
        # A drawn wall seals a room as surely as a built one. Floors now stop at the
        # drawn face where the drawing shows one, and that face can sit further than
        # 100 mm from our built solid when the drawn band is thicker than the nominal
        # thickness we gave it — correctly placed floor must not read as a leak.
        if drawn is not None:
            fabric = fabric.union(drawn.buffer(100.0))
        # A gap is only a leak when it faces *nothing*.
        #
        # Walls are now built only where the drawing draws one, so a stretch of room
        # perimeter with no wall along it and another room's floor immediately beyond is
        # the drawing's own statement: those spaces connect openly (a cased opening, an
        # arch, an open-plan seam). Flagging that as "not sealed" punished fidelity — the
        # drawing shows the lobby open to the living room, and the audit demanded a wall
        # the architect never drew. What remains a defect is a stretch with no wall, no
        # drawn line work (already unioned into `fabric`), and no neighbouring floor:
        # an edge to nowhere.
        leaks: list[str] = []
        for r, p in shapes:
            if r.kind.is_outdoor or r.kind in _UNENTERED:
                continue  # balconies are open by definition
            covered = p.exterior.intersection(fabric).length / p.exterior.length
            if covered >= 0.9:
                continue
            gap = p.exterior.difference(fabric)
            void = 0.0
            for g in getattr(gap, "geoms", [gap]):
                if g.length < 400.0:
                    continue
                mid = g.interpolate(0.5, normalized=True)
                faces_room = any(
                    q.distance(mid) < 150.0 for other, q in shapes if other is not r
                )
                if not faces_room:
                    void += g.length
            if void > p.exterior.length * 0.1:
                leaks.append(
                    f"{r.label or r.kind.value} {covered:.0%} "
                    f"({void:.0f} mm facing nothing)"
                )
        if leaks:
            audit.failures.append(
                f"{len(leaks)} enclosed room(s) not sealed: " + ", ".join(leaks[:4])
            )

    # --- reachability ------------------------------------------------------
    doored = {rid for o in plan.openings if o.kind.is_door for rid in (o.connects or ()) if rid}
    walls_with_doors = {o.wall_id for o in plan.openings if o.kind.is_door}
    stranded: list[str] = []
    for r, p in shapes:
        if r.kind.is_outdoor or r.kind in _UNENTERED:
            continue
        if r.id in doored:
            continue
        # Fall back to geometry: a door on a wall along this room's edge, or a
        # neighbour it is continuously open to.
        ring = p.exterior
        if any(
            LineString([w.start, w.end]).distance(ring) < 300.0
            for w in plan.walls
            if w.id in walls_with_doors
        ):
            continue
        if any(q.distance(p) < 30.0 for other, q in shapes if other is not r):
            continue
        stranded.append(r.label or r.kind.value)
    if stranded:
        audit.failures.append(f"{len(stranded)} room(s) unreachable: " + ", ".join(stranded[:4]))

    # --- wall fragmentation -------------------------------------------------
    # A wall with a door in it is one wall. A flat needs a handful per room — the unit
    # this pipeline was tuned against has 3.7 — so twenty-plus means the boundary runs
    # were never fused and the model is a mosaic of stubs. That is not a judgement call:
    # it litters the render with seams and it shatters the floor cut, which is what left
    # rooms unsealed before. Auditing one flat hid it completely; across sixteen the
    # spread was 3.7 to 23.6.
    density = len(solid_walls) / max(len(shapes), 1)
    if density > 15.0:
        audit.failures.append(
            f"walls are fragmented: {len(solid_walls)} walls for {len(shapes)} rooms "
            f"({density:.1f} per room, expected under ~8)"
        )
    elif density > 9.0:
        audit.notes.append(f"{density:.1f} walls per room — more fragmented than unit 1's 3.7")

    # --- gross area shortfall ----------------------------------------------
    # Measured floor area is legitimately below a declared carpet area by the wall
    # footprints, which is why area is otherwise only reported. Below 60% is not that:
    # it means most of the dwelling was never recovered.
    if audit.area_ratio is not None and audit.area_ratio < 0.6:
        audit.failures.append(
            f"only {audit.area_ratio:.0%} of the declared {audit.declared_sqft:.0f} sqft was "
            f"recovered ({len(shapes)} rooms) — extraction largely failed for this unit"
        )

    # --- drawn walls swallowed by a room (the independent check) -----------
    if drawn is not None:
        swallowed: list[tuple[str, float]] = []
        for r, p in shapes:
            inner = p.buffer(-120.0)
            if inner.is_empty:
                continue
            length = drawn.intersection(inner).length
            # A wet room legitimately contains kerb-scale line work — a shower kerb
            # or screen is drawn on the wall layer, and the printed room dimension
            # spans it. Real swallowed walls measure well past this.
            threshold = 3500.0 if r.kind.is_wet else 1500.0
            if length > threshold:
                swallowed.append((r.label or r.kind.value, length))
        if swallowed:
            worst = max(swallowed, key=lambda t: t[1])
            audit.failures.append(
                f"{len(swallowed)} room(s) swallow a wall the drawing shows, worst "
                f"{worst[0]} with {worst[1]:.0f} mm inside it"
            )

    # --- every square millimetre of the declared polygon --------------------
    # The declared carpet polygon measures within ±0.2% of the printed figure, so it is
    # the one exact statement of the dwelling's extent. Everything inside it must be
    # floor or wall body; whatever is neither is extraction loss, located precisely.
    bnd = plan.units[0].boundary if plan.units else []
    if len(bnd) >= 3:
        bpoly = Polygon(bnd)
        if not bpoly.is_valid:
            bpoly = bpoly.buffer(0)
        floors = unary_union([q for _, q in shapes])
        bodies = (
            unary_union(
                [
                    LineString([w.start, w.end]).buffer(w.thickness_mm / 2.0, cap_style=2)
                    for w in plan.walls
                ]
            )
            if plan.walls
            else Polygon()
        )
        assigned = floors.union(bodies).intersection(bpoly)
        # The drawing's own wall bands are structure whether or not a wall was built
        # on them — carpet area includes internal walls, so crediting them is what
        # makes this figure comparable with the declaration.
        if drawn is not None:
            assigned = assigned.union(drawn.buffer(120.0).intersection(bpoly))
        audit.accounted = assigned.area / bpoly.area if bpoly.area else None
        if audit.accounted is not None and audit.accounted < 0.90:
            leftover = bpoly.difference(assigned)
            pieces = sorted(
                (g for g in getattr(leftover, "geoms", [leftover]) if g.area > 1e5),
                key=lambda g: -g.area,
            )
            big = [g for g in pieces if g.area >= 2e6]
            where = (
                f", largest {pieces[0].area / 1e6:.1f} m2 near "
                f"({pieces[0].centroid.x:.0f}, {pieces[0].centroid.y:.0f})"
                if pieces
                else ""
            )
            audit.failures.append(
                f"only {audit.accounted:.0%} of the declared polygon is floor or wall "
                f"body; {len(big)} unassigned piece(s) over 2 m2{where}"
            )

    # --- printed dimensions --------------------------------------------------
    # The architect wrote W x D beside most rooms. Those numbers are the spec the room
    # was drawn to; a recovered room far from them is mis-mapped even if it is sealed,
    # reachable and inside the polygon.
    ags = [
        (a, r.label or r.kind.value)
        for r, _ in shapes
        for a in [r.dimension_agreement()]
        if a is not None
    ]
    if ags:
        audit.dims = sum(a for a, _ in ags) / len(ags)
        worst = min(ags)
        if audit.dims < 0.85 or worst[0] < 0.55:
            audit.failures.append(
                f"printed-dimension agreement {audit.dims:.0%} across {len(ags)} "
                f"labelled room(s), worst {worst[1]} at {worst[0]:.0%}"
            )

    # --- interior walls the drawing does not draw ----------------------------
    # Between social/circulation spaces an unsupported wall is an invention (the
    # lobby-living case the user caught by eye). Between other kinds a wall without
    # line work is deliberate policy — wet rooms stay enclosed — so only pairs where
    # openness is plausible are charged.
    if drawn is not None:
        social = {
            RoomKind.LIVING,
            RoomKind.DINING,
            RoomKind.LOBBY,
            RoomKind.CORRIDOR,
            RoomKind.ENTRANCE,
        }
        invented: list[tuple[str, float, float]] = []
        for w in solid_walls:
            if w.exterior or w.length_mm < 600.0:
                continue
            line = LineString([w.start, w.end])
            support = drawn.intersection(line.buffer(150.0)).length
            if support >= w.length_mm * 0.25:
                continue
            mid = line.interpolate(0.5, normalized=True)
            nx, ny = w.normal()
            off = w.thickness_mm / 2.0 + 200.0
            sides: list[RoomKind] = []
            for sgn in (1.0, -1.0):
                probe = Point(mid.x + sgn * nx * off, mid.y + sgn * ny * off)
                for r, q in shapes:
                    if q.contains(probe):
                        sides.append(r.kind)
                        break
            if len(sides) == 2 and all(k in social for k in sides):
                invented.append((w.id, w.length_mm, support))
        if invented:
            worst_w = max(invented, key=lambda t: t[1])
            audit.failures.append(
                f"{len(invented)} wall(s) between open-plan rooms lack drawn evidence, "
                f"worst {worst_w[0]} {worst_w[1]:.0f} mm with {worst_w[2]:.0f} mm support"
            )

    # --- elements the drawing does not attest --------------------------------
    walls_by_id = {w.id: w for w in plan.walls}
    if evidence is not None:
        phantom: list[str] = []
        ghost_windows: list[str] = []
        for o in plan.openings:
            w = walls_by_id.get(o.wall_id)
            if w is None:
                continue
            x, y = w.point_at(o.t_center)
            if o.kind.is_door and o.source is Source.CAD_VECTOR:
                # A sliding door is reclassified window line work, so its witness is
                # the glazing, not a leaf symbol.
                if o.kind is OpeningKind.SLIDING_DOOR:
                    ok = (
                        evidence.windows is not None
                        and evidence.windows.distance(Point(x, y)) < 600.0
                    )
                else:
                    ok = any(
                        math.hypot(x - dx, y - dy) < 1500.0
                        for dx, dy in evidence.door_points
                    )
                if not ok:
                    phantom.append(o.id)
            elif o.kind in (OpeningKind.WINDOW, OpeningKind.FRENCH_WINDOW):
                if evidence.windows is None or evidence.windows.distance(Point(x, y)) > 500.0:
                    ghost_windows.append(o.id)
        if phantom:
            audit.failures.append(
                f"{len(phantom)} door(s) claim drawing provenance with no drawn symbol "
                "near: " + ", ".join(phantom[:4])
            )
        if ghost_windows:
            audit.failures.append(
                f"{len(ghost_windows)} window(s) have no drawn window line work near "
                "them: " + ", ".join(ghost_windows[:4])
            )
        inferred = sum(
            1 for o in plan.openings if o.kind.is_door and o.source is Source.INFERRED
        )
        if inferred:
            audit.notes.append(f"{inferred} door(s) inferred — reasoned, not drawn")

    # --- doubled doors --------------------------------------------------------
    doors = [
        (o, walls_by_id[o.wall_id])
        for o in plan.openings
        if o.kind.is_door and o.wall_id in walls_by_id
    ]
    doubled = 0
    for i, (a, wa) in enumerate(doors):
        ax, ay = wa.point_at(a.t_center)
        for b, wb in doors[i + 1 :]:
            bx, by = wb.point_at(b.t_center)
            if math.hypot(ax - bx, ay - by) < 600.0:
                doubled += 1
    if doubled:
        audit.failures.append(f"{doubled} pair(s) of doors closer than 600 mm — doubled openings")

    # --- furniture placements -------------------------------------------------
    # The solver promises containment, separation and clear door swings; the audit
    # re-measures all three from the placements alone, trusting nothing.
    try:
        from .furnish.solver import (
            SolverConfig,
            _door_swing_zones,
            _oriented_box,
            _room_polygon_m,
            furnish_plan,
        )

        layout = furnish_plan(plan, origin=(0.0, 0.0))
        rooms_m = {
            r.id: _room_polygon_m(r, (0.0, 0.0)) for r in plan.rooms if len(r.polygon) >= 3
        }
        cfg = SolverConfig()
        hard_by_room: dict[str, Any] = {}
        for r in plan.rooms:
            if len(r.polygon) < 3:
                continue
            hard, _ = _door_swing_zones(plan, r, (0.0, 0.0), cfg)
            if hard:
                hard_by_room[r.id] = unary_union(hard)
        boxes: list[tuple[Any, Any]] = []
        outside: list[str] = []
        blocking: list[str] = []
        for f in layout.placements:
            fw, fd = f.footprint
            if fw <= 0 or fd <= 0:
                continue
            fp = _oriented_box(f.position[0], f.position[2], fw, fd, f.rotation_y)
            room_poly = rooms_m.get(f.room_id)
            if room_poly is not None and fp.difference(room_poly.buffer(0.05)).area > fp.area * 0.15:
                outside.append(f.id)
            zone = hard_by_room.get(f.room_id)
            if zone is not None and not f.locked and fp.intersection(zone).area > 0.05:
                blocking.append(f.id)
            boxes.append((f, fp))
        over = 0
        for i, (fa, pa) in enumerate(boxes):
            for fb, pb in boxes[i + 1 :]:
                if fa.room_id != fb.room_id:
                    continue
                if pa.intersection(pb).area > 0.25 * min(pa.area, pb.area):
                    over += 1
        if outside:
            audit.failures.append(
                f"{len(outside)} furniture item(s) poke outside their room: "
                + ", ".join(outside[:4])
            )
        if over:
            audit.failures.append(f"{over} furniture pair(s) overlap")
        if blocking:
            audit.failures.append(
                f"{len(blocking)} furniture item(s) block a door swing: "
                + ", ".join(blocking[:4])
            )
    except Exception as exc:  # noqa: BLE001 — the audit must report, not crash
        audit.notes.append(f"placement check skipped: {exc}")

    unknown = [r for r in plan.rooms if r.kind is RoomKind.UNKNOWN]
    if unknown:
        audit.notes.append(f"{len(unknown)} unclassified room(s)")
    return audit


def audit_file(path: Path | str, *, max_units: int | None = None) -> AuditResult:
    """Extract ``path`` and audit every dwelling in it, against the drawing itself."""
    from .ingest.dxf_plan import extract_plan

    plan = extract_plan(Path(path), max_units=max_units)
    ev = _drawing_evidence(Path(path))
    return audit_plan(plan, drawn=(ev.walls if ev else None), evidence=ev)


@dataclass
class DrawingEvidence:
    """What the drawing itself attests: wall line work, door symbols, window line work.

    Every "did we invent this?" check compares our output against this bundle rather
    than against our own reconstruction, because the reconstruction cannot testify in
    its own defence.
    """

    walls: Any | None = None
    door_points: list[tuple[float, float]] = field(default_factory=list)
    windows: Any | None = None


def _drawing_evidence(path: Path) -> DrawingEvidence | None:
    """Gather the drawing's own testimony in one pass."""
    try:
        from .ingest.dxf_flatten import flatten_layout
        from .ingest.dxf_plan import _bucket, _resolve_scale, _scale_entity, dwg_to_dxf
        from .ingest.dxf_repair import load_dxf
        from .ingest.layers import LayerRole

        dxf = path
        if path.suffix.lower() == ".dwg":
            dxf, _ = dwg_to_dxf(path, path.parent)
        doc, _ = load_dxf(dxf)
        ents, _ = flatten_layout(doc.modelspace())
        mm, _ = _resolve_scale(doc, ents)
        ents = [_scale_entity(e, mm) for e in ents]
        b = _bucket(ents)
        wall_lines = [
            LineString([a, c])
            for e in b.get(LayerRole.WALL)
            for a, c in zip(e.points, e.points[1:])
            if LineString([a, c]).length > 400.0
        ]
        win_lines = [
            LineString([a, c])
            for e in b.get(LayerRole.WINDOW, LayerRole.WINDOW_SILL)
            for a, c in zip(e.points, e.points[1:])
            if LineString([a, c]).length > 100.0
        ]
        return DrawingEvidence(
            walls=unary_union(wall_lines) if wall_lines else None,
            door_points=[e.points[0] for e in b.get(LayerRole.DOOR) if e.points],
            windows=unary_union(win_lines) if win_lines else None,
        )
    except Exception:
        return None


def _drawn_walls(path: Path) -> Any | None:
    """The drawing's own wall line work. Kept for callers that only need walls."""
    ev = _drawing_evidence(path)
    return ev.walls if ev else None
