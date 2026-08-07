"""Assemble a PlanIR into a renderable SceneIR.

This is the stage that turns a measured plan into something that looks designed.
Three decisions here matter more than the rest:

**Re-centre the model.** The demo drawing sits at survey coordinates around
(27,700,000, -7,050,000) mm. Passed straight to a renderer, single-precision floats
have ~2 m of resolution at that magnitude, so the geometry visibly jitters and
shadows crawl. The offset is subtracted here and recorded in
``SceneIR.origin_offset`` so nothing is lost.

**Choose an accent wall per room.** Real interiors have a hierarchy: the wall the
bed's headboard backs onto, the wall the sofa faces. Picking one and giving it its
own material slot is a small amount of code that does a disproportionate amount of
the aesthetic work, because it is the difference between a room with four identical
walls and a room with a front.

**Light the space properly.** A single overhead point light per room gives the flat,
grey, evenly-lit look of a CAD viewport. What reads as architectural photography is
directional daylight through the actual window openings plus soft interior fill at a
warmer temperature, so surfaces get a gradient across them and rooms differ from
each other by orientation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from ..planir import OpeningKind, PlanIR, Room, RoomKind, Wall
from ..sceneir import (
    MM_TO_M,
    CameraSpec,
    FurniturePlacement,
    LightSpec,
    MaterialSlot,
    OpeningSpec,
    RoomZone,
    SceneIR,
    Surface,
    SurfaceKind,
)
from ..furnish.solver import SolverConfig, furnish_plan
from .schemes import get_scheme, scheme_summaries
from .solids import BuildConfig, build_room_surfaces, build_wall_surfaces, floor_slot_for, wall_slot_for

__all__ = ["AssembleConfig", "assemble_scene"]


@dataclass(slots=True)
class AssembleConfig:
    build: BuildConfig = None  # type: ignore[assignment]
    solver: SolverConfig = None  # type: ignore[assignment]
    scheme_id: str = "warm-minimal"
    direct_art: bool = True
    """Let the art director choose the scheme, mood and accents. When False the
    explicitly requested ``scheme_id`` is used unchanged."""

    furnish: bool = True
    generate_lights: bool = True
    accent_walls: bool = True

    def __post_init__(self) -> None:
        if self.build is None:
            self.build = BuildConfig()
        if self.solver is None:
            self.solver = SolverConfig()


def _plan_origin(plan: PlanIR) -> tuple[float, float]:
    """Centre of the plan's extent, subtracted from every coordinate."""
    xs: list[float] = []
    ys: list[float] = []
    for w in plan.walls:
        xs += [w.start[0], w.end[0]]
        ys += [w.start[1], w.end[1]]
    for r in plan.rooms:
        for x, y in r.polygon:
            xs.append(x)
            ys.append(y)
    if not xs:
        return (0.0, 0.0)
    return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)


def _bounding_walls(plan: PlanIR, room: Room) -> list[Wall]:
    """Walls that form this room's boundary."""
    from shapely.geometry import LineString, Polygon as ShpPolygon

    if len(room.polygon) < 3:
        return []
    ring = ShpPolygon(room.polygon).exterior
    out: list[Wall] = []
    for w in plan.walls:
        if w.kind in ("parapet", "railing"):
            continue
        if LineString([w.start, w.end]).distance(ring) < 250.0:
            out.append(w)
    return out


def _pick_accent_wall(plan: PlanIR, room: Room) -> str | None:
    """The wall that should carry the room's feature finish.

    Chosen as the longest bounding wall with no opening in it. The reasoning is
    what a designer would do: the accent wall is the one you look at and put
    something against, so it must be long enough to matter and unbroken by a door
    or window. Falling back to the longest wall regardless is deliberate — better a
    slightly awkward accent than none, which loses the room's hierarchy entirely.
    """
    if not room.kind.is_habitable:
        return None
    walls = _bounding_walls(plan, room)
    if not walls:
        return None
    clear = [w for w in walls if not plan.openings_for_wall(w.id) and not w.exterior]
    pool = clear or [w for w in walls if not w.exterior] or walls
    best = max(pool, key=lambda w: w.length_mm)
    return best.id if best.length_mm >= 1800.0 else None


def _opening_specs(plan: PlanIR, origin: tuple[float, float]) -> list[OpeningSpec]:
    ox, oy = origin
    out: list[OpeningSpec] = []
    for o in plan.openings:
        wall = plan.wall_by_id(o.wall_id)
        if wall is None:
            continue
        c = wall.point_at(o.t_center)
        dx, dy = wall.direction()
        # Yaw so the opening's local +x runs along the wall, in the Y-flipped frame.
        yaw = math.atan2(-dy, dx)
        glazed = not o.kind.is_door or o.kind is OpeningKind.FRENCH_WINDOW
        out.append(
            OpeningSpec(
                id=o.id,
                kind=o.kind.value,
                wall_id=o.wall_id,
                centre=(
                    (c[0] - ox) * MM_TO_M,
                    (o.sill_mm + o.height_mm / 2.0) * MM_TO_M,
                    -(c[1] - oy) * MM_TO_M,
                ),
                width=o.width_mm * MM_TO_M,
                height=o.height_mm * MM_TO_M,
                sill=o.sill_mm * MM_TO_M,
                rotation_y=yaw,
                thickness=wall.thickness_mm * MM_TO_M,
                swing=o.swing,
                swing_inward=o.swing_inward,
                leaf_slot=(
                    MaterialSlot.DOOR_LEAF.value if o.kind.is_door else MaterialSlot.WINDOW_FRAME.value
                ),
                frame_slot=(
                    MaterialSlot.DOOR_FRAME.value if o.kind.is_door else MaterialSlot.WINDOW_FRAME.value
                ),
                glazed=glazed,
            )
        )
    return out


def _lights(
    plan: PlanIR, rooms: list[RoomZone], openings: list[OpeningSpec], scheme_id: str
) -> list[LightSpec]:
    """Sun, plus one soft interior source per room, plus real-time-only window fill.

    Intensities are relative and each backend scales them; what matters here is the
    *ratio*. The first attempt emitted a window fill light for every opening at full
    strength on top of the sun, and in Cycles that triple-counted the daylight and
    produced a pure white image. Those fills are now flagged ``realtime_only``,
    because a path tracer already gets the same light physically from the sky
    through the opening, while a rasteriser cannot and needs the stand-in.

    Interior sources are area lights sized to the room rather than point lights. A
    point light throws a single hard shadow per object, which reads as a video game;
    an area light gives the soft gradient of a real ceiling fitting.
    """
    scheme = get_scheme(scheme_id)
    out: list[LightSpec] = [
        LightSpec(
            id="sun",
            kind="sun",
            direction=(-0.42, -0.78, -0.46),
            intensity=scheme.sun_intensity,
            colour_k=scheme.sun_temperature_k,
            cast_shadow=True,
        )
    ]

    for op in openings:
        if not op.glazed:
            continue
        nx, nz = math.sin(op.rotation_y + math.pi / 2), math.cos(op.rotation_y + math.pi / 2)
        out.append(
            LightSpec(
                id=f"daylight-{op.id}",
                kind="area",
                position=(op.centre[0] + nx * 0.12, op.centre[1], op.centre[2] + nz * 0.12),
                direction=(-nx, -0.12, -nz),
                intensity=min(6.0, op.width * op.height * 1.1),
                colour_k=scheme.sun_temperature_k + 700.0,
                size=(max(0.3, op.width), max(0.3, op.height)),
                cast_shadow=False,
                realtime_only=True,
            )
        )

    for rz in rooms:
        if rz.kind in (RoomKind.DUCT.value, RoomKind.LIFT.value) or rz.area_m2 < 1.5:
            continue
        # Scaled to room area but firmly capped: a 27 m2 room lit by one enormous
        # source looks like a film set, and uncapped scaling was what blew out the
        # first render.
        intensity = min(4.5, 1.2 + rz.area_m2 * 0.09)
        side = min(1.1, 0.35 + rz.area_m2 * 0.03)
        out.append(
            LightSpec(
                id=f"fill-{rz.id}",
                kind="area",
                position=(rz.centroid[0], rz.floor_level + rz.ceiling_height - 0.08, rz.centroid[2]),
                direction=(0.0, -1.0, 0.0),
                intensity=intensity,
                colour_k=scheme.interior_temperature_k,
                size=(side, side),
                room_id=rz.id,
                cast_shadow=False,
            )
        )
    return out


def _principal_axis(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Direction of greatest spread of a polygon, by 2x2 PCA.

    A room's long axis is the direction you look along to show its depth. Computing
    it from the covariance eigenvector handles rotated and L-shaped rooms, where
    just taking the bounding box would give an axis-aligned answer that has nothing
    to do with the room.
    """
    n = len(points)
    if n < 2:
        return (1.0, 0.0)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points) / n
    syy = sum((p[1] - my) ** 2 for p in points) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points) / n
    # Largest eigenvector of [[sxx, sxy], [sxy, syy]].
    tr, det = sxx + syy, sxx * syy - sxy * sxy
    disc = max(0.0, tr * tr / 4.0 - det)
    lam = tr / 2.0 + math.sqrt(disc)
    vx, vy = (sxy, lam - sxx) if abs(sxy) > 1e-9 else (1.0, 0.0)
    n2 = math.hypot(vx, vy)
    return (vx / n2, vy / n2) if n2 > 1e-9 else (1.0, 0.0)


def _dress_scene(
    plan: PlanIR,
    scene: SceneIR,
    accent_by_wall: dict[str, str],
    origin: tuple[float, float],
) -> list[FurniturePlacement]:
    """Hang curtains at the windows and art on the accent walls.

    This is the styling layer, kept separate from the layout solver on purpose: the
    solver decides where furniture goes by circulation and clearance, whereas a
    curtain's position is fixed by its window and a picture's by the wall it hangs on.
    Neither is a search problem.

    It is also where the largest remaining gap to a published render was. Geometry,
    materials and light were all physically right and the images still read as models,
    because a real room has soft vertical folds beside its windows and something on its
    walls. Both are emitted as ordinary catalogue placements, so the browser viewer and
    the Cycles render pick them up with no renderer-specific code.

    Every placement is **verified against the room it claims to be in** before being
    emitted. The first version chose a room for each wall by proximity and then probed
    which side was inward, and for a wall between two rooms that choice is ambiguous —
    it picked the neighbour, the inward test then flipped, and curtains ended up
    standing in the middle of the floor 0.9 m outside any room. Probing first and
    letting the *containing room* identify itself removes the ambiguity entirely.
    """
    from shapely.geometry import Point as SPt, Polygon as SP

    from ..furnish.catalog import CATALOG

    out: list[FurniturePlacement] = []
    zone_by_id = {z.id: z for z in scene.rooms}

    # Room shapes in scene metres, once.
    shapes: list[tuple[Room, SP]] = []
    for r in plan.rooms:
        if len(r.polygon) < 3:
            continue
        shapes.append(
            (
                r,
                SP([((x - origin[0]) * MM_TO_M, -(y - origin[1]) * MM_TO_M) for x, y in r.polygon]),
            )
        )

    def room_containing(x: float, z: float) -> tuple[Room, SP] | None:
        for r, shp in shapes:
            if shp.contains(SPt(x, z)):
                return r, shp
        return None

    # Wall footprints and already-placed furniture, so dressing can collision-check.
    # Dressing runs after the layout solver and so was never checked against anything:
    # curtain panels ended up standing in the middle of rooms and passing through
    # walls, because only their centre point was validated, not their footprint.
    from shapely.geometry import LineString as SLine
    from shapely.ops import unary_union as sunion

    wall_solid = sunion(
        [
            SLine(
                [
                    ((w.start[0] - origin[0]) * MM_TO_M, -(w.start[1] - origin[1]) * MM_TO_M),
                    ((w.end[0] - origin[0]) * MM_TO_M, -(w.end[1] - origin[1]) * MM_TO_M),
                ]
            ).buffer(w.thickness_mm * MM_TO_M / 2.0, cap_style=2, join_style=2)
            for w in plan.walls
            if w.length_mm > 1e-6
        ]
        or [SP()]
    )

    def footprint(cx: float, cz: float, w: float, d: float, yaw: float) -> SP:
        hw, hd = w / 2.0, d / 2.0
        c, sn = math.cos(yaw), math.sin(yaw)
        return SP(
            [
                (cx + lx * c + lz * sn, cz - lx * sn + lz * c)
                for lx, lz in ((-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd))
            ]
        )

    occupied: list[SP] = []
    for pl in scene.furniture:
        fw, fd = pl.footprint
        if fw > 0 and fd > 0:
            occupied.append(footprint(pl.position[0], pl.position[2], fw, fd, pl.rotation_y))

    def fits(fp: SP, shp: SP) -> bool:
        """Inside the room, clear of walls, and not on top of existing furniture."""
        if fp.area <= 0:
            return False
        if fp.difference(shp.buffer(0.05)).area > 0.06 * fp.area:
            return False
        if fp.intersection(wall_solid).area > 0.06 * fp.area:
            return False
        return all(fp.intersection(o).area <= 0.10 * fp.area for o in occupied)

    # --- curtains at exterior windows only ---
    exterior_walls = {w.id for w in plan.walls if w.exterior}
    for op in scene.openings:
        if not op.glazed:
            continue
        # Only dress a genuine exterior window. Interior openings get classified as
        # glazed too (a pass-through, a mis-read door), and hanging curtains on them
        # put panels across the middle of the flat.
        if op.wall_id not in exterior_walls:
            continue
        # The opening's yaw defines its along-wall direction as (cos yaw, -sin yaw) in
        # scene (x, z); the wall normal is that turned a quarter: (sin yaw, cos yaw).
        nx, nz = math.sin(op.rotation_y), math.cos(op.rotation_y)
        probe = op.thickness / 2.0 + 0.45

        found = None
        for sign in (1.0, -1.0):
            hit = room_containing(op.centre[0] + nx * sign * probe, op.centre[2] + nz * sign * probe)
            if hit and hit[0].kind.is_habitable:
                found = (sign, *hit)
                break
        if found is None:
            continue
        sign, room, shp = found
        zone = zone_by_id.get(room.id)
        if zone is None:
            continue

        item_id = "curtain-wide" if op.width >= 2.0 else "curtain-std"
        item = CATALOG.get(item_id)
        if item is None:
            continue
        # A curtain must not be wider than the wall it hangs on.
        wall = plan.wall_by_id(op.wall_id)
        if wall is None or item.width_mm * MM_TO_M > wall.length_mm * MM_TO_M * 0.98:
            continue
        off = op.thickness / 2.0 + 0.10
        px = op.centre[0] + nx * sign * off
        pz = op.centre[2] + nz * sign * off
        # Validate the whole footprint, not just the centre. Panels extend well past
        # the window reveal, so a centre-only test happily approved curtains with half
        # their width buried in a return wall.
        if not fits(
            footprint(px, pz, item.width_mm * MM_TO_M, item.depth_mm * MM_TO_M, op.rotation_y),
            shp,
        ):
            continue
        out.append(
            FurniturePlacement(
                id=f"dress-curtain-{op.id}",
                catalog_id=item_id,
                room_id=room.id,
                position=(px, zone.floor_level, pz),
                # Local +x must run along the wall, which op.rotation_y already gives;
                # the panel is symmetric so no flip is needed for the other side.
                rotation_y=op.rotation_y,
                footprint=(item.width_mm * MM_TO_M, item.depth_mm * MM_TO_M),
                height=min(item.height_mm * MM_TO_M, zone.ceiling_height - 0.05),
                source="dressing",
                reason=f"hung at window {op.id}",
            )
        )
        occupied.append(
            footprint(px, pz, item.width_mm * MM_TO_M, item.depth_mm * MM_TO_M, op.rotation_y)
        )

    # --- art on each accent wall ---
    for wall_id, room_id in accent_by_wall.items():
        wall = plan.wall_by_id(wall_id)
        zone = zone_by_id.get(room_id)
        entry = next(((r, shp) for r, shp in shapes if r.id == room_id), None)
        if wall is None or zone is None or entry is None:
            continue
        room, shp = entry
        if not room.kind.is_habitable or plan.openings_for_wall(wall_id):
            continue  # never hang art over a door or window

        item_id = "art-triptych" if wall.length_mm >= 2600 else "art-single"
        item = CATALOG.get(item_id)
        if item is None or wall.length_mm < item.width_mm + 600:
            continue

        c = wall.point_at(0.5)
        dx, dy = wall.direction()
        nx_p, ny_p = wall.normal()

        def to_scene(px: float, py: float) -> tuple[float, float]:
            return ((px - origin[0]) * MM_TO_M, -(py - origin[1]) * MM_TO_M)

        # Which side of this wall is the room on? Ask the room, do not assume.
        placed = None
        for sign in (1.0, -1.0):
            probe = to_scene(
                c[0] + nx_p * sign * (wall.thickness_mm / 2.0 + 400.0),
                c[1] + ny_p * sign * (wall.thickness_mm / 2.0 + 400.0),
            )
            if shp.contains(SPt(*probe)):
                placed = sign
                break
        if placed is None:
            continue

        face = to_scene(
            c[0] + nx_p * placed * (wall.thickness_mm / 2.0 + 30.0),
            c[1] + ny_p * placed * (wall.thickness_mm / 2.0 + 30.0),
        )
        # The accent wall of a *partitioned zone* can be metres from that zone's own
        # boundary; hanging art there leaves it floating in mid-air.
        if shp.exterior.distance(SPt(*face)) > 0.4:
            continue
        art_yaw = math.atan2(-dy, dx)
        if not fits(
            footprint(
                face[0], face[1], item.width_mm * MM_TO_M, item.depth_mm * MM_TO_M, art_yaw
            ),
            shp,
        ):
            continue

        centre_h = 1.45 if item_id == "art-triptych" else 1.35
        out.append(
            FurniturePlacement(
                id=f"dress-art-{wall_id}",
                catalog_id=item_id,
                room_id=room_id,
                position=(
                    face[0],
                    zone.floor_level + centre_h - item.height_mm * MM_TO_M / 2.0,
                    face[1],
                ),
                rotation_y=art_yaw,
                footprint=(item.width_mm * MM_TO_M, item.depth_mm * MM_TO_M),
                height=item.height_mm * MM_TO_M,
                anchored_to_wall=wall_id,
                source="dressing",
                reason=f"hung on the accent wall of {room.label or room.kind.value}",
            )
        )
    return out


def _ceiling_infill(plan: PlanIR, *, origin: tuple[float, float], cfg: AssembleConfig) -> list[Surface]:
    """A ceiling patch covering the wall zones between rooms."""
    from shapely.geometry import Polygon as SP
    from shapely.ops import unary_union

    from .solids import MeshBuffer

    indoor = [r for r in plan.rooms if len(r.polygon) >= 3 and not r.kind.is_outdoor]
    if not indoor:
        return []
    rooms_union = unary_union([SP(r.polygon) for r in indoor])
    # Grow by a generous wall thickness to reach across every partition, then keep only
    # what is not already ceiled.
    envelope = rooms_union.buffer(360.0, join_style=2).buffer(-40.0, join_style=2)
    gap = envelope.difference(rooms_union)
    if gap.is_empty:
        return []

    height = max(r.ceiling_height_mm for r in indoor)
    pieces = list(getattr(gap, "geoms", [gap]))
    out: list[Surface] = []
    for i, piece in enumerate(pieces):
        if not isinstance(piece, SP) or piece.area < 10_000.0:
            continue
        ring = [
            ((x - origin[0]) * MM_TO_M, -(y - origin[1]) * MM_TO_M)
            for x, y in piece.exterior.coords[:-1]
        ]
        if len(ring) < 3:
            continue
        mesh = MeshBuffer()
        mesh.polygon(ring, height * MM_TO_M, up=False)
        surf = mesh.to_surface(
            f"s-ceil-infill-{i}", SurfaceKind.CEILING, MaterialSlot.CEILING.value
        )
        if surf:
            out.append(surf)
    return out


def _perimeter_enclosed(
    zone: RoomZone, wall_lines: list[Any], tol: float = 0.45
) -> float:
    """Fraction of a room's perimeter that runs along a wall.

    A room can pass every size and shape test and still be the wrong subject for a
    photograph because one side of it is missing — an exterior wall the solver could
    not recover leaves the room open to the sky, and the camera then looks straight out
    of the building at a gradient. Sampling the perimeter and asking how much of it has
    a wall next to it catches exactly that, and nothing else does.
    """
    from shapely.geometry import LineString as ShpLine, Point as P

    if len(zone.polygon) < 3:
        return 0.0
    ring = ShpLine(list(zone.polygon) + [zone.polygon[0]])
    if ring.length <= 0 or not wall_lines:
        return 0.0
    n = max(24, int(ring.length / 0.35))
    hit = 0
    for i in range(n):
        pt = ring.interpolate(i / n, normalized=True)
        if any(w.distance(pt) <= tol for w in wall_lines):
            hit += 1
    return hit / n


def _frame_cameras(
    zones: list[RoomZone],
    bounds: tuple[float, float, float, float, float, float],
    wall_lines: list[Any] | None = None,
    scene_openings: Sequence[OpeningSpec] = (),
    scene_furniture: Sequence[FurniturePlacement] = (),
    *,
    max_interior: int = 3,
    roofed: bool = True,
) -> list[CameraSpec]:
    """Frame one doll's-house overview plus a few interior views.

    The overview comes first because it is the image that communicates a flat:
    a raised three-quarter view with the ceilings hidden, which is what estate
    agents and architects both use to show a whole plan at once. It is also immune
    to the framing problems interior shots have.

    Interior views are framed along each room's principal axis rather than from a
    corner. The corner-based first attempt put the camera 0.45 m from a wardrobe,
    which filled the frame with a door front — technically inside the room and
    useless as a photograph. Looking *along* the long axis from near one end is what
    an architectural photographer does, and it needs the axis, a containment test
    and a real stand-off distance.
    """
    from shapely.geometry import Point as ShpPoint, Polygon as ShpPolygon

    out: list[CameraSpec] = []
    eye_h = 1.55
    min_standoff = 1.25
    cfg_min_view_depth = 2.6
    min_composition = 3.2
    """Minimum composition score for an interior view to be worth rendering."""

    # --- doll's house overview ---
    minx, miny, minz, maxx, maxy, maxz = bounds
    cx, cz = (minx + maxx) / 2.0, (minz + maxz) / 2.0
    span = max(maxx - minx, maxz - minz, 1.0)
    out.append(
        CameraSpec(
            id="cam-overview",
            room_id="",
            name="Overview",
            # Raised and pulled back on the diagonal; 40 mm keeps the perspective
            # calm so the plan reads as a plan rather than a wide-angle sweep.
            position=(cx + span * 0.62, maxy + span * 0.78, cz + span * 0.62),
            target=(cx, 0.6, cz),
            focal_mm=40.0,
        )
    )

    # Only rooms that can actually be photographed well get an interior view.
    #
    # Framing a 24 mm lens inside an irregular room is not always possible, and the
    # earlier attempt produced images filled by a wardrobe door 0.9 m from the lens.
    # Rooms whose recovered polygon is a thin sliver or an awkward L — which the
    # open-plan zone split produces — are skipped rather than photographed badly.
    # Fewer good views beat more bad ones, and the doll's-house overview always
    # covers the whole flat regardless.
    candidates: list[tuple[float, RoomZone]] = []
    for z in zones:
        if z.area_m2 < 7.0 or len(z.polygon) < 3:
            continue
        xs = [p[0] for p in z.polygon]
        zs = [p[1] for p in z.polygon]
        bw, bd = max(xs) - min(xs), max(zs) - min(zs)
        if min(bw, bd) < 2.2:
            continue  # too narrow to stand back in
        # Compactness: how much of its bounding box the room fills. A rectangle
        # scores 1.0; the wedge-shaped zones the open-plan split produces score far
        # lower, and those are the ones that cannot be photographed well.
        compactness = z.area_m2 / max(bw * bd, 1e-6)
        # Open-plan spaces are L-shaped by nature — this living room fills 45% of its
        # bounding box — so the compactness floor that protects against unphotographable
        # slivers also rejected the flat's main room outright. It is relaxed rather than
        # removed, and the framing pass below still has to find a position that is inside
        # the room, has real stand-off, and scores above `min_composition`. Those are the
        # tests that actually catch a bad photograph; compactness was only a proxy.
        open_plan = z.kind in ("living", "dining")
        if compactness < (0.42 if open_plan else 0.6):
            continue
        enclosure = _perimeter_enclosed(z, wall_lines or [])
        # Same reasoning for enclosure: an open-plan space is *defined* by not being
        # fully enclosed, and its unwalled edges open onto another interior space, so the
        # view still terminates on a wall — just a further one.
        if enclosure < (0.40 if open_plan else 0.62):
            continue
        # And the principal living space leads, because it is the room a practice puts
        # first. Enclosure still counts, it just no longer vetoes.
        lead = {"living": 1.6, "dining": 1.25}.get(z.kind, 1.0)
        candidates.append((z.area_m2 * compactness * max(enclosure, 0.55) * lead, z))
    candidates.sort(key=lambda t: -t[0])

    # Reserve the first interior slot for the principal living space rather than letting
    # it compete on score. It cannot win on score and should not have to: an L-shaped
    # 17.6 m² living room scores below a rectangular 13 m² bedroom on
    # area x compactness, so weighting alone still returned three bedrooms and no living
    # room. Which room leads a set of images is an editorial decision, not a geometric
    # one, and for a flat it is the living space.
    living = next((c for c in candidates if c[1].kind == "living"), None)
    if living is not None:
        candidates.remove(living)
        candidates.insert(0, living)

    for _, z in candidates[:max_interior]:
        poly = ShpPolygon(z.polygon)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or not isinstance(poly, ShpPolygon):
            continue

        try:
            from shapely.algorithms.polylabel import polylabel

            pole = polylabel(poly, tolerance=0.05)
        except Exception:
            pole = poly.representative_point()

        ax, az = _principal_axis(list(z.polygon))
        # Secondary axis too: the long axis is not always the best-composed view, and
        # a practice would look at both before choosing.
        axes = [(ax, az), (-ax, -az), (-az, ax), (az, -ax)]

        def reach_from(eye: tuple[float, float], dx: float, dz: float) -> float:
            t, out = 0.0, 0.0
            while t < 40.0:
                t += 0.1
                if poly.contains(ShpPoint(eye[0] + dx * t, eye[1] + dz * t)):
                    out = t
                else:
                    break
            return out

        def back_off(dx: float, dz: float) -> tuple[tuple[float, float], float]:
            best, moved, t = (pole.x, pole.y), 0.0, 0.0
            while t < 30.0:
                t += 0.1
                cand = ShpPoint(pole.x + dx * t, pole.y + dz * t)
                if poly.contains(cand) and poly.exterior.distance(cand) >= min_standoff:
                    best, moved = (cand.x, cand.y), t
                else:
                    break
            return best, moved

        # --- compose, rather than just point the camera somewhere legal ---
        #
        # The earlier version stood at one end of the long axis and looked at the other.
        # It produced technically valid frames that no one would publish: a blank wall
        # filling the image, or the camera aimed out through a missing envelope wall at
        # a sky gradient. What a practice actually optimises for is what is *in* the
        # frame — a window for light and depth, furniture for occupancy and scale, and
        # enough distance that the room reads as a space. So candidate views are scored
        # on exactly that and the best one wins.
        interest: list[tuple[float, float, float]] = []  # (x, z, weight)
        for op in scene_openings:
            if abs(op.centre[1] - z.floor_level) < 4.0:
                # Windows are worth more than doors: they bring the light and the view.
                interest.append((op.centre[0], op.centre[2], 2.4 if op.glazed else 1.0))
        for pl in scene_furniture:
            if pl.room_id == z.id:
                interest.append((pl.position[0], pl.position[2], 1.3))

        best_view: tuple[float, tuple[float, float], float, float] | None = None
        for dx, dz in axes:
            eye, moved = back_off(-dx, -dz)
            r = reach_from(eye, dx, dz)
            if r < 2.2:
                continue
            exit_pt = ShpPoint(eye[0] + dx * (r + 0.05), eye[1] + dz * (r + 0.05))
            if not any(w.distance(exit_pt) <= 0.6 for w in (wall_lines or [])):
                continue  # this view leaves the building

            score = min(r, 7.0) * 0.5
            half_fov = math.radians(36.0)  # ~24 mm on 36 mm sensor, horizontally
            for ix, iz, weight in interest:
                vx, vz = ix - eye[0], iz - eye[1]
                dist = math.hypot(vx, vz)
                if dist < 0.35 or dist > 12.0:
                    continue
                cosang = (vx * dx + vz * dz) / dist
                if cosang <= math.cos(half_fov):
                    continue  # outside the frame
                # Off-centre subjects compose better than dead-centre ones, so peak
                # the reward part-way out toward the frame edge rather than on axis.
                off = math.acos(max(-1.0, min(1.0, cosang))) / half_fov
                score += weight * (0.55 + 0.45 * math.sin(off * math.pi))
            if best_view is None or score > best_view[0]:
                best_view = (score, eye, dx, dz)

        # A view that satisfies every geometric rule can still be a photograph of a
        # blank wall. Requiring a minimum composition score means a room with nothing
        # worth framing is left to the overview instead of contributing a dull image.
        if best_view is None or best_view[0] < min_composition:
            continue
        _, eye, dir_x, dir_z = best_view
        reach = reach_from(eye, dir_x, dir_z)
        look = (eye[0] + dir_x * (reach + 1.5), eye[1] + dir_z * (reach + 1.5))

        if roofed:
            out.append(
                CameraSpec(
                    id=f"cam-{z.id}",
                    room_id=z.id,
                    name=z.name,
                    position=(eye[0], z.floor_level + eye_h, eye[1]),
                    target=(look[0], z.floor_level + eye_h * 0.92, look[1]),
                    focal_mm=24.0,
                )
            )
        else:
            # Open-top model: an eye-level shot looks over the walls at open sky, which
            # reads as a roofless box rather than a room. So the room is presented as a
            # cutaway from above — but the *angle* has to be chosen deliberately rather
            # than falling out of a height and a set-back.
            #
            # Picking those two independently is what went wrong: raising the camera to
            # 1.75x ceiling height and pulling back by half the room's depth put the
            # living room at roughly 75 degrees below horizontal, which is a photograph
            # of a floor with some furniture on it, not an interior. The pitch is the
            # thing being composed, so it is fixed first and the geometry solved for it.
            #
            # ~34 degrees is the doll's-house angle: high enough to clear the near wall
            # and read the plan, shallow enough that the furniture is seen in elevation
            # and the far wall stays in frame.
            pitch = math.radians(34.0)
            target_h = z.floor_level + 0.9
            # Just above the wall head. Higher would steepen the view back toward a plan.
            height = z.floor_level + z.ceiling_height + 0.35
            # The camera stays over the room's *own* footprint, at the composed
            # viewpoint. Solving for the set-back instead put it outside the room, and
            # the partitions in between then filled half the frame — the shot became a
            # view *past two walls* into the room rather than of the room. Directly above
            # the near end nothing can intervene, because there is no ceiling to look
            # through and no wall between the lens and the floor below it.
            span = (height - target_h) / math.tan(pitch)
            ahead = reach * 0.9
            # The composed viewpoint already stands a stand-off's distance into the room,
            # which left only ~2.2 m of depth in front of it and forced the pitch to 47
            # degrees. Retreating across the room's own footprint buys back the depth the
            # shallower angle needs while keeping the lens over the floor it is shooting.
            behind = reach_from(eye, -dir_x, -dir_z)
            pull = max(0.0, min(behind * 0.9, span - ahead))
            look_dist = ahead + pull
            out.append(
                CameraSpec(
                    id=f"cam-{z.id}",
                    room_id=z.id,
                    name=z.name,
                    position=(eye[0] - dir_x * pull, height, eye[1] - dir_z * pull),
                    target=(
                        eye[0] + dir_x * ahead,
                        target_h,
                        eye[1] + dir_z * ahead,
                    ),
                    # Wider than the 35 mm this used: on a shallower angle the room has
                    # to fill the frame from a viewpoint inside it.
                    focal_mm=28.0,
                )
            )
    return out


def assemble_scene(
    plan: PlanIR, *, config: AssembleConfig | None = None, name: str = ""
) -> SceneIR:
    """Build a complete SceneIR from a PlanIR."""
    cfg = config or AssembleConfig()
    origin = _plan_origin(plan)
    scene = SceneIR(
        name=name or (plan.units[0].name if plan.units else "Plan"),
        source_plan=plan.source_file,
        origin_offset=origin,
        scheme_id=cfg.scheme_id,
        available_schemes=[s["id"] for s in scheme_summaries()],
        imperial_display=plan.imperial_display,
    )

    accent_by_wall: dict[str, str] = {}
    zones: list[RoomZone] = []
    for room in plan.rooms:
        accent_id = _pick_accent_wall(plan, room) if cfg.accent_walls else None
        if accent_id:
            accent_by_wall[accent_id] = room.id
        cx = (room.centroid[0] - origin[0]) * MM_TO_M
        cz = -(room.centroid[1] - origin[1]) * MM_TO_M
        zones.append(
            RoomZone(
                id=room.id,
                name=room.label or room.kind.value.replace("_", " ").title(),
                kind=room.kind.value,
                centroid=(cx, room.floor_level_mm * MM_TO_M, cz),
                area_m2=room.area_mm2 / 1e6,
                floor_slot=floor_slot_for(room.kind),
                wall_slot=wall_slot_for(room.kind),
                ceiling_slot=MaterialSlot.CEILING.value,
                accent_wall_id=accent_id,
                polygon=[
                    ((x - origin[0]) * MM_TO_M, -(y - origin[1]) * MM_TO_M)
                    for x, y in room.polygon
                ],
                ceiling_height=room.ceiling_height_mm * MM_TO_M,
                floor_level=room.floor_level_mm * MM_TO_M,
                stated_dimension_m=(
                    (room.stated_dimension.width_mm * MM_TO_M, room.stated_dimension.depth_mm * MM_TO_M)
                    if room.stated_dimension
                    else None
                ),
            )
        )
    scene.rooms = zones

    # Walls. A wall's slot depends on which room it belongs to, so resolve the
    # room first; walls shared between rooms take the more specific finish (a wet
    # wall beats a dry one, since tiling stops at the room boundary).
    wall_room: dict[str, Room] = {}
    for room in plan.rooms:
        for w in _bounding_walls(plan, room):
            prev = wall_room.get(w.id)
            if prev is None or (room.kind.is_wet and not prev.kind.is_wet):
                wall_room[w.id] = room

    surfaces: list[Surface] = []
    for wall in plan.walls:
        room = wall_room.get(wall.id)
        slot = wall_slot_for(room.kind) if room else MaterialSlot.WALL_INTERIOR.value
        surfaces.extend(
            build_wall_surfaces(
                plan,
                wall,
                origin=origin,
                cfg=cfg.build,
                slot=slot,
                accent=wall.id in accent_by_wall,
            )
        )
    for room in plan.rooms:
        surfaces.extend(build_room_surfaces(room, origin=origin, cfg=cfg.build))

    # Close the ceiling over the wall zones.
    #
    # Ceilings are generated per room so each can take its own finish, which leaves the
    # strips *above the walls* uncovered — and looking up over a partition in an
    # interior render then shows sky through the roof. In the building there is one slab
    # over everything, so the gap is filled with a single surface on the same material
    # slot: modularity preserved, hole closed.
    # Only needed when ceilings exist; with an open top there is nothing to patch.
    if cfg.build.generate_ceilings:
        surfaces.extend(_ceiling_infill(plan, origin=origin, cfg=cfg))
    scene.surfaces = surfaces

    scene.openings = _opening_specs(plan, origin)

    if cfg.furnish:
        layout = furnish_plan(plan, origin=origin, cfg=cfg.solver)
        scene.furniture = layout.placements
        scene.notes.extend(layout.notes)
        if layout.rejected:
            top = list(layout.rejected.items())[:5]
            scene.notes.append(
                "could not fit: " + "; ".join(f"{k} ({v})" for k, v in top)
            )

    # Art direction before styling and lighting: the scheme it picks decides the
    # materials the dressing and the lights are then tuned to.
    if cfg.direct_art:
        from ..furnish.director import direct

        ad = direct(plan, scene)
        scene.scheme_id = ad.scheme_id or scene.scheme_id
        scene.slot_overrides = dict(ad.slot_overrides)
        # A per-room accent becomes an override on that room's own accent wall id, so
        # it survives independently of the global slot.
        for room_id, material_id in ad.accent_by_room.items():
            for wall_id, rid in accent_by_wall.items():
                if rid == room_id:
                    scene.slot_overrides[f"wall.accent@{wall_id}"] = material_id
        scene.art_direction = {
            "scheme_id": ad.scheme_id,
            "rationale": ad.rationale,
            "mood": ad.mood,
            "hero_room_id": ad.hero_room_id,
            "sun_azimuth_deg": ad.sun_azimuth_deg,
            "sun_elevation_deg": ad.sun_elevation_deg,
            "sun_temperature_k": ad.sun_temperature_k,
            "interior_temperature_k": ad.interior_temperature_k,
            "exposure": ad.exposure,
            "source": ad.source,
        }
        scene.notes.extend(ad.notes)
        if ad.rationale:
            scene.notes.append(f"direction: {ad.rationale}")

    # Styling last, so it can see the finished openings and accent walls.
    scene.furniture.extend(_dress_scene(plan, scene, accent_by_wall, origin))

    if cfg.generate_lights:
        scene.lights = _lights(plan, zones, scene.openings, scene.scheme_id)
        ad_dict = scene.art_direction
        if ad_dict:
            for l in scene.lights:
                if l.kind == "sun":
                    l.colour_k = float(ad_dict.get("sun_temperature_k", l.colour_k))
                elif l.room_id:
                    l.colour_k = float(ad_dict.get("interior_temperature_k", l.colour_k))

    # Bounds, for framing the default camera.
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for s in surfaces:
        v = s.vertices
        xs.extend(v[0::3])
        ys.extend(v[1::3])
        zs.extend(v[2::3])
    if xs:
        scene.bounds = (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))

    # Framed after bounds, since the overview camera is placed from the extent.
    from shapely.geometry import LineString as _SL

    _wall_lines = [
        _SL(
            [
                ((w.start[0] - origin[0]) * MM_TO_M, -(w.start[1] - origin[1]) * MM_TO_M),
                ((w.end[0] - origin[0]) * MM_TO_M, -(w.end[1] - origin[1]) * MM_TO_M),
            ]
        )
        for w in plan.walls
    ]
    scene.cameras = _frame_cameras(
        zones,
        scene.bounds,
        _wall_lines,
        scene.openings,
        scene.furniture,
        roofed=cfg.build.generate_ceilings,
    )
    # Lead with the room the director chose, after the overview.
    hero = scene.art_direction.get("hero_room_id")
    if hero:
        scene.cameras.sort(key=lambda c: (c.id != "cam-overview", c.room_id != hero))

    scene.notes.append(
        f"{len(surfaces)} surfaces, {sum(s.triangle_count for s in surfaces)} triangles, "
        f"{len(scene.openings)} openings, {len(scene.furniture)} furniture items, "
        f"{len(scene.lights)} lights, {len(scene.cameras)} framed views"
    )
    scene.notes.append(
        f"re-centred by ({origin[0]:.0f}, {origin[1]:.0f}) mm to keep float precision usable"
    )
    return scene
