"""Turn PlanIR into 3D surfaces.

The interesting problem here is cutting door and window openings out of walls.

The obvious route is constructive solid geometry: extrude the wall, extrude a box
per opening, subtract. It works, and it is what most pipelines do, but it is the
wrong tool. Boolean subtraction on coplanar faces is numerically fragile, so a
window flush with a wall face produces z-fighting slivers; it emits ragged triangle
soup that no longer has clean quads for UV mapping, so tiled materials smear
around openings; and it is slow enough to feel it when a user drags a window.

Walls are prisms and openings are axis-aligned boxes within them, so the
subtraction has a closed-form answer and CSG is unnecessary. Working in wall-local
coordinates — ``u`` along the wall, ``v`` across its thickness, ``w`` up — the wall
minus its openings decomposes exactly into boxes:

    w=h ┌────────┬────────┬────────┬────────┐
        │        │ lintel │        │ lintel │   above each opening
        │  pier  ├────────┤  pier  ├────────┤
        │        │  void  │        │  void  │   the opening itself
        │        ├────────┤        ├────────┤
        │        │ apron  │        │ apron  │   below a window sill
    w=0 └────────┴────────┴────────┴────────┘
        u=0                                 u=L

Every piece is a box, so the output is clean quads with predictable UVs, it is
exact, and it is fast. The reveals — the wall's thickness seen looking through the
opening — come free as the inward faces of the surrounding boxes, and getting those
right is what stops a model looking like cardboard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..planir import Opening, PlanIR, Room, RoomKind, Wall
from ..sceneir import MM_TO_M, MaterialSlot, Surface, SurfaceKind

__all__ = [
    "BuildConfig",
    "MeshBuffer",
    "build_wall_surfaces",
    "build_room_surfaces",
    "floor_slot_for",
    "wall_slot_for",
]


@dataclass(slots=True)
class BuildConfig:
    skirting_height_mm: float = 90.0
    skirting_proud_mm: float = 12.0
    """How far the skirting stands out from the wall face."""

    slab_thickness_mm: float = 150.0
    reveal_inset_mm: float = 25.0
    """How far a window/door frame is recessed from the wall face. A frame flush
    with the wall reads as a decal; a small reveal is what makes it read as built."""

    ceiling_inset_mm: float = 0.0
    generate_ceilings: bool = False
    """Off by default: the deliverable is an open-top cutaway model, which is how a
    whole flat is shown. It also removes a class of problem — with no ceiling there is
    no sky leaking through the strips above partitions, and no need to hide ceilings
    per camera."""
    generate_skirting: bool = True
    uv_scale_m: float = 1.0
    """Metres per UV unit. Materials then scale themselves via
    ``Material.texture_scale_m`` rather than needing per-surface UV rebuilds."""


class MeshBuffer:
    """Accumulates triangles for one surface, welding nothing and trusting quads.

    Vertices are emitted per-face rather than shared between faces. Sharing would
    force normal averaging across a 90° edge, which rounds off every corner of
    every wall into a soft blob — the opposite of the crisp arrises that make
    architectural renders read as architecture.
    """

    __slots__ = ("vertices", "indices", "normals", "uvs")

    def __init__(self) -> None:
        self.vertices: list[float] = []
        self.indices: list[int] = []
        self.normals: list[float] = []
        self.uvs: list[float] = []

    def __len__(self) -> int:
        return len(self.vertices) // 3

    def quad(
        self,
        a: tuple[float, float, float],
        b: tuple[float, float, float],
        c: tuple[float, float, float],
        d: tuple[float, float, float],
        uv: tuple[tuple[float, float], ...] | None = None,
    ) -> None:
        """Add a planar quad ``a-b-c-d`` (counter-clockwise seen from outside)."""
        n = _normal(a, b, c)
        base = len(self)
        for p in (a, b, c, d):
            self.vertices.extend(p)
            self.normals.extend(n)
        if uv is None:
            # Derive UVs from the quad's own edge lengths so texture density is
            # uniform in world space regardless of face size.
            w = _dist(a, b)
            h = _dist(b, c)
            uv = ((0.0, 0.0), (w, 0.0), (w, h), (0.0, h))
        for u, v in uv:
            self.uvs.extend((u, v))
        self.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])

    def polygon(
        self, points_xz: Sequence[tuple[float, float]], y: float, *, up: bool = True
    ) -> None:
        """Add a horizontal polygon at height ``y`` (a floor or a ceiling)."""
        if len(points_xz) < 3:
            return
        tris = _triangulate(points_xz)
        if not tris:
            return
        n = (0.0, 1.0, 0.0) if up else (0.0, -1.0, 0.0)
        base = len(self)
        for x, z in points_xz:
            self.vertices.extend((x, y, z))
            self.normals.extend(n)
            self.uvs.extend((x, z))
        for t in tris:
            # Handedness trap, worth spelling out because getting it backwards
            # renders every floor black and is not obvious from the code.
            #
            # Callers hand in a ring wound counter-clockwise in the (x, z) plane.
            # In a right-handed Y-up frame that winding produces a normal pointing
            # *down*: for a=(0,0,0), b=(1,0,0), c=(0,0,1) the cross product
            # (b-a)x(c-a) is (0,-1,0). So an up-facing floor needs the triangles
            # reversed, and a down-facing ceiling keeps them as they come.
            i, j, k = (t[0], t[2], t[1]) if up else (t[0], t[1], t[2])
            self.indices.extend((base + i, base + j, base + k))

    def to_surface(
        self,
        sid: str,
        kind: SurfaceKind,
        slot: str,
        *,
        room_id: str | None = None,
        wall_id: str | None = None,
    ) -> Surface | None:
        if not self.indices:
            return None
        return Surface(
            id=sid,
            kind=kind,
            slot=slot,
            vertices=self.vertices,
            indices=self.indices,
            normals=self.normals,
            uvs=self.uvs,
            room_id=room_id,
            wall_id=wall_id,
        )


def _normal(
    a: tuple[float, float, float], b: tuple[float, float, float], c: tuple[float, float, float]
) -> tuple[float, float, float]:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    n = math.sqrt(nx * nx + ny * ny + nz * nz)
    if n < 1e-12:
        return (0.0, 1.0, 0.0)
    return (nx / n, ny / n, nz / n)


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.dist(a, b)


def _triangulate(points_xz: Sequence[tuple[float, float]]) -> list[tuple[int, int, int]]:
    """Ear-clipping triangulation of a simple polygon, via mapbox_earcut."""
    try:
        import numpy as np
        from mapbox_earcut import triangulate_float64

        verts = np.array(points_xz, dtype=np.float64).reshape(-1, 2)
        rings = np.array([len(points_xz)], dtype=np.uint32)
        idx = triangulate_float64(verts, rings)
        # earcut returns numpy uint32; cast to Python int so SceneIR stays plain
        # JSON-serialisable data with no numpy dependency downstream.
        return [
            (int(idx[i]), int(idx[i + 1]), int(idx[i + 2])) for i in range(0, len(idx), 3)
        ]
    except Exception:
        # Fan triangulation. Correct for convex polygons and a visible-but-safe
        # degradation for concave ones, rather than dropping the surface.
        return [(0, i, i + 1) for i in range(1, len(points_xz) - 1)]


# ---------------------------------------------------------------------------
# material slot selection
# ---------------------------------------------------------------------------

_FLOOR_SLOTS: dict[RoomKind, MaterialSlot] = {
    RoomKind.LIVING: MaterialSlot.FLOOR_LIVING,
    RoomKind.DINING: MaterialSlot.FLOOR_LIVING,
    RoomKind.STUDY: MaterialSlot.FLOOR_LIVING,
    RoomKind.KITCHEN: MaterialSlot.FLOOR_KITCHEN,
    RoomKind.BEDROOM: MaterialSlot.FLOOR_BEDROOM,
    RoomKind.MASTER_BEDROOM: MaterialSlot.FLOOR_BEDROOM,
    RoomKind.KIDS_BEDROOM: MaterialSlot.FLOOR_BEDROOM,
    RoomKind.WARDROBE: MaterialSlot.FLOOR_BEDROOM,
    RoomKind.BATHROOM: MaterialSlot.FLOOR_WET,
    RoomKind.POWDER: MaterialSlot.FLOOR_WET,
    RoomKind.UTILITY: MaterialSlot.FLOOR_WET,
    RoomKind.BALCONY: MaterialSlot.FLOOR_OUTDOOR,
    RoomKind.DRY_BALCONY: MaterialSlot.FLOOR_OUTDOOR,
    RoomKind.TERRACE: MaterialSlot.FLOOR_OUTDOOR,
}


def floor_slot_for(kind: RoomKind) -> str:
    """Choose the floor slot for a room programme.

    Defaults to circulation rather than living, so an unclassified region gets a
    neutral finish instead of confidently laying oak in what might be a duct.
    """
    return _FLOOR_SLOTS.get(kind, MaterialSlot.FLOOR_CIRCULATION).value


def wall_slot_for(kind: RoomKind) -> str:
    if kind.is_wet:
        return MaterialSlot.WALL_WET.value
    return MaterialSlot.WALL_INTERIOR.value


# ---------------------------------------------------------------------------
# wall solids
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _Span:
    """An opening projected onto the wall's local ``u`` axis."""

    u0: float
    u1: float
    sill: float
    head: float


def _opening_spans(wall: Wall, openings: Iterable[Opening], length: float) -> list[_Span]:
    """Project openings onto the wall axis, clamped and sorted, in metres."""
    spans: list[_Span] = []
    for o in openings:
        half = (o.width_mm * MM_TO_M) / 2.0
        centre = o.t_center * length
        u0 = max(0.0, centre - half)
        u1 = min(length, centre + half)
        if u1 - u0 < 0.05:
            continue
        sill = max(0.0, o.sill_mm * MM_TO_M)
        head = sill + o.height_mm * MM_TO_M
        spans.append(_Span(u0, u1, sill, head))

    spans.sort(key=lambda s: s.u0)
    # Merge overlaps. Two openings that overlap would otherwise generate a pier of
    # negative width, which folds the wall inside out.
    merged: list[_Span] = []
    for s in spans:
        if merged and s.u0 <= merged[-1].u1 + 1e-6:
            prev = merged[-1]
            prev.u1 = max(prev.u1, s.u1)
            prev.sill = min(prev.sill, s.sill)
            prev.head = max(prev.head, s.head)
        else:
            merged.append(s)
    return merged


def build_wall_surfaces(
    plan: PlanIR,
    wall: Wall,
    *,
    origin: tuple[float, float],
    cfg: BuildConfig,
    slot: str,
    accent: bool = False,
) -> list[Surface]:
    """Build one wall as boxes, with its openings cut out exactly.

    Works in wall-local ``(u, v, w)`` and maps to world at the end, so all the
    geometry reasoning is axis-aligned and the only trigonometry is one rotation.
    """
    length = wall.length_mm * MM_TO_M
    if length < 1e-4:
        return []
    thickness = wall.thickness_mm * MM_TO_M
    height = wall.height_mm * MM_TO_M
    half_t = thickness / 2.0

    dx, dy = wall.direction()
    ox, oy = origin
    sx = (wall.start[0] - ox) * MM_TO_M
    sz = -(wall.start[1] - oy) * MM_TO_M  # CAD +Y is north; glTF -Z is forward.

    def world(u: float, v: float, w: float) -> tuple[float, float, float]:
        """Wall-local to world. ``u`` along, ``v`` across (± half thickness), ``w`` up."""
        # The plan normal (-dy, dx) becomes (-(-dy), -(dx)) under the Y-flip above,
        # i.e. x uses -dy and z uses -dx.
        px = sx + dx * u - dy * v
        pz = sz - dy * u - dx * v
        return (px, w, pz)

    def box(u0: float, u1: float, w0: float, w1: float, mesh: MeshBuffer) -> None:
        """A full-thickness box spanning ``[u0,u1] × [w0,w1]``, all six faces."""
        if u1 - u0 < 1e-6 or w1 - w0 < 1e-6:
            return
        # +v and -v faces (the two wall faces), wound outward.
        mesh.quad(world(u0, half_t, w0), world(u1, half_t, w0), world(u1, half_t, w1), world(u0, half_t, w1))
        mesh.quad(world(u1, -half_t, w0), world(u0, -half_t, w0), world(u0, -half_t, w1), world(u1, -half_t, w1))
        # End caps at u0 and u1 — these are the reveals beside an opening.
        mesh.quad(world(u0, -half_t, w0), world(u0, half_t, w0), world(u0, half_t, w1), world(u0, -half_t, w1))
        mesh.quad(world(u1, half_t, w0), world(u1, -half_t, w0), world(u1, -half_t, w1), world(u1, half_t, w1))
        # Top and bottom.
        mesh.quad(world(u0, -half_t, w1), world(u0, half_t, w1), world(u1, half_t, w1), world(u1, -half_t, w1))
        mesh.quad(world(u0, half_t, w0), world(u0, -half_t, w0), world(u1, -half_t, w0), world(u1, half_t, w0))

    spans = _opening_spans(wall, plan.openings_for_wall(wall.id), length)
    mesh = MeshBuffer()

    if not spans:
        box(0.0, length, 0.0, height, mesh)
    else:
        cursor = 0.0
        for s in spans:
            # Pier: full-height wall before this opening.
            box(cursor, s.u0, 0.0, height, mesh)
            # Apron below the sill (a window) and lintel above the head (both).
            if s.sill > 1e-6:
                box(s.u0, s.u1, 0.0, min(s.sill, height), mesh)
            if s.head < height - 1e-6:
                box(s.u0, s.u1, s.head, height, mesh)
            cursor = s.u1
        box(cursor, length, 0.0, height, mesh)

    kind = SurfaceKind.PARAPET if wall.kind in ("parapet", "railing") else SurfaceKind.WALL
    resolved = MaterialSlot.WALL_ACCENT.value if accent else slot
    if wall.kind == "railing":
        # A balcony guard in glass, not blockwork. It reads as a railing because it is
        # thin and transparent, so the balcony stays part of the elevation instead of
        # disappearing behind a solid wall.
        resolved = MaterialSlot.GLAZING.value
    elif wall.kind == "parapet":
        resolved = MaterialSlot.PARAPET.value
    elif wall.exterior:
        resolved = MaterialSlot.WALL_EXTERIOR.value if not accent else resolved

    out = mesh.to_surface(f"s-{wall.id}", kind, resolved, wall_id=wall.id)
    return [out] if out else []


# ---------------------------------------------------------------------------
# room floors, ceilings, skirting
# ---------------------------------------------------------------------------

def _to_xz(
    polygon: Sequence[tuple[float, float]], origin: tuple[float, float]
) -> list[tuple[float, float]]:
    ox, oy = origin
    return [((x - ox) * MM_TO_M, -(y - oy) * MM_TO_M) for x, y in polygon]


def _signed_area(pts: Sequence[tuple[float, float]]) -> float:
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def build_room_surfaces(
    room: Room, *, origin: tuple[float, float], cfg: BuildConfig
) -> list[Surface]:
    """Floor, ceiling and skirting for one room."""
    pts = _to_xz(room.polygon, origin)
    if len(pts) < 3:
        return []
    # Ensure counter-clockwise in the XZ plane so the floor's winding gives an
    # upward normal; a clockwise ring renders the floor invisible from above.
    if _signed_area(pts) < 0:
        pts = list(reversed(pts))

    out: list[Surface] = []
    y0 = room.floor_level_mm * MM_TO_M

    floor = MeshBuffer()
    floor.polygon(pts, y0, up=True)
    s = floor.to_surface(f"s-{room.id}-floor", SurfaceKind.FLOOR, floor_slot_for(room.kind), room_id=room.id)
    if s:
        out.append(s)

    # Outdoor rooms get no ceiling — a balcony with a ceiling plane is a room.
    if cfg.generate_ceilings and not room.kind.is_outdoor:
        ceil = MeshBuffer()
        ceil.polygon(pts, y0 + room.ceiling_height_mm * MM_TO_M, up=False)
        s = ceil.to_surface(f"s-{room.id}-ceil", SurfaceKind.CEILING, MaterialSlot.CEILING.value, room_id=room.id)
        if s:
            out.append(s)

    if cfg.generate_skirting and not room.kind.is_outdoor and not room.kind.is_wet:
        sk = MeshBuffer()
        h = cfg.skirting_height_mm * MM_TO_M
        proud = cfg.skirting_proud_mm * MM_TO_M
        n = len(pts)
        for i in range(n):
            ax, az = pts[i]
            bx, bz = pts[(i + 1) % n]
            ex, ez = bx - ax, bz - az
            L = math.hypot(ex, ez)
            if L < 0.05:
                continue
            # Inward normal for a CCW ring in XZ.
            nx, nz = ez / L, -ex / L
            a = (ax + nx * proud, y0, az + nz * proud)
            b = (bx + nx * proud, y0, bz + nz * proud)
            sk.quad(a, b, (b[0], y0 + h, b[2]), (a[0], y0 + h, a[2]))
        s = sk.to_surface(f"s-{room.id}-skirt", SurfaceKind.SKIRTING, MaterialSlot.SKIRTING.value, room_id=room.id)
        if s:
            out.append(s)

    return out
