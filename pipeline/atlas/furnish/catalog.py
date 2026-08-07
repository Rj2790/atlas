"""Dimensioned furniture catalogue.

Every item is *parametric*: it is defined by real dimensions and generates its own
geometry, rather than referencing a downloaded mesh. That is a deliberate choice
with three concrete payoffs:

**Dimensional truth.** The goal is furniture that matches the dimensions on the
drawing. A downloaded sofa model is whatever size its author made it, usually
unstated and often wrong by a factor of ten; you then guess a scale factor and the
apartment fills with subtly-wrong furniture. A parametric 2.1 m sofa is 2.1 m
because 2.1 is in its definition. Sizes here are real: a queen bed is 1524 mm wide
because that is what a queen mattress is, a kitchen counter is 900 mm high and
600 mm deep because that is the standard, a dining chair needs 750 mm of pull-out
behind it because a person does.

**Licensing.** Generated geometry has no provenance problem. Objaverse and
3D-FUTURE are per-object licensed, which is a real obstacle to shipping
commercially; every asset would need auditing.

**It fits in a browser.** A furnished flat is a few hundred KB of boxes rather than
tens of MB of scanned meshes, so the configurator stays responsive while swapping.

The trade-off is honest: these read as well-proportioned massing models, not as
photographed products. So ``FurnitureItem.external_model`` exists — set it and the
build loads that glTF instead, at the item's declared dimensions. Parametric
geometry is the floor of quality, not the ceiling.

Each item also carries the placement knowledge that makes auto-layout work:
what it must back onto, how much clearance it needs and on which side, and what
it wants to face.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ..planir import RoomKind
from ..sceneir import MaterialSlot

__all__ = [
    "Anchor",
    "Part",
    "FurnitureItem",
    "CATALOG",
    "items_for_room",
    "get_item",
]


class Anchor(StrEnum):
    """How an item wants to sit in a room. Drives the layout solver."""

    WALL_BACK = "wall_back"
    """Back against a wall: beds, sofas, wardrobes, counters."""

    WALL_FACE = "wall_face"
    """Mounted on the wall face: TVs, mirrors, wall lights."""

    CORNER = "corner"
    """Into a corner: plants, floor lamps, corner chairs."""

    FREE = "free"
    """Free in the room, usually related to another item: coffee tables, rugs."""

    CENTRE = "centre"
    """Centred in the room: dining tables, islands."""

    RELATIVE = "relative"
    """Positioned relative to a parent item — see ``relative_to``."""


@dataclass(frozen=True, slots=True)
class Part:
    """One box of an item's geometry, in local millimetres.

    Local frame: origin at the centre of the footprint, ``+x`` right, ``+z``
    *forward* (the direction the item faces), ``y`` up from the floor. Keeping a
    single convention means the solver only ever has to decide a position and a
    yaw.
    """

    x: float
    y: float
    z: float
    """Centre of the box."""

    w: float
    h: float
    d: float
    """Size along x, y, z."""

    slot: str = MaterialSlot.WOOD.value


@dataclass(slots=True)
class FurnitureItem:
    """A catalogue entry: real dimensions, geometry, and placement rules."""

    id: str
    name: str
    category: str
    width_mm: float
    depth_mm: float
    height_mm: float
    parts: list[Part]
    anchor: Anchor = Anchor.FREE
    rooms: tuple[RoomKind, ...] = ()
    """Room kinds this item belongs in. Keeps a WC out of the living room."""

    clearance_front_mm: float = 0.0
    clearance_back_mm: float = 0.0
    clearance_sides_mm: float = 0.0
    """Free space the item needs to be usable. A wardrobe with no clearance in
    front cannot be opened; a bed with none at the side cannot be got into."""

    priority: int = 50
    """Placed in descending priority. The bed goes in before the side table,
    because the room is *for* the bed and the side table can adapt."""

    min_scale: float = 1.0
    """How far this item may be shrunk to fit a tight room, as a fraction of its
    catalogue size. 1.0 means it does not shrink.

    Real furniture comes in sizes: a 1200 mm desk and a 900 mm desk are both desks, and a
    compact room gets the smaller one rather than no desk at all. Nine of seventeen
    placement failures on the demo flat were bedroom items — desks, chairs, bedside tables —
    rejected outright in rooms that would comfortably take a smaller version.

    Only for items that genuinely come smaller. Sanitaryware does not: a WC has fixed
    connections and a 700 mm shower tray is not a shower. Nor does a bed frame below the
    mattress sizes the catalogue already offers as separate entries."""

    max_count: int = 1
    relative_to: str | None = None
    relative_offset_mm: tuple[float, float] = (0.0, 0.0)
    faces: str = "room"
    """``"room"`` to face the room centre, ``"wall"`` to face its host wall."""

    external_model: str | None = None
    """Optional glTF path. Loaded at the declared dimensions, replacing ``parts``."""

    min_room_area_m2: float = 0.0
    """Do not place unless the room is at least this big — the difference between
    furnishing a room and cramming it."""

    tags: tuple[str, ...] = ()
    notes: str = ""


def _box(w: float, h: float, d: float, slot: str, y0: float = 0.0) -> Part:
    """A box resting with its base at ``y0``, centred in plan."""
    return Part(0.0, y0 + h / 2.0, 0.0, w, h, d, slot)


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------

def _bed(mid: str, name: str, w: float, l: float, *, priority: int = 95, min_area: float = 7.0) -> FurnitureItem:
    """A platform bed: base, mattress, headboard, two pillows.

    Heights are real: a 250 mm base plus a 200 mm mattress puts the sleeping
    surface at 450 mm, which is what a bed actually is. Modelling a bed as one
    600 mm box is the sort of shortcut that makes a render feel wrong without the
    viewer being able to say why.
    """
    base_h, matt_h, head_h = 220.0, 210.0, 980.0
    pillow_w = min(520.0, w / 2 - 70)
    top = base_h + matt_h
    # Layered rather than one slab. A bed is read from its layers — the plinth set
    # back in shadow, the mattress, the duvet turned down over it, the pillows
    # standing proud — and each of those is a silhouette edge the eye uses to judge
    # that the object is real. A single 600 mm box reads as a block.
    parts = [
        # Plinth, inset so the bed appears to float and casts a shadow gap.
        Part(0, 60, 0, w - 120, 120, l - 120, MaterialSlot.WOOD.value),
        Part(0, base_h - 45, 0, w, 130, l, MaterialSlot.WOOD.value),
        # Mattress.
        Part(0, base_h + matt_h / 2, 20, w - 70, matt_h, l - 90, MaterialSlot.FABRIC.value),
        # Duvet over the lower two thirds, slightly wider than the mattress so it
        # drapes, plus a turned-down fold at its head.
        Part(0, top + 55, l * 0.16, w - 20, 110, l * 0.62, MaterialSlot.FABRIC.value),
        Part(0, top + 95, -l * 0.16, w - 20, 70, l * 0.10, MaterialSlot.FABRIC.value),
        # Headboard with a shadow reveal above the mattress line.
        Part(0, head_h / 2 + 120, -l / 2 + 45, w, head_h - 120, 90, MaterialSlot.FABRIC.value),
        Part(0, 130, -l / 2 + 25, w, 60, 40, MaterialSlot.WOOD.value),
    ]
    # Pillows: two stacked pairs, tilted back against the headboard.
    for sx in (-1, 1):
        parts.append(
            Part(sx * w / 4, top + 75, -l / 2 + 300, pillow_w, 150, 360, MaterialSlot.FABRIC.value)
        )
        parts.append(
            Part(sx * w / 4, top + 195, -l / 2 + 250, pillow_w - 60, 110, 300, MaterialSlot.FABRIC.value)
        )
    # Folded throw across the foot.
    parts.append(Part(0, top + 130, l / 2 - 260, w - 40, 70, 380, MaterialSlot.FABRIC.value))

    return FurnitureItem(
        id=mid,
        name=name,
        category="bed",
        width_mm=w,
        depth_mm=l,
        height_mm=head_h,
        parts=parts,
        anchor=Anchor.WALL_BACK,
        rooms=(RoomKind.BEDROOM, RoomKind.MASTER_BEDROOM, RoomKind.KIDS_BEDROOM),
        clearance_front_mm=700.0,
        clearance_sides_mm=550.0,
        priority=priority,
        min_room_area_m2=min_area,
        tags=("sleeping",),
        notes="Headboard wall becomes the room's accent wall.",
    )


def _sofa(mid: str, name: str, w: float, *, seats: int, priority: int = 90) -> FurnitureItem:
    d, seat_h, back_h, arm_h = 900.0, 430.0, 790.0, 620.0
    arm_w = 160.0
    inner = w - 2 * arm_w
    parts = [
        # Plinth set back under the body, so the sofa reads as sitting on legs.
        Part(0, 55, 0, w - 180, 110, d - 160, MaterialSlot.WOOD.value),
        Part(0, 200, 0, w, 180, d, MaterialSlot.FABRIC.value),
        # Back, tilted read via two stacked bands rather than one slab.
        Part(0, back_h / 2 + 40, -d / 2 + 120, w, back_h - 80, 200, MaterialSlot.FABRIC.value),
        # Arms, with a proud top roll.
        Part(-w / 2 + arm_w / 2, arm_h / 2, 0, arm_w, arm_h, d, MaterialSlot.FABRIC.value),
        Part(w / 2 - arm_w / 2, arm_h / 2, 0, arm_w, arm_h, d, MaterialSlot.FABRIC.value),
        Part(-w / 2 + arm_w / 2, arm_h + 25, 10, arm_w + 20, 60, d - 40, MaterialSlot.FABRIC.value),
        Part(w / 2 - arm_w / 2, arm_h + 25, 10, arm_w + 20, 60, d - 40, MaterialSlot.FABRIC.value),
    ]
    # One seat cushion and one back cushion per seat, gapped, so a 3-seater reads as
    # a 3-seater rather than as a bench.
    pitch = inner / seats
    for i in range(seats):
        cx = -w / 2 + arm_w + (i + 0.5) * pitch
        parts.append(
            Part(cx, seat_h - 25, 70, pitch - 30, 150, d - 300, MaterialSlot.FABRIC.value)
        )
        parts.append(
            Part(cx, seat_h + 210, -d / 2 + 265, pitch - 40, 300, 180, MaterialSlot.FABRIC.value)
        )
    # A pair of scatter cushions, offset, to break the symmetry.
    parts.append(Part(-inner / 2 + 120, seat_h + 190, -d / 2 + 380, 340, 340, 110, MaterialSlot.FABRIC.value))
    parts.append(Part(inner / 2 - 140, seat_h + 175, -d / 2 + 390, 300, 300, 100, MaterialSlot.FABRIC.value))
    return FurnitureItem(
        id=mid,
        name=name,
        category="sofa",
        width_mm=w,
        depth_mm=d,
        height_mm=back_h,
        parts=parts,
        anchor=Anchor.WALL_BACK,
        rooms=(RoomKind.LIVING,),
        clearance_front_mm=900.0,
        clearance_sides_mm=200.0,
        priority=priority,
        min_room_area_m2=9.0,
        tags=("seating",),
    )


def _wardrobe(mid: str, name: str, w: float, *, doors: int) -> FurnitureItem:
    d, h = 600.0, 2400.0
    parts = [Part(0, h / 2, 0, w, h, d, MaterialSlot.JOINERY.value)]
    # Door reveals: thin proud panels with gaps, so the front reads as joinery
    # rather than as one slab.
    leaf = w / doors
    for i in range(doors):
        cx = -w / 2 + (i + 0.5) * leaf
        parts.append(
            Part(cx, h / 2, d / 2 + 9, leaf - 14, h - 28, 18, MaterialSlot.JOINERY.value)
        )
        parts.append(
            Part(cx + leaf / 2 - 60, 1200, d / 2 + 24, 16, 220, 16, MaterialSlot.METAL.value)
        )
    return FurnitureItem(
        id=mid,
        name=name,
        category="wardrobe",
        width_mm=w,
        depth_mm=d,
        height_mm=h,
        parts=parts,
        anchor=Anchor.WALL_BACK,
        rooms=(RoomKind.BEDROOM, RoomKind.MASTER_BEDROOM, RoomKind.KIDS_BEDROOM, RoomKind.WARDROBE),
        clearance_front_mm=750.0,
        priority=85,
        min_room_area_m2=6.0,
        tags=("storage",),
        notes="750 mm front clearance is the door swing; less and it cannot open.",
    )


def _dining(mid: str, name: str, seats: int) -> FurnitureItem:
    per_side = max(1, seats // 2)
    w = max(900.0, per_side * 600.0)
    d, h, top = 900.0, 750.0, 40.0
    parts = [Part(0, h - top / 2, 0, w, top, d, MaterialSlot.WOOD.value)]
    for sx in (-1, 1):
        for sz in (-1, 1):
            parts.append(
                Part(sx * (w / 2 - 90), (h - top) / 2, sz * (d / 2 - 90), 70, h - top, 70, MaterialSlot.WOOD.value)
            )
    # Chairs, modelled simply but placed correctly: 600 mm pitch is the real
    # spacing, and it is what determines whether the table fits the room.
    for i in range(per_side):
        cx = -w / 2 + (i + 0.5) * (w / per_side)
        for sz in (-1, 1):
            cz = sz * (d / 2 + 260)
            parts.append(Part(cx, 230, cz, 440, 60, 440, MaterialSlot.WOOD.value))
            for dx in (-1, 1):
                for dz in (-1, 1):
                    parts.append(
                        Part(cx + dx * 180, 115, cz + dz * 180, 40, 230, 40, MaterialSlot.WOOD.value)
                    )
            parts.append(Part(cx, 560, cz - sz * 200, 440, 600, 45, MaterialSlot.FABRIC.value))
    return FurnitureItem(
        id=mid,
        name=name,
        category="dining_table",
        width_mm=w,
        depth_mm=d + 1040,
        height_mm=h,
        parts=parts,
        anchor=Anchor.CENTRE,
        rooms=(RoomKind.DINING, RoomKind.LIVING),
        clearance_sides_mm=750.0,
        priority=92,
        min_room_area_m2=6.5,
        tags=("dining",),
        notes="Footprint includes 520 mm of chair pull-out on each side.",
    )


def _counter(mid: str, name: str, w: float, *, sink: bool, hob: bool) -> FurnitureItem:
    d, h, top = 600.0, 900.0, 40.0
    parts = [
        Part(0, (h - top) / 2, 0, w, h - top, d, MaterialSlot.JOINERY.value),
        Part(0, h - top / 2, 0, w, top, d + 20, MaterialSlot.COUNTER.value),
        # Upstand where the counter meets the wall — stops water tracking behind,
        # and its absence is instantly readable as an unbuilt model.
        Part(0, h + 60, -d / 2 + 10, w, 120, 20, MaterialSlot.COUNTER.value),
    ]
    if sink:
        parts.append(Part(-w / 4, h - top - 90, 0, 500, 180, 400, MaterialSlot.METAL.value))
        parts.append(Part(-w / 4, h + 150, -d / 2 + 80, 40, 300, 40, MaterialSlot.METAL.value))
    if hob:
        parts.append(Part(w / 4, h + 10, 0, 580, 20, 500, MaterialSlot.METAL.value))
    # Overhead units, set at 1500 mm — the standard gap above a worktop.
    parts.append(Part(0, 1500 + 350, -d / 2 + 175, w, 700, 350, MaterialSlot.JOINERY.value))
    return FurnitureItem(
        id=mid,
        name=name,
        category="counter",
        width_mm=w,
        depth_mm=d,
        height_mm=2200.0,
        parts=parts,
        anchor=Anchor.WALL_BACK,
        rooms=(RoomKind.KITCHEN,),
        clearance_front_mm=900.0,
        priority=98,
        tags=("kitchen",),
        notes="900 mm front clearance is the minimum galley working width.",
    )


def _wc() -> FurnitureItem:
    return FurnitureItem(
        id="wc-wall-hung",
        name="Wall-hung WC",
        category="wc",
        width_mm=380.0,
        depth_mm=560.0,
        height_mm=1000.0,
        parts=[
            Part(0, 500, -230, 500, 1000, 200, MaterialSlot.WALL_WET.value),
            Part(0, 400, 30, 380, 160, 520, MaterialSlot.COUNTER.value),
            Part(0, 490, 60, 360, 30, 440, MaterialSlot.COUNTER.value),
        ],
        anchor=Anchor.WALL_BACK,
        rooms=(RoomKind.BATHROOM, RoomKind.POWDER),
        clearance_front_mm=600.0,
        clearance_sides_mm=200.0,
        priority=99,
        tags=("sanitary",),
    )


def _basin() -> FurnitureItem:
    return FurnitureItem(
        id="basin-vanity",
        name="Vanity basin",
        category="basin",
        width_mm=750.0,
        depth_mm=480.0,
        height_mm=1100.0,
        parts=[
            Part(0, 400, 0, 750, 800, 480, MaterialSlot.JOINERY.value),
            Part(0, 820, 0, 780, 40, 500, MaterialSlot.COUNTER.value),
            Part(0, 870, 0, 480, 60, 340, MaterialSlot.COUNTER.value),
            Part(0, 960, -180, 40, 220, 40, MaterialSlot.METAL.value),
            Part(0, 1500, -230, 700, 900, 30, MaterialSlot.GLAZING.value),
        ],
        anchor=Anchor.WALL_BACK,
        rooms=(RoomKind.BATHROOM, RoomKind.POWDER),
        clearance_front_mm=600.0,
        priority=97,
        tags=("sanitary",),
    )


def _shower() -> FurnitureItem:
    return FurnitureItem(
        id="shower-enclosure",
        name="Shower enclosure",
        category="shower",
        width_mm=900.0,
        depth_mm=900.0,
        height_mm=2100.0,
        parts=[
            Part(0, 25, 0, 900, 50, 900, MaterialSlot.FLOOR_WET.value),
            Part(450, 1050, 0, 20, 2100, 900, MaterialSlot.GLAZING.value),
            Part(0, 1050, 450, 900, 2100, 20, MaterialSlot.GLAZING.value),
            Part(0, 2050, -380, 240, 30, 240, MaterialSlot.METAL.value),
        ],
        anchor=Anchor.CORNER,
        rooms=(RoomKind.BATHROOM,),
        clearance_front_mm=400.0,
        priority=96,
        min_room_area_m2=3.0,
        tags=("sanitary",),
    )


def _simple(
    mid: str,
    name: str,
    cat: str,
    w: float,
    d: float,
    h: float,
    slot: str,
    anchor: Anchor,
    rooms: tuple[RoomKind, ...],
    *,
    priority: int = 40,
    front: float = 0.0,
    min_area: float = 0.0,
    extra: list[Part] | None = None,
    faces: str = "room",
    tags: tuple[str, ...] = (),
) -> FurnitureItem:
    return FurnitureItem(
        id=mid,
        name=name,
        category=cat,
        width_mm=w,
        depth_mm=d,
        height_mm=h,
        parts=[_box(w, h, d, slot)] + (extra or []),
        anchor=anchor,
        rooms=rooms,
        clearance_front_mm=front,
        priority=priority,
        min_room_area_m2=min_area,
        faces=faces,
        tags=tags,
    )


def _curtains(mid: str, name: str, w: float, h: float) -> FurnitureItem:
    """A pair of full-height curtain panels with a track.

    Every published interior render has these, and their absence is one of the
    strongest tells that an image is a model rather than a room: a bare window reads
    as a hole. They also do real work in the image — vertical soft folds beside a hard
    rectangular opening, and a warm bounce surface next to the brightest thing in the
    frame.

    Folds are modelled as alternating slabs at two depths. Cheap, and enough to catch
    a gradient across the panel instead of rendering as a flat card.
    """
    panel = max(320.0, w * 0.34)
    parts = [
        # Track, held off the ceiling.
        Part(0, h - 40, 0, w + 300, 45, 90, MaterialSlot.METAL.value),
    ]
    for side in (-1, 1):
        cx = side * (w / 2 + 90)
        n = max(3, int(panel / 130))
        for i in range(n):
            fx = cx + (i - (n - 1) / 2) * (panel / n)
            parts.append(
                Part(fx, (h - 90) / 2, 30 if i % 2 else -10, panel / n + 22, h - 130, 80,
                     MaterialSlot.FABRIC.value)
            )
    return FurnitureItem(
        id=mid, name=name, category="curtain",
        width_mm=w + 480, depth_mm=140, height_mm=h,
        parts=parts, anchor=Anchor.WALL_FACE, rooms=(), priority=10,
        tags=("dressing", "soft"),
        notes="Placed against a window by the dressing pass, not by the layout solver.",
    )


def _wall_art(mid: str, name: str, w: float, h: float, panels: int) -> FurnitureItem:
    """Framed panels for a room's accent wall.

    Art is what tells the eye a room is inhabited. A deep frame with the canvas set
    back catches a shadow line, which is what stops it reading as a decal.
    """
    parts: list[Part] = []
    pitch = w / panels
    for i in range(panels):
        cx = -w / 2 + (i + 0.5) * pitch
        pw = pitch - 60
        parts.append(Part(cx, h / 2, 0, pw, h, 45, MaterialSlot.JOINERY.value))
        parts.append(Part(cx, h / 2, 14, pw - 70, h - 70, 20, MaterialSlot.FABRIC.value))
    return FurnitureItem(
        id=mid, name=name, category="wall_art",
        width_mm=w, depth_mm=50, height_mm=h,
        parts=parts, anchor=Anchor.WALL_FACE, rooms=(), priority=8,
        tags=("dressing", "art"),
        notes="Hung on the room's accent wall by the dressing pass.",
    )


def _build_catalog() -> dict[str, FurnitureItem]:
    items: list[FurnitureItem] = [
        # --- sleeping. Mattress sizes are the real standards. ---
        _bed("bed-king", "King bed (1830×2000)", 1830, 2000, priority=95, min_area=11.0),
        _bed("bed-queen", "Queen bed (1524×2000)", 1524, 2000, priority=95, min_area=8.5),
        _bed("bed-single", "Single bed (900×1900)", 900, 1900, priority=94, min_area=5.0),
        _wardrobe("wardrobe-3d", "Wardrobe, 3 door (1800)", 1800, doors=3),
        _wardrobe("wardrobe-2d", "Wardrobe, 2 door (1200)", 1200, doors=2),
        _simple(
            "bedside", "Bedside table", "side_table", 450, 400, 550,
            MaterialSlot.WOOD.value, Anchor.RELATIVE,
            (RoomKind.BEDROOM, RoomKind.MASTER_BEDROOM, RoomKind.KIDS_BEDROOM),
            priority=60, tags=("storage",),
        ),

        # --- living ---
        _sofa("sofa-3", "3-seat sofa (2100)", 2100, seats=3),
        _sofa("sofa-2", "2-seat sofa (1600)", 1600, seats=2, priority=88),
        _simple(
            "coffee-table", "Coffee table", "coffee_table", 1100, 600, 380,
            MaterialSlot.WOOD.value, Anchor.RELATIVE, (RoomKind.LIVING,),
            priority=70, min_area=9.0, tags=("surface",),
        ),
        _simple(
            "tv-unit", "TV console", "tv_unit", 1600, 400, 450,
            MaterialSlot.JOINERY.value, Anchor.WALL_BACK, (RoomKind.LIVING,),
            priority=80, min_area=9.0, tags=("media",),
            extra=[Part(0, 1150, -140, 1300, 750, 60, MaterialSlot.METAL.value)],
        ),
        _simple(
            "armchair", "Armchair", "armchair", 800, 800, 780,
            MaterialSlot.FABRIC.value, Anchor.CORNER, (RoomKind.LIVING, RoomKind.BEDROOM),
            priority=55, min_area=13.0, tags=("seating",),
        ),
        _simple(
            "rug-living", "Area rug", "rug", 2400, 1700, 14,
            MaterialSlot.FABRIC.value, Anchor.RELATIVE, (RoomKind.LIVING,),
            priority=30, min_area=11.0, tags=("soft",),
        ),

        # --- dining ---
        _dining("dining-6", "Dining table, 6 seat", 6),
        _dining("dining-4", "Dining table, 4 seat", 4),

        # --- kitchen ---
        _counter("counter-run", "Kitchen counter run", 2400, sink=True, hob=True),
        _simple(
            "fridge", "Refrigerator", "fridge", 700, 700, 1800,
            MaterialSlot.METAL.value, Anchor.WALL_BACK, (RoomKind.KITCHEN,),
            priority=90, front=700, tags=("appliance",),
        ),

        # --- bathroom ---
        _wc(),
        _basin(),
        _shower(),

        # --- utility ---
        _simple(
            "washing-machine", "Washing machine", "washing_machine", 600, 600, 850,
            MaterialSlot.METAL.value, Anchor.WALL_BACK,
            (RoomKind.UTILITY, RoomKind.DRY_BALCONY, RoomKind.BALCONY),
            priority=88, front=600, tags=("appliance",),
        ),

        # --- study ---
        _simple(
            "desk", "Desk", "desk", 1400, 700, 750,
            MaterialSlot.WOOD.value, Anchor.WALL_BACK, (RoomKind.STUDY, RoomKind.BEDROOM),
            priority=75, front=900, min_area=7.0, tags=("work",),
        ),
        _simple(
            "task-chair", "Task chair", "chair", 600, 600, 950,
            MaterialSlot.FABRIC.value, Anchor.RELATIVE, (RoomKind.STUDY, RoomKind.BEDROOM),
            priority=50, tags=("seating",),
        ),

        # --- outdoor ---
        _simple(
            "outdoor-chair", "Balcony chair", "outdoor_chair", 650, 650, 750,
            MaterialSlot.METAL.value, Anchor.CORNER, (RoomKind.BALCONY, RoomKind.TERRACE),
            priority=45, min_area=2.2, tags=("outdoor",),
        ),
        _simple(
            "outdoor-table", "Balcony table", "outdoor_table", 600, 600, 450,
            MaterialSlot.METAL.value, Anchor.RELATIVE, (RoomKind.BALCONY, RoomKind.TERRACE),
            priority=40, min_area=3.0, tags=("outdoor",),
        ),

        # --- dressing. Placed by the dressing pass against windows and accent walls
        # rather than by the layout solver, because their position is determined by
        # the architecture, not by circulation. ---
        _curtains("curtain-wide", "Curtain pair, wide", 2600.0, 2750.0),
        _curtains("curtain-std", "Curtain pair", 1600.0, 2750.0),
        _wall_art("art-triptych", "Framed triptych", 1650.0, 700.0, 3),
        _wall_art("art-single", "Framed canvas", 900.0, 1150.0, 1),

        # --- greenery. Cheap to place and disproportionately effective: a plant
        # breaks the box-like silhouette of generated furniture and gives the eye
        # organic detail to land on. ---
        _simple(
            "plant-tall", "Tall planter", "plant", 500, 500, 1500,
            MaterialSlot.WOOD.value, Anchor.CORNER,
            (RoomKind.LIVING, RoomKind.DINING, RoomKind.BEDROOM, RoomKind.LOBBY, RoomKind.BALCONY),
            priority=25, min_area=7.0, tags=("greenery",),
        ),
    ]
    # A handful of items are backed by a real CC0 mesh instead of generated boxes.
    # Only the small, detailed, organically-shaped ones: their silhouette is what
    # reads as "real", and none of them has to hit an exact dimension to be correct.
    # The dimension-critical pieces (bed, wardrobe, counter, sanitaryware) stay
    # parametric on purpose — see atlas.build.models for the full reasoning.
    from ..build.models import POLYHAVEN_MODELS

    by_id = {i.id: i for i in items}
    for item_id, slug in POLYHAVEN_MODELS.items():
        if item_id in by_id:
            by_id[item_id].external_model = slug

    by_id["bedside"].max_count = 2
    by_id["outdoor-chair"].max_count = 2
    by_id["armchair"].max_count = 2

    # How far each item may shrink to fit a tight room. These are the sizes the thing
    # genuinely comes in, not arbitrary squashing:
    #
    #   desk          1400 -> 980 mm     a 1 m desk is a normal desk
    #   wardrobe      1800 -> 1260 mm    a two-door instead of a three-door
    #   coffee table  1100 ->  770 mm    small-room coffee tables are this size
    #   rug           2400 -> 1560 mm    rugs come in every size
    #   bedside/chair /planter/outdoor    all made small
    #
    # Deliberately absent: sanitaryware (a WC has fixed connections, and a 700 mm tray is
    # not a shower), the beds (the catalogue already lists king/queen/single separately, so
    # shrinking would duplicate that), and the kitchen counter run (it is sized to the wall
    # it fits, not to a range).
    for item_id, floor in {
        "desk": 0.70,
        "task-chair": 0.80,
        "bedside": 0.75,
        "wardrobe-3d": 0.70,
        "coffee-table": 0.70,
        "rug-living": 0.65,
        "plant-tall": 0.70,
        "outdoor-table": 0.75,
        "outdoor-chair": 0.80,
        "washing-machine": 0.90,
        "tv-unit": 0.75,
        "sofa-2": 0.85,
        "dining-4": 0.85,
    }.items():
        if item_id in by_id:
            by_id[item_id].min_scale = floor
    return by_id


CATALOG: dict[str, FurnitureItem] = _build_catalog()


def get_item(item_id: str) -> FurnitureItem | None:
    return CATALOG.get(item_id)


def items_for_room(kind: RoomKind, area_m2: float) -> list[FurnitureItem]:
    """Candidate items for a room, highest priority first.

    Filtered by both programme and area, so a 3 m² box room is not offered a king
    bed and a 20 m² living room is not left with only a rug.
    """
    out = [
        i
        for i in CATALOG.values()
        if kind in i.rooms and area_m2 >= i.min_room_area_m2
    ]
    out.sort(key=lambda i: -i.priority)
    return out
