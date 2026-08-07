"""PlanIR — the canonical 2D representation of an ingested floor plan.

PlanIR is the seam in the architecture. Everything upstream (DWG, DXF, raster
image via a vision model) reduces to PlanIR; everything downstream (3D build,
furnishing, rendering, the web configurator) reads only PlanIR. That means a new
input format is a new extractor and nothing else, and the 3D build never has to
know whether its input was measured from vector geometry or inferred from pixels.

Design commitments:

**Millimetres, always.** No unit field on geometry. Conversion happens once, in
the extractor, where the evidence for the correct scale lives.

**Walls are centrelines plus thickness, not rectangles.** A rectangle-per-wall
representation cannot express a T-junction or a corner without either a gap or an
overlap, and repairing that later is guesswork. Centrelines join exactly, and the
solid geometry is generated from them at build time where mitring can be solved
properly.

**Confidence and provenance on every element.** A vision model's guess and a
surveyed CAD polyline are both valid inputs but they are not equally trustworthy.
Carrying ``confidence`` and ``source`` per element lets the UI show what to check,
and lets the build stage refuse to silently invent a building from weak evidence.

**Stated intent is first-class.** ``Room.stated_dimension`` (the printed
``11'0"X13'6"``) is kept separately from the measured polygon. When they disagree
the discrepancy is a fact about the drawing worth surfacing, not an error to
average away.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal

from .units import Dimension, mm_to_sqft

__all__ = [
    "SCHEMA_VERSION",
    "RoomKind",
    "OpeningKind",
    "Source",
    "Point",
    "Polygon",
    "Wall",
    "Opening",
    "Room",
    "FixtureHint",
    "Level",
    "PlanUnit",
    "PlanIR",
    "ROOM_KIND_ALIASES",
    "classify_room_name",
]

SCHEMA_VERSION = "1.0"

Point = tuple[float, float]
Polygon = list[Point]


class RoomKind(StrEnum):
    """Functional room classification.

    Drives furnishing (a BEDROOM gets a bed), material defaults (a TOILET gets
    tile, not oak), and lighting. Kept deliberately coarse — finer distinctions
    belong in the design scheme, not the geometry.
    """

    LIVING = "living"
    DINING = "dining"
    KITCHEN = "kitchen"
    BEDROOM = "bedroom"
    MASTER_BEDROOM = "master_bedroom"
    KIDS_BEDROOM = "kids_bedroom"
    BATHROOM = "bathroom"
    POWDER = "powder"
    BALCONY = "balcony"
    DRY_BALCONY = "dry_balcony"
    UTILITY = "utility"
    STORE = "store"
    WARDROBE = "wardrobe"
    ENTRANCE = "entrance"
    """The front door and the space inside it. Distinct from a lobby on purpose: it
    carries the dwelling's *entrance door*, which is what makes the flat enterable at
    all, and it is finished differently — hard floor, no soft furnishing."""

    LOBBY = "lobby"
    CORRIDOR = "corridor"
    STAIRS = "stairs"
    LIFT = "lift"
    DUCT = "duct"
    TERRACE = "terrace"
    STUDY = "study"
    PUJA = "puja"
    UNKNOWN = "unknown"

    @property
    def is_habitable(self) -> bool:
        """Habitable rooms get full furnishing and finish treatment."""
        return self in _HABITABLE

    @property
    def is_wet(self) -> bool:
        """Wet areas default to tiled floors and get a floor drain."""
        return self in _WET

    @property
    def is_circulation(self) -> bool:
        """Entrance, lobby, corridor — hard floors, no soft furnishing, always a route."""
        return self in _CIRCULATION

    @property
    def is_outdoor(self) -> bool:
        """Outdoor areas get no ceiling and exterior-grade finishes."""
        return self in _OUTDOOR


_HABITABLE = frozenset(
    {
        RoomKind.LIVING,
        RoomKind.DINING,
        RoomKind.KITCHEN,
        RoomKind.BEDROOM,
        RoomKind.MASTER_BEDROOM,
        RoomKind.KIDS_BEDROOM,
        RoomKind.STUDY,
    }
)
_WET = frozenset({RoomKind.BATHROOM, RoomKind.POWDER, RoomKind.UTILITY, RoomKind.DRY_BALCONY})
_CIRCULATION = frozenset({RoomKind.ENTRANCE, RoomKind.LOBBY, RoomKind.CORRIDOR})
_OUTDOOR = frozenset({RoomKind.BALCONY, RoomKind.DRY_BALCONY, RoomKind.TERRACE})


# Room label vocabulary as it actually appears on drawings. Indian residential
# plans use a specific register ("W.I.W" for walk-in wardrobe, "DRY BAL" for the
# service balcony, "C.TOILET" for the common bathroom, "PUJA" for the prayer
# niche) that a generic English mapping misses entirely.
ROOM_KIND_ALIASES: dict[str, RoomKind] = {
    "living": RoomKind.LIVING,
    "living room": RoomKind.LIVING,
    "hall": RoomKind.LIVING,
    "drawing": RoomKind.LIVING,
    "lounge": RoomKind.LIVING,
    "living/dining": RoomKind.LIVING,
    "dining": RoomKind.DINING,
    "dining room": RoomKind.DINING,
    "kitchen": RoomKind.KITCHEN,
    "kit": RoomKind.KITCHEN,
    "pantry": RoomKind.KITCHEN,
    "bedroom": RoomKind.BEDROOM,
    "bed room": RoomKind.BEDROOM,
    "bed": RoomKind.BEDROOM,
    "br": RoomKind.BEDROOM,
    "master bedroom": RoomKind.MASTER_BEDROOM,
    "master bed": RoomKind.MASTER_BEDROOM,
    "m.bedroom": RoomKind.MASTER_BEDROOM,
    "mbr": RoomKind.MASTER_BEDROOM,
    "kids bedroom": RoomKind.KIDS_BEDROOM,
    "children bedroom": RoomKind.KIDS_BEDROOM,
    "toilet": RoomKind.BATHROOM,
    "bath": RoomKind.BATHROOM,
    "bathroom": RoomKind.BATHROOM,
    "wc": RoomKind.POWDER,
    "c.toilet": RoomKind.BATHROOM,
    "common toilet": RoomKind.BATHROOM,
    "att.toilet": RoomKind.BATHROOM,
    "powder": RoomKind.POWDER,
    "powder room": RoomKind.POWDER,
    "balcony": RoomKind.BALCONY,
    "bal": RoomKind.BALCONY,
    "sit out": RoomKind.BALCONY,
    "sitout": RoomKind.BALCONY,
    "deck": RoomKind.BALCONY,
    "dry balcony": RoomKind.DRY_BALCONY,
    "dry bal": RoomKind.DRY_BALCONY,
    "dry.bal": RoomKind.DRY_BALCONY,
    "service balcony": RoomKind.DRY_BALCONY,
    "utility": RoomKind.UTILITY,
    "wash": RoomKind.UTILITY,
    "laundry": RoomKind.UTILITY,
    "store": RoomKind.STORE,
    "storage": RoomKind.STORE,
    "wardrobe": RoomKind.WARDROBE,
    "w.i.w": RoomKind.WARDROBE,
    "wiw": RoomKind.WARDROBE,
    "walk in wardrobe": RoomKind.WARDROBE,
    "walk-in closet": RoomKind.WARDROBE,
    "closet": RoomKind.WARDROBE,
    "dress": RoomKind.WARDROBE,
    "dressing": RoomKind.WARDROBE,
    "lobby": RoomKind.LOBBY,
    "lift lobby": RoomKind.LOBBY,
    "ent. lobby": RoomKind.ENTRANCE,
    "ent.lobby": RoomKind.ENTRANCE,
    "ent lobby": RoomKind.ENTRANCE,
    "entrance lobby": RoomKind.ENTRANCE,
    "entrance": RoomKind.ENTRANCE,
    "entry": RoomKind.ENTRANCE,
    "foyer": RoomKind.ENTRANCE,
    "vestibule": RoomKind.ENTRANCE,
    "passage": RoomKind.CORRIDOR,
    "corridor": RoomKind.CORRIDOR,
    "stair": RoomKind.STAIRS,
    "stairs": RoomKind.STAIRS,
    "staircase": RoomKind.STAIRS,
    "lift": RoomKind.LIFT,
    "elevator": RoomKind.LIFT,
    "lift lobby": RoomKind.LOBBY,
    "duct": RoomKind.DUCT,
    "shaft": RoomKind.DUCT,
    "terrace": RoomKind.TERRACE,
    "study": RoomKind.STUDY,
    "office": RoomKind.STUDY,
    "puja": RoomKind.PUJA,
    "pooja": RoomKind.PUJA,
    "temple": RoomKind.PUJA,
}


def classify_room_name(label: str) -> RoomKind:
    """Map a drawing label to a :class:`RoomKind`.

    Matching is longest-alias-first so ``"MASTER BEDROOM"`` does not degrade to
    ``BEDROOM`` and ``"DRY BAL"`` does not degrade to ``BALCONY``.
    """
    if not label:
        return RoomKind.UNKNOWN
    # Use only the first line: labels are "BEDROOM\n11'0\"X13'6\"".
    text = label.strip().splitlines()[0]
    norm = " ".join(text.lower().replace("_", " ").split())
    norm = norm.strip(" .:-")

    if norm in ROOM_KIND_ALIASES:
        return ROOM_KIND_ALIASES[norm]

    # Substring match, longest alias first, so specific beats generic.
    for alias in sorted(ROOM_KIND_ALIASES, key=len, reverse=True):
        if alias in norm:
            return ROOM_KIND_ALIASES[alias]
    return RoomKind.UNKNOWN


class OpeningKind(StrEnum):
    DOOR = "door"
    SLIDING_DOOR = "sliding_door"
    DOUBLE_DOOR = "double_door"
    WINDOW = "window"
    FRENCH_WINDOW = "french_window"
    OPENING = "opening"
    """A cased opening with no leaf — an arch or a pass-through."""

    @property
    def is_door(self) -> bool:
        return self in (OpeningKind.DOOR, OpeningKind.SLIDING_DOOR, OpeningKind.DOUBLE_DOOR)


class Source(StrEnum):
    """Where an element's geometry came from, for trust and for the UI."""

    CAD_VECTOR = "cad_vector"
    """Measured from DWG/DXF geometry. Trustworthy to the millimetre."""

    CAD_TEXT = "cad_text"
    """Read from CAD annotation — the designer's stated intent."""

    VISION = "vision"
    """Inferred by a vision model from a raster image. Needs review."""

    INFERRED = "inferred"
    """Derived by Atlas (e.g. a ceiling implied by a floor). Deterministic."""

    USER = "user"
    """Edited by a human. Always wins."""


@dataclass(slots=True)
class Wall:
    """A wall segment as a centreline with thickness.

    Height is per-wall because parapets, half-walls and sill-height kitchen
    counters are all walls with a non-default height, and modelling them as
    something else means the 3D build has to special-case them.
    """

    id: str
    start: Point
    end: Point
    thickness_mm: float
    height_mm: float
    kind: Literal["structural", "partition", "parapet", "railing", "column"] = "partition"
    source: Source = Source.CAD_VECTOR
    confidence: float = 1.0
    exterior: bool = False
    """True when this wall faces outside — drives exterior material and the
    decision to place a window rather than a door."""

    @property
    def length_mm(self) -> float:
        return math.dist(self.start, self.end)

    @property
    def angle_rad(self) -> float:
        return math.atan2(self.end[1] - self.start[1], self.end[0] - self.start[0])

    @property
    def midpoint(self) -> Point:
        return ((self.start[0] + self.end[0]) / 2.0, (self.start[1] + self.end[1]) / 2.0)

    def direction(self) -> Point:
        """Unit vector along the wall; ``(1, 0)`` for a degenerate wall."""
        n = self.length_mm
        if n < 1e-9:
            return (1.0, 0.0)
        return ((self.end[0] - self.start[0]) / n, (self.end[1] - self.start[1]) / n)

    def normal(self) -> Point:
        """Unit vector perpendicular to the wall (left-hand side)."""
        dx, dy = self.direction()
        return (-dy, dx)

    def point_at(self, t: float) -> Point:
        """Point at normalised position ``t`` in [0, 1] along the centreline."""
        return (
            self.start[0] + (self.end[0] - self.start[0]) * t,
            self.start[1] + (self.end[1] - self.start[1]) * t,
        )


@dataclass(slots=True)
class Opening:
    """A door or window, positioned along a host wall.

    Anchoring to a wall by ``(wall_id, t)`` rather than by absolute position is
    what makes the model editable: move or restretch a wall and its openings
    follow instead of ending up floating in space.
    """

    id: str
    wall_id: str
    kind: OpeningKind
    t_center: float
    """Normalised position of the opening's centre along the wall, in [0, 1]."""

    width_mm: float
    height_mm: float
    sill_mm: float = 0.0
    """Height of the opening's bottom above the floor. 0 for a door."""

    swing: Literal["left", "right", "none"] = "none"
    swing_inward: bool = True
    source: Source = Source.CAD_VECTOR
    confidence: float = 1.0
    connects: tuple[str | None, str | None] = (None, None)
    """Room ids on either side, once resolved. Drives circulation and privacy."""


@dataclass(slots=True)
class FixtureHint:
    """A fixture or furniture item read off the drawing.

    Architects draw the sanitaryware and the kitchen counter, and that placement
    encodes plumbing and service constraints that must not be second-guessed. We
    keep these as *hints with real coordinates* so the furnishing stage can honour
    them instead of inventing a new bathroom layout.
    """

    id: str
    label: str
    footprint: Polygon
    room_id: str | None = None
    category: str = "unknown"
    source: Source = Source.CAD_VECTOR
    confidence: float = 1.0
    block_name: str | None = None
    """Originating CAD block name — often the most reliable category signal."""


@dataclass(slots=True)
class Room:
    """An enclosed space with a function, geometry, and stated intent."""

    id: str
    kind: RoomKind
    label: str
    polygon: Polygon
    """Room boundary in mm, counter-clockwise, at the *inner face* of walls."""

    centroid: Point
    stated_dimension: Dimension | None = None
    """The printed ``11'0"X13'6"``, when the drawing carries one."""

    ceiling_height_mm: float = 2900.0
    floor_level_mm: float = 0.0
    """Floor offset from the level datum. Balconies and wet areas step down."""

    source: Source = Source.CAD_VECTOR
    confidence: float = 1.0
    notes: list[str] = field(default_factory=list)

    @property
    def area_mm2(self) -> float:
        """Shoelace area of the polygon."""
        p = self.polygon
        if len(p) < 3:
            return 0.0
        s = 0.0
        for i in range(len(p)):
            x1, y1 = p[i]
            x2, y2 = p[(i + 1) % len(p)]
            s += x1 * y2 - x2 * y1
        return abs(s) / 2.0

    @property
    def area_sqft(self) -> float:
        return mm_to_sqft(self.area_mm2)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(min_x, min_y, max_x, max_y)`` of the polygon."""
        if not self.polygon:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return (min(xs), min(ys), max(xs), max(ys))

    def dimension_agreement(self) -> float | None:
        """How well the measured polygon matches the printed dimension, in [0, 1].

        Returns None when the drawing states no dimension. A value near 1 means
        the extraction agrees with what the designer wrote; a low value means one
        of the two is wrong and a human should look. Compared on area because the
        printed W×D is a bounding rectangle and real rooms are rarely rectangles.
        """
        if self.stated_dimension is None:
            return None
        stated = self.stated_dimension.area_mm2
        measured = self.area_mm2
        if stated <= 0 or measured <= 0:
            return 0.0
        return min(stated, measured) / max(stated, measured)


@dataclass(slots=True)
class Level:
    """One storey. Multi-storey is in the model from the start because retrofitting
    it means changing every coordinate in the system."""

    id: str
    name: str
    elevation_mm: float = 0.0
    height_mm: float = 2900.0
    slab_thickness_mm: float = 150.0


@dataclass(slots=True)
class PlanUnit:
    """One dwelling within the drawing.

    A production DWG is rarely one apartment — the demo file is a whole
    masterplan holding 20+ units across several towers. Ingest therefore returns
    a *set* of units and the caller chooses; a pipeline that assumes one plan per
    file cannot read real drawings at all.
    """

    id: str
    name: str
    """e.g. ``"3 BHK"`` read from the drawing, or ``"Unit 3"`` if unlabelled."""

    room_ids: list[str]
    wall_ids: list[str]
    stated_area_sqft: float | None = None
    """Declared carpet area, when printed. An independent check on extraction."""

    boundary: list[Point] = field(default_factory=list)
    """The declared carpet-area polygon itself, when the drawing provides one. This is
    the legal statement of the dwelling's extent, measured exact (±0.2%) against the
    printed figure — so the audit accounts every square millimetre inside it: floor,
    wall body, or unassigned leftover. Without it, area checks can only compare sums."""

    bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    confidence: float = 1.0

    @property
    def bedroom_count(self) -> int:
        """Parsed from an Indian ``N BHK`` designation when present."""
        import re

        m = re.search(r"(\d+(?:\.\d+)?)\s*BHK", self.name, re.IGNORECASE)
        return int(float(m.group(1))) if m else 0


@dataclass(slots=True)
class PlanIR:
    """A complete ingested plan: the contract between ingest and build."""

    schema_version: str = SCHEMA_VERSION
    source_file: str = ""
    source_kind: Literal["dwg", "dxf", "raster", "manual"] = "manual"
    levels: list[Level] = field(default_factory=lambda: [Level("L0", "Ground Floor")])
    walls: list[Wall] = field(default_factory=list)
    openings: list[Opening] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    fixtures: list[FixtureHint] = field(default_factory=list)
    units: list[PlanUnit] = field(default_factory=list)
    imperial_display: bool = False
    """Show feet-and-inches in the UI, because that is how the plan was drawn."""

    ingest_notes: list[str] = field(default_factory=list)
    """Human-readable trace of every decision ingest made — the scale it chose and
    why, what it repaired, what it could not classify. Shipping a black box that
    silently guesses the scale is how you get a 1.4 m door."""

    warnings: list[str] = field(default_factory=list)
    """Problems a human should resolve before trusting the 3D output."""

    def wall_by_id(self, wall_id: str) -> Wall | None:
        return next((w for w in self.walls if w.id == wall_id), None)

    def room_by_id(self, room_id: str) -> Room | None:
        return next((r for r in self.rooms if r.id == room_id), None)

    def openings_for_wall(self, wall_id: str) -> list[Opening]:
        return [o for o in self.openings if o.wall_id == wall_id]

    def subset(self, unit: PlanUnit) -> "PlanIR":
        """Extract a single dwelling as a standalone PlanIR.

        Coordinates are preserved rather than re-origined so the unit can still be
        related back to the source drawing; the build stage re-centres for export.
        """
        rooms = [r for r in self.rooms if r.id in set(unit.room_ids)]
        walls = [w for w in self.walls if w.id in set(unit.wall_ids)]
        wall_ids = {w.id for w in walls}
        room_ids = {r.id for r in rooms}
        return PlanIR(
            schema_version=self.schema_version,
            source_file=self.source_file,
            source_kind=self.source_kind,
            levels=list(self.levels),
            walls=walls,
            openings=[o for o in self.openings if o.wall_id in wall_ids],
            rooms=rooms,
            fixtures=[f for f in self.fixtures if f.room_id in room_ids or f.room_id is None],
            units=[unit],
            imperial_display=self.imperial_display,
            ingest_notes=list(self.ingest_notes),
            warnings=list(self.warnings),
        )

    def quality_report(self) -> dict[str, Any]:
        """Summarise how much the output should be trusted.

        Used by the CLI and the UI to decide whether to present the model as
        finished or as a draft needing review.
        """
        agreements = [
            a for a in (r.dimension_agreement() for r in self.rooms) if a is not None
        ]
        habitable = [r for r in self.rooms if r.kind.is_habitable]
        unknown = [r for r in self.rooms if r.kind is RoomKind.UNKNOWN]
        return {
            "rooms": len(self.rooms),
            "walls": len(self.walls),
            "openings": len(self.openings),
            "fixtures": len(self.fixtures),
            "units": len(self.units),
            "habitable_rooms": len(habitable),
            "unclassified_rooms": len(unknown),
            "rooms_with_stated_dimension": len(agreements),
            "mean_dimension_agreement": (
                sum(agreements) / len(agreements) if agreements else None
            ),
            "worst_dimension_agreement": min(agreements) if agreements else None,
            "total_area_sqft": round(sum(r.area_sqft for r in self.rooms), 1),
            "warnings": len(self.warnings),
        }

    def to_dict(self) -> dict[str, Any]:
        def encode(obj: Any) -> Any:
            if isinstance(obj, StrEnum):
                return obj.value
            if isinstance(obj, Dimension):
                return {
                    "width_mm": obj.width_mm,
                    "depth_mm": obj.depth_mm,
                    "source_text": obj.source_text,
                    "imperial": obj.imperial,
                }
            if isinstance(obj, dict):
                return {k: encode(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [encode(v) for v in obj]
            return obj

        return encode(asdict(self))

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanIR":
        def pt(v: Any) -> Point:
            return (float(v[0]), float(v[1]))

        def poly(v: Any) -> Polygon:
            return [pt(p) for p in v]

        def dim(v: Any) -> Dimension | None:
            if not v:
                return None
            return Dimension(
                float(v["width_mm"]),
                float(v["depth_mm"]),
                v.get("source_text", ""),
                bool(v.get("imperial", False)),
            )

        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            source_file=data.get("source_file", ""),
            source_kind=data.get("source_kind", "manual"),
            levels=[Level(**l) for l in data.get("levels", [])]
            or [Level("L0", "Ground Floor")],
            walls=[
                Wall(
                    id=w["id"],
                    start=pt(w["start"]),
                    end=pt(w["end"]),
                    thickness_mm=float(w["thickness_mm"]),
                    height_mm=float(w["height_mm"]),
                    kind=w.get("kind", "partition"),
                    source=Source(w.get("source", "cad_vector")),
                    confidence=float(w.get("confidence", 1.0)),
                    exterior=bool(w.get("exterior", False)),
                )
                for w in data.get("walls", [])
            ],
            openings=[
                Opening(
                    id=o["id"],
                    wall_id=o["wall_id"],
                    kind=OpeningKind(o["kind"]),
                    t_center=float(o["t_center"]),
                    width_mm=float(o["width_mm"]),
                    height_mm=float(o["height_mm"]),
                    sill_mm=float(o.get("sill_mm", 0.0)),
                    swing=o.get("swing", "none"),
                    swing_inward=bool(o.get("swing_inward", True)),
                    source=Source(o.get("source", "cad_vector")),
                    confidence=float(o.get("confidence", 1.0)),
                    connects=tuple(o.get("connects", (None, None))),  # type: ignore[arg-type]
                )
                for o in data.get("openings", [])
            ],
            rooms=[
                Room(
                    id=r["id"],
                    kind=RoomKind(r["kind"]),
                    label=r.get("label", ""),
                    polygon=poly(r["polygon"]),
                    centroid=pt(r["centroid"]),
                    stated_dimension=dim(r.get("stated_dimension")),
                    ceiling_height_mm=float(r.get("ceiling_height_mm", 2900.0)),
                    floor_level_mm=float(r.get("floor_level_mm", 0.0)),
                    source=Source(r.get("source", "cad_vector")),
                    confidence=float(r.get("confidence", 1.0)),
                    notes=list(r.get("notes", [])),
                )
                for r in data.get("rooms", [])
            ],
            fixtures=[
                FixtureHint(
                    id=f["id"],
                    label=f.get("label", ""),
                    footprint=poly(f.get("footprint", [])),
                    room_id=f.get("room_id"),
                    category=f.get("category", "unknown"),
                    source=Source(f.get("source", "cad_vector")),
                    confidence=float(f.get("confidence", 1.0)),
                    block_name=f.get("block_name"),
                )
                for f in data.get("fixtures", [])
            ],
            units=[
                PlanUnit(
                    id=u["id"],
                    name=u.get("name", ""),
                    room_ids=list(u.get("room_ids", [])),
                    wall_ids=list(u.get("wall_ids", [])),
                    stated_area_sqft=u.get("stated_area_sqft"),
                    boundary=[tuple(pt) for pt in u.get("boundary", [])],
                    bounds=tuple(u.get("bounds", (0, 0, 0, 0))),  # type: ignore[arg-type]
                    confidence=float(u.get("confidence", 1.0)),
                )
                for u in data.get("units", [])
            ],
            imperial_display=bool(data.get("imperial_display", False)),
            ingest_notes=list(data.get("ingest_notes", [])),
            warnings=list(data.get("warnings", [])),
        )
