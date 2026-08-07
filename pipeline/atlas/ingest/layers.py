"""Classify CAD layer names into semantic roles.

Layer naming is where a CAD file tells you what things *are*, and it is the single
highest-leverage signal in the whole ingest. It is also chaotic. The demo file
alone mixes three conventions in one drawing:

* a practice's own scheme — ``ABA_02_wall``, ``ABA_06_door``, ``ABA_08_furniture``
* the AIA/NCS standard — ``A-WALL``, ``A-DOOR``, ``A-GLAZ``, ``A-FLOR-FURN``
* leftovers from a consultant working in another language — ``A03-보``, ``A22_콤비``

Rather than one regex per convention, roles are scored against weighted keyword
groups. A layer must hit a *positive* group and miss the *negative* ones, which is
what stops ``ABA_09_hatch wall`` (a poché fill, not a wall) and
``ABA_02_wall parapet`` (a wall, but a 1 m balcony parapet, not a room divider)
from being treated as ordinary walls. Both mistakes are visible instantly in 3D:
the first fills rooms with solid concrete, the second walls the balcony off to the
ceiling.

Layer evidence is strong but not sufficient. The extractor also uses geometry
(a 900 mm arc is a door swing) and block names, and layer role is one vote.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

__all__ = ["LayerRole", "LayerVerdict", "classify_layer", "ROLE_RULES"]


class LayerRole(StrEnum):
    WALL = "wall"
    """Room-dividing wall. Full height."""

    WALL_LOW = "wall_low"
    """Parapet, railing base, half-height wall. Height comes from the rule, not
    the level, or balconies end up enclosed."""

    COLUMN = "column"
    DOOR = "door"
    WINDOW = "window"
    WINDOW_SILL = "window_sill"
    ROOM_TEXT = "room_text"
    """Carries the room name and usually its printed dimension."""

    AREA_BOUNDARY = "area_boundary"
    """Carpet/built-up area polygon. Authoritative dwelling boundary."""

    FURNITURE = "furniture"
    SANITARY = "sanitary"
    KITCHEN_FIT = "kitchen_fit"
    FLOORING = "flooring"
    CEILING = "ceiling"
    STAIR = "stair"
    HATCH_WALL = "hatch_wall"
    """Wall poché. Confirms a wall exists but is not itself geometry to build."""

    HATCH_ROOM = "hatch_room"
    """Room floor fill. A useful hint that a region is a room."""

    DIMENSION = "dimension"
    ANNOTATION = "annotation"
    GRID = "grid"
    SITE = "site"
    """Roads, plot boundaries, contours, trees — outside the building."""

    MEP = "mep"
    IGNORE = "ignore"
    UNKNOWN = "unknown"

    @property
    def is_buildable(self) -> bool:
        """Roles that contribute solid geometry to the 3D model."""
        return self in (LayerRole.WALL, LayerRole.WALL_LOW, LayerRole.COLUMN)


@dataclass(frozen=True, slots=True)
class LayerVerdict:
    role: LayerRole
    confidence: float
    matched: str
    """Which rule fired, so a misclassification is debuggable."""


@dataclass(frozen=True, slots=True)
class _Rule:
    role: LayerRole
    include: tuple[str, ...]
    """Any one of these substrings must appear (already lowercased)."""

    exclude: tuple[str, ...] = ()
    """None of these may appear."""

    weight: float = 1.0
    priority: int = 0
    """Higher priority wins when several rules match. Used to make the specific
    beat the general: ``wall parapet`` must outrank plain ``wall``."""


# Ordered by specificity within each role. Note the systematic use of `exclude`
# to keep hatches, text and area layers out of the geometry roles.
ROLE_RULES: tuple[_Rule, ...] = (
    # --- hatches: matched early so they never fall through to their base role ---
    _Rule(LayerRole.HATCH_WALL, ("hatch wall", "hatch_wall", "wall-patt", "wall patt", "hatch rcc"), weight=0.95, priority=90),
    _Rule(LayerRole.HATCH_ROOM, ("hatch room", "hatch_room", "hatch balcony", "hatch sunk", "hatch floor"), weight=0.9, priority=90),
    _Rule(LayerRole.IGNORE, ("hatch furniture", "hatch misc", "hatch landscape", "hatch false ceiling", "elev-patt"), weight=0.9, priority=90),

    # --- text / annotation: also early, since "text room" contains "room" ---
    _Rule(LayerRole.ROOM_TEXT, ("text room", "text_room", "room name", "rmname", "anno-room", "room-iden"), weight=0.98, priority=88),
    _Rule(LayerRole.ANNOTATION, ("text ", "_text", "anno", "notes", "comment", "signage", "sheet data",
                                 "text levels", "unit number", "label"), weight=0.7, priority=40),
    _Rule(LayerRole.DIMENSION, ("dimension", "dim", "-dims", "measure"), weight=0.85, priority=45),

    # --- area boundaries: the authoritative dwelling outline ---
    _Rule(LayerRole.AREA_BOUNDARY, ("carpet", "area fsi", "area_fsi", "built up", "builtup",
                                    "area construction", "rera"), weight=0.95, priority=85),

    # --- walls, specific before general ---
    _Rule(LayerRole.WALL_LOW, ("parapet", "railing", "handrail", "balustrade", "kerb", "curb",
                               "wall low", "half wall", "dwarf wall"), weight=0.9, priority=80),
    _Rule(LayerRole.COLUMN, ("column", "col ", "-cols", "_cols", "pillar", "rcc column", "ms column"),
          exclude=("text column",), weight=0.9, priority=75),
    _Rule(LayerRole.WALL, ("wall", "a-wall", "partition", "rcc wall", "masonry", "brick", "벽"),
          exclude=("hatch", "patt", "parapet", "railing", "text", "retaining", "carpet"),
          weight=0.95, priority=60),

    # --- openings ---
    _Rule(LayerRole.WINDOW_SILL, ("window sill", "window_sill", "sill"), weight=0.9, priority=78),
    _Rule(LayerRole.DOOR, ("door", "a-door", "shutter", "문"),
          exclude=("text", "schedule", "window"), weight=0.95, priority=70),
    _Rule(LayerRole.WINDOW, ("window", "glaz", "a-glaz", "glazing", "ventilator", "louver", "louvre", "창"),
          exclude=("text", "sill", "schedule"), weight=0.95, priority=70),

    # --- contents ---
    _Rule(LayerRole.SANITARY, ("sanitary", "sanitaryware", "plumb fixture", "wc", "toilet fixture"),
          exclude=("text",), weight=0.9, priority=72),
    _Rule(LayerRole.KITCHEN_FIT, ("flor-kit", "kitchen", "counter", "platform", "eqpm-fixd", "casework"),
          exclude=("text",), weight=0.8, priority=68),
    _Rule(LayerRole.FURNITURE, ("furniture", "furn", "flor-furn", "fixture", "equipment", "wardrobe",
                                "cupboard", "pallet"),
          exclude=("hatch", "text", "area cupboard"), weight=0.9, priority=65),
    _Rule(LayerRole.FLOORING, ("flooring", "floor pattern", "floor finish", "tile"),
          exclude=("hatch",), weight=0.8, priority=55),
    _Rule(LayerRole.CEILING, ("false ceiling", "ceiling", "rcp", "soffit"),
          exclude=("hatch", "text"), weight=0.8, priority=55),
    _Rule(LayerRole.STAIR, ("stair", "step", "tread", "riser", "lift", "elevator", "escalator"),
          exclude=("text",), weight=0.85, priority=62),

    # --- context we must recognise in order to discard ---
    _Rule(LayerRole.GRID, ("grid", "center grid", "centre grid", "axis", "lines center"), weight=0.8, priority=50),
    _Rule(LayerRole.SITE, ("plot", "road", "driveway", "contour", "survey", "tree", "landscape",
                           "lanscape", "boundary", "demarcation", "margin", "vehicle", "car",
                           "cycle", "scooter", "terrace garden", "flowerbed"),
          exclude=("text",), weight=0.8, priority=48),
    _Rule(LayerRole.MEP, ("mep", "hvac", "duct", "electr", "plumb", "drain", "sewer", "water line",
                          "fire", "sprinkler", "conduit"),
          exclude=("text",), weight=0.8, priority=52),
    _Rule(LayerRole.IGNORE, ("defpoints", "viewport", "titleblock", "title block", "revision",
                             "elevation 0", "lines above", "lines dotted", "lines section",
                             "lines block", "refedit", "north", "hatch", "drop", "screen",
                             "cladding", "grill", "glass", "rafters", "arrow", "levels"),
          weight=0.6, priority=20),
)

# Layers that carry no information at all.
_EMPTY_NAMES = frozenset({"", "0", ".", "4", "defpoints"})

_NON_LATIN = re.compile(r"[^\x00-\x7f]")


def classify_layer(layer: str) -> LayerVerdict:
    """Assign a :class:`LayerRole` to a normalised layer name.

    Expects the name already stripped of xref binding prefixes (see
    :func:`atlas.ingest.dxf_flatten.normalise_layer`).
    """
    name = (layer or "").strip()
    if name.lower() in _EMPTY_NAMES:
        # Layer 0 is not meaningless — it is the *default*, and plenty of drawings
        # put real geometry there. Unknown, so geometry decides.
        return LayerVerdict(LayerRole.UNKNOWN, 0.0, "empty-name")

    norm = name.lower().replace("_", " ").replace("-", " ")
    norm = " ".join(norm.split())
    # Also keep a hyphen-preserving form so AIA codes like "a-wall" still match.
    norm_hyphen = " ".join(name.lower().replace("_", " ").split())

    best: LayerVerdict | None = None
    for rule in ROLE_RULES:
        if any(bad in norm or bad in norm_hyphen for bad in rule.exclude):
            continue
        hit = next((inc for inc in rule.include if inc in norm or inc in norm_hyphen), None)
        if hit is None:
            continue
        if best is None or rule.priority > _priority_of(best.matched):
            best = LayerVerdict(rule.role, rule.weight, f"{rule.role.value}:{hit}@{rule.priority}")
    if best is not None:
        return best

    if _NON_LATIN.search(name):
        # A non-Latin layer we have no rule for. Say so rather than guessing —
        # the extractor will fall back to geometry, and the note tells the user
        # a layer went unread.
        return LayerVerdict(LayerRole.UNKNOWN, 0.0, "non-latin-unmapped")
    return LayerVerdict(LayerRole.UNKNOWN, 0.0, "no-rule")


def _priority_of(matched: str) -> int:
    """Recover the priority encoded in a verdict's ``matched`` string."""
    try:
        return int(matched.rsplit("@", 1)[1])
    except (IndexError, ValueError):
        return -1
