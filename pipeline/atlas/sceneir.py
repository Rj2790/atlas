"""SceneIR — the 3D scene specification, designed for modularity.

PlanIR says what the building *is*. SceneIR says what to *render*, and it is the
document the web configurator loads and mutates.

The central design decision is that **nothing visual is baked in**. Every surface
carries a *material slot* — a named role like ``floor.living`` or ``wall.interior``
— and the slot is resolved to an actual material through a separate
:class:`DesignScheme`. Every furniture item is a reference to a catalogue id plus
a transform, never geometry. So:

* changing the flooring is reassigning one slot, not rebuilding the model;
* changing the whole look is swapping one scheme object;
* swapping a sofa is changing one id.

If materials were written onto the meshes instead, every one of those operations
would mean re-running the geometry build, and an interactive configurator would be
impossible. This is what "modular" has to mean at the data level for it to be true
at the product level.

Geometry is metres here, not millimetres. Millimetres are right for CAD arithmetic
and wrong for real-time 3D: glTF's convention is metres, and physically-based
shading, light falloff and camera near/far planes all assume it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal

__all__ = [
    "SCENE_SCHEMA_VERSION",
    "MaterialSlot",
    "Material",
    "DesignScheme",
    "SurfaceKind",
    "Surface",
    "OpeningSpec",
    "FurniturePlacement",
    "LightSpec",
    "CameraSpec",
    "RoomZone",
    "SceneIR",
    "MM_TO_M",
]

SCENE_SCHEMA_VERSION = "1.0"
MM_TO_M = 0.001


class MaterialSlot(StrEnum):
    """Named surface roles a scheme assigns materials to.

    Deliberately a closed set: it is the contract between the geometry build and
    the design schemes, so a scheme author knows exactly what they must cover and
    the UI knows exactly what it can offer to change.
    """

    FLOOR_LIVING = "floor.living"
    FLOOR_BEDROOM = "floor.bedroom"
    FLOOR_KITCHEN = "floor.kitchen"
    FLOOR_WET = "floor.wet"
    FLOOR_OUTDOOR = "floor.outdoor"
    FLOOR_CIRCULATION = "floor.circulation"
    WALL_INTERIOR = "wall.interior"
    WALL_ACCENT = "wall.accent"
    """One wall per habitable room, chosen by the build as the visual anchor —
    the wall a bed or sofa backs onto. Architects treat it as a feature; the
    configurator exposes it as its own swap."""

    WALL_WET = "wall.wet"
    WALL_EXTERIOR = "wall.exterior"
    CEILING = "ceiling"
    SKIRTING = "skirting"
    DOOR_LEAF = "door.leaf"
    DOOR_FRAME = "door.frame"
    WINDOW_FRAME = "window.frame"
    GLAZING = "glazing"
    PARAPET = "parapet"
    COUNTER = "counter"
    JOINERY = "joinery"
    """Built-in carpentry: wardrobes, kitchen carcasses, vanity units."""

    METAL = "metal"
    FABRIC = "fabric"
    WOOD = "wood"


@dataclass(slots=True)
class Material:
    """A physically-based material.

    Values are glTF metallic-roughness, which is what both three.js and Blender
    consume natively, so a material authored once is identical in the browser
    preview and the offline hero render. Texture fields are optional paths; a
    material with no textures still renders correctly from its scalar values,
    which keeps the scene usable before any assets are downloaded.
    """

    id: str
    name: str
    base_color: tuple[float, float, float] = (0.8, 0.8, 0.8)
    roughness: float = 0.6
    metallic: float = 0.0
    base_color_texture: str | None = None
    normal_texture: str | None = None
    roughness_texture: str | None = None
    ao_texture: str | None = None
    texture_scale_m: float = 1.0
    """Real-world size the texture tiles over, in metres. Getting this wrong is
    the fastest way to make a render look fake — a 4 m oak plank or a 5 cm floor
    tile reads as obviously wrong even to a non-specialist."""

    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)
    opacity: float = 1.0
    ior: float = 1.5
    sheen: float = 0.0
    """Fabric sheen. Sofas and bedding look like plastic without it."""

    texture_neutralize: bool = False
    """Strip the colour map's own hue, keeping only its surface structure.

    For a painted or plastered surface the hue is a *design decision* and the
    texture's job is only to supply the trowel marks and roller stipple. Multiplying
    a tint by a map that carries its own colour compounds the two: tinting an
    off-white paint by 0.90 over a warm plaster photograph produced visibly pink
    walls. Neutralised, the map contributes luminance variation only and
    ``base_color`` sets the colour exactly as specified.

    Left False for materials whose colour *is* the material — oak, marble, terrazzo,
    tile, fabric — where discarding the map's hue would throw away the point of it.
    """

    acg_asset: str | None = None
    """ambientCG asset id supplying this material's PBR maps (all CC0).

    Kept as an *id* rather than a set of paths so the scheme stays a small,
    declarative document: the texture cache resolves ids to files, and a scheme
    can be authored and diffed without any binary assets present."""

    credit: str | None = None
    """Attribution and licence for a sourced texture. Tracked because shipping
    someone's CC-BY texture without credit is a licence breach."""


@dataclass(slots=True)
class DesignScheme:
    """A complete curated look: every material slot filled, plus lighting mood.

    This is where "equivalent to a top-notch creative architect firm" is actually
    won. Correct geometry with default grey materials looks like a CAD viewport.
    The scheme is the design content — a coherent palette, correct sheens, sensible
    texture scales, a colour temperature — and swapping it restyles the whole
    apartment at once, which is exactly the operation an interior designer wants.
    """

    id: str
    name: str
    description: str
    materials: dict[str, Material]
    slots: dict[str, str]
    """Material slot name -> material id."""

    accent_colour: tuple[float, float, float] = (0.55, 0.42, 0.32)
    ambient_intensity: float = 0.35
    sun_intensity: float = 3.0
    sun_temperature_k: float = 5200.0
    interior_temperature_k: float = 3000.0
    hdri: str | None = None

    def material_for(self, slot: str | MaterialSlot) -> Material | None:
        mid = self.slots.get(str(slot))
        return self.materials.get(mid) if mid else None


class SurfaceKind(StrEnum):
    WALL = "wall"
    FLOOR = "floor"
    CEILING = "ceiling"
    SKIRTING = "skirting"
    PARAPET = "parapet"
    REVEAL = "reveal"
    """The inner faces of an opening — the thickness of the wall seen through a
    door or window. Omitting these is the classic tell of a cheap 3D model: you
    see through the wall's zero-thickness edge."""

    THRESHOLD = "threshold"


@dataclass(slots=True)
class Surface:
    """A triangulated surface with a material slot, not a material.

    Vertices are flat ``[x, y, z, ...]`` in metres, Y-up (glTF convention), and
    indices are flat triangles. Storing raw buffers rather than a mesh object keeps
    SceneIR pure data, so it serialises to JSON and crosses the Python/TypeScript
    boundary without a mesh library on either side.
    """

    id: str
    kind: SurfaceKind
    slot: str
    vertices: list[float]
    indices: list[int]
    normals: list[float] = field(default_factory=list)
    uvs: list[float] = field(default_factory=list)
    room_id: str | None = None
    wall_id: str | None = None

    @property
    def triangle_count(self) -> int:
        return len(self.indices) // 3


@dataclass(slots=True)
class OpeningSpec:
    """A door or window instanced in 3D, kept as a parametric description.

    Parametric rather than baked geometry so the configurator can restyle a door
    (flush to panelled, oak to lacquer) or animate a swing without touching the
    wall it sits in.
    """

    id: str
    kind: str
    wall_id: str
    centre: tuple[float, float, float]
    width: float
    height: float
    sill: float
    rotation_y: float
    thickness: float
    swing: str = "none"
    swing_inward: bool = True
    leaf_slot: str = MaterialSlot.DOOR_LEAF.value
    frame_slot: str = MaterialSlot.DOOR_FRAME.value
    glazed: bool = False


@dataclass(slots=True)
class FurniturePlacement:
    """One catalogue item placed in the scene.

    ``catalog_id`` is a reference, never geometry, which is what makes swapping a
    single call. ``footprint`` and ``clearance`` are carried along so the
    configurator can re-run collision checks client-side when the user drags an
    item, instead of round-tripping to the solver.
    """

    id: str
    catalog_id: str
    room_id: str
    position: tuple[float, float, float]
    rotation_y: float
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    footprint: tuple[float, float] = (0.0, 0.0)
    height: float = 0.0
    clearance: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    """Required free space (front, back, left, right) in metres."""

    material_overrides: dict[str, str] = field(default_factory=dict)
    """Per-item slot overrides, so one sofa can be re-upholstered without
    affecting every other item using the same material."""

    anchored_to_wall: str | None = None
    locked: bool = False
    """Set for items derived from the drawing (sanitaryware, kitchen counters).
    These encode plumbing and service constraints and must not be moved by the
    auto-layout, only by a human."""

    source: str = "solver"
    reason: str = ""
    """Why the solver placed it here — surfaced in the UI so a designer can argue
    with the machine instead of guessing at it."""


@dataclass(slots=True)
class LightSpec:
    id: str
    kind: Literal["sun", "area", "point", "spot", "cove"]
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: tuple[float, float, float] = (0.0, -1.0, 0.0)
    intensity: float = 1.0
    """Renderer-neutral relative intensity. Each backend maps it to its own units
    (Watts in Cycles, candela-ish in three.js), because a single number cannot mean
    the same thing to a path tracer and a rasteriser."""

    colour_k: float = 3000.0
    size: tuple[float, float] = (0.3, 0.3)
    room_id: str | None = None
    cast_shadow: bool = True
    realtime_only: bool = False
    """Skip this light in a path-traced render.

    Real-time renderers cannot compute indirect light, so they need fill lights
    inside each window to stand in for daylight bouncing off the floor. Cycles
    computes that bounce for real from the sky through the same opening, so
    including these as well double-counts the daylight and blows the image out.
    One scene description, two correct interpretations."""


@dataclass(slots=True)
class CameraSpec:
    """A pre-framed view. Computed once so the browser and the offline render agree.

    Framing is computed here, with the room polygons in hand, rather than in each
    renderer. Getting a camera reliably *inside* an irregular room needs the
    geometry — placing it by eye from the centroid puts it through a wall whenever
    the room is L-shaped, which is most of them."""

    id: str
    room_id: str
    name: str
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    focal_mm: float = 24.0
    """24 mm is the architectural-photography standard. Wider bends verticals and
    exaggerates the space."""


@dataclass(slots=True)
class RoomZone:
    """A room as the configurator sees it: what can be changed, and its extent."""

    id: str
    name: str
    kind: str
    centroid: tuple[float, float, float]
    area_m2: float
    floor_slot: str
    wall_slot: str
    ceiling_slot: str
    accent_wall_id: str | None = None
    polygon: list[tuple[float, float]] = field(default_factory=list)
    ceiling_height: float = 2.9
    floor_level: float = 0.0
    stated_dimension_m: tuple[float, float] | None = None


@dataclass(slots=True)
class SceneIR:
    """A complete, renderable, swappable 3D scene."""

    schema_version: str = SCENE_SCHEMA_VERSION
    name: str = ""
    source_plan: str = ""
    up_axis: Literal["Y"] = "Y"
    unit: Literal["m"] = "m"
    origin_offset: tuple[float, float] = (0.0, 0.0)
    """Plan-space millimetres subtracted before conversion, so the model sits at
    the origin. Kept so a coordinate can be traced back to the source drawing —
    survey coordinates in the tens of millions destroy float precision in a
    renderer, so re-centring is mandatory, and recording it keeps it reversible."""

    surfaces: list[Surface] = field(default_factory=list)
    openings: list[OpeningSpec] = field(default_factory=list)
    furniture: list[FurniturePlacement] = field(default_factory=list)
    lights: list[LightSpec] = field(default_factory=list)
    cameras: list[CameraSpec] = field(default_factory=list)
    rooms: list[RoomZone] = field(default_factory=list)
    scheme_id: str = ""
    available_schemes: list[str] = field(default_factory=list)
    slot_overrides: dict[str, str] = field(default_factory=dict)
    """Material slot overrides chosen by the art director, on top of the scheme.

    Kept separate from the scheme rather than folded into it: the scheme stays a
    reusable curated document, and the configurator can show the director's deviations
    as deviations, and undo them."""

    art_direction: dict[str, Any] = field(default_factory=dict)
    """The design decisions and the argument for them — scheme, mood, sun angle, hero
    room. Surfaced in the UI so a designer can see what the machine decided and why,
    rather than being handed an unexplained look."""
    bounds: tuple[float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0)
    imperial_display: bool = False
    notes: list[str] = field(default_factory=list)

    def stats(self) -> dict[str, Any]:
        return {
            "surfaces": len(self.surfaces),
            "triangles": sum(s.triangle_count for s in self.surfaces),
            "openings": len(self.openings),
            "furniture": len(self.furniture),
            "lights": len(self.lights),
            "cameras": len(self.cameras),
            "rooms": len(self.rooms),
            "slots_used": sorted({s.slot for s in self.surfaces}),
        }

    def to_dict(self) -> dict[str, Any]:
        def enc(o: Any) -> Any:
            if isinstance(o, StrEnum):
                return o.value
            if isinstance(o, dict):
                return {k: enc(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [enc(v) for v in o]
            if isinstance(o, float):
                # Six decimals is sub-micrometre at these scales and roughly
                # halves the JSON the browser has to parse.
                return round(o, 6)
            return o

        return enc(asdict(self))

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
