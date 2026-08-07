"""Curated design schemes.

Correct geometry with default materials looks like a CAD viewport, not a project.
The difference between the two is almost entirely in this file, so it is written as
design content rather than as configuration.

Three things here are doing the real work, and each is a common way renders go
wrong:

**Colours are muted and never pure.** No value is 1.0 or 0.0, and neutrals are
given a slight temperature bias rather than being grey. A wall at ``(1, 1, 1)``
clips to white under any real light and destroys the impression of daylight;
architectural white paint is closer to ``0.87`` with a warm cast.

**Roughness is specific per material.** Polished marble at 0.12, matt emulsion at
0.9, oiled oak at 0.45. A scene where everything sits near 0.5 has no material
contrast, which is the single strongest tell of an amateur render — the eye reads
material from the shape of its highlight before it reads colour.

**Texture scale is stated in metres.** A 600 mm floor tile and a 1.2 m oak plank
are different objects even with identical maps. ``texture_scale_m`` keeps the tile
the size it would really be, so the space reads at the right scale.

Schemes are complete: every :class:`MaterialSlot` is filled, so no surface ever
falls back to a default grey.
"""

from __future__ import annotations

from ..sceneir import DesignScheme, Material, MaterialSlot

__all__ = ["SCHEMES", "default_scheme", "get_scheme", "scheme_summaries"]


def _m(
    mid: str,
    name: str,
    rgb: tuple[float, float, float],
    rough: float,
    *,
    metal: float = 0.0,
    scale: float = 1.0,
    sheen: float = 0.0,
    opacity: float = 1.0,
    ior: float = 1.5,
    acg: str | None = None,
    neutral: bool = False,
) -> Material:
    """Define a material.

    ``rgb`` still matters when ``acg`` is set: the colour map is *multiplied* by it,
    so the same oak asset can be tinted warm or smoked, and the same plaster can be
    off-white or clay, without downloading a separate texture. It is also the
    fallback the material renders with if the texture download fails.
    """
    return Material(
        id=mid,
        name=name,
        base_color=rgb,
        roughness=rough,
        metallic=metal,
        texture_scale_m=scale,
        sheen=sheen,
        opacity=opacity,
        ior=ior,
        acg_asset=acg,
        texture_neutralize=neutral,
    )


# Materials shared by every scheme: glazing physics and metalwork do not change
# with the interior palette, and duplicating them invites them drifting apart.
def _common() -> dict[str, Material]:
    return {
        "glass-clear": _m("glass-clear", "Clear glazing", (0.93, 0.95, 0.96), 0.03, opacity=0.16, ior=1.52),
        "steel-dark": _m("steel-dark", "Dark bronze metal", (0.24, 0.23, 0.22), 0.35, metal=0.9),
        "steel-brushed": _m("steel-brushed", "Brushed nickel", (0.72, 0.73, 0.74), 0.3, metal=1.0),
    }


_WARM_MINIMAL = DesignScheme(
    id="warm-minimal",
    name="Warm Minimal",
    description=(
        "Wide-plank European oak, lime-plaster walls and a single clay accent. "
        "Restrained, warm and daylight-led — the safe choice for a residential sale."
    ),
    materials={
        **_common(),
        "oak-wide": _m("oak-wide", "Wide-plank white oak", (0.72, 0.60, 0.45), 0.45, scale=1.9, acg="WoodFloor043"),
        "oak-warm": _m("oak-warm", "Warm oak", (0.70, 0.56, 0.40), 0.5, scale=1.6, acg="WoodFloor051"),
        "lime-white": _m("lime-white", "Lime plaster, off-white", (0.90, 0.885, 0.86), 0.9, scale=2.6, acg="PaintedPlaster017", neutral=True),
        "clay-accent": _m("clay-accent", "Clay render accent", (0.70, 0.58, 0.49), 0.85, scale=2.4, acg="Plaster001", neutral=True),
        "terrazzo-warm": _m("terrazzo-warm", "Warm terrazzo", (0.84, 0.81, 0.76), 0.35, scale=0.9, acg="Terrazzo013"),
        "stone-honed": _m("stone-honed", "Honed limestone", (0.82, 0.80, 0.76), 0.4, scale=0.6, acg="Travertine009"),
        "tile-sand": _m("tile-sand", "Sand ceramic tile", (0.80, 0.76, 0.70), 0.3, scale=0.6, acg="Tiles139"),
        "ceiling-white": _m("ceiling-white", "Flat ceiling white", (0.91, 0.905, 0.895), 0.95),
        "skirting-oak": _m("skirting-oak", "Oak skirting", (0.66, 0.52, 0.37), 0.4, scale=0.9, acg="Wood051"),
        "render-ext": _m("render-ext", "Exterior render", (0.84, 0.825, 0.80), 0.8, scale=3.0, acg="Concrete034", neutral=True),
        "deck-teak": _m("deck-teak", "Teak decking", (0.58, 0.45, 0.31), 0.55, scale=1.1, acg="WoodFloor064"),
        "quartz-cream": _m("quartz-cream", "Cream quartz counter", (0.88, 0.86, 0.82), 0.18, scale=2.4, acg="Marble012"),
        "joinery-oak": _m("joinery-oak", "Oak veneer joinery", (0.70, 0.57, 0.42), 0.42, scale=1.2, acg="Wood066"),
        "linen-oat": _m("linen-oat", "Oatmeal linen", (0.80, 0.76, 0.68), 0.85, sheen=0.35, scale=0.45, acg="Fabric066"),
    },
    slots={
        MaterialSlot.FLOOR_LIVING.value: "oak-wide",
        MaterialSlot.FLOOR_BEDROOM.value: "oak-warm",
        MaterialSlot.FLOOR_KITCHEN.value: "terrazzo-warm",
        MaterialSlot.FLOOR_WET.value: "tile-sand",
        MaterialSlot.FLOOR_OUTDOOR.value: "deck-teak",
        MaterialSlot.FLOOR_CIRCULATION.value: "stone-honed",
        MaterialSlot.WALL_INTERIOR.value: "lime-white",
        MaterialSlot.WALL_ACCENT.value: "clay-accent",
        MaterialSlot.WALL_WET.value: "stone-honed",
        MaterialSlot.WALL_EXTERIOR.value: "render-ext",
        MaterialSlot.CEILING.value: "ceiling-white",
        MaterialSlot.SKIRTING.value: "skirting-oak",
        MaterialSlot.DOOR_LEAF.value: "oak-warm",
        MaterialSlot.DOOR_FRAME.value: "lime-white",
        MaterialSlot.WINDOW_FRAME.value: "steel-dark",
        MaterialSlot.GLAZING.value: "glass-clear",
        MaterialSlot.PARAPET.value: "render-ext",
        MaterialSlot.COUNTER.value: "quartz-cream",
        MaterialSlot.JOINERY.value: "joinery-oak",
        MaterialSlot.METAL.value: "steel-dark",
        MaterialSlot.FABRIC.value: "linen-oat",
        MaterialSlot.WOOD.value: "oak-warm",
    },
    accent_colour=(0.66, 0.55, 0.47),
    ambient_intensity=0.36,
    sun_intensity=3.2,
    sun_temperature_k=5400.0,
    interior_temperature_k=2900.0,
)


_JAPANDI = DesignScheme(
    id="japandi",
    name="Japandi",
    description=(
        "Pale ash, micro-cement and charred timber. Low contrast, high craft — "
        "reads as considered rather than decorated."
    ),
    materials={
        **_common(),
        "ash-pale": _m("ash-pale", "Pale ash flooring", (0.80, 0.73, 0.62), 0.5, scale=1.7, acg="WoodFloor070"),
        "ash-bedroom": _m("ash-bedroom", "Pale ash, matt", (0.78, 0.71, 0.60), 0.58, scale=1.5, acg="WoodFloor070"),
        "microcement": _m("microcement", "Micro-cement", (0.78, 0.77, 0.75), 0.75, scale=2.8, acg="Concrete036", neutral=True),
        "plaster-bone": _m("plaster-bone", "Bone plaster", (0.89, 0.88, 0.86), 0.92, scale=2.6, acg="PaintedPlaster016", neutral=True),
        "shou-sugi": _m("shou-sugi", "Charred cedar", (0.26, 0.24, 0.23), 0.62, scale=1.3, acg="Wood095"),
        "tile-slate": _m("tile-slate", "Slate tile", (0.44, 0.445, 0.45), 0.45, scale=0.8, acg="Tiles141"),
        "ceiling-bone": _m("ceiling-bone", "Bone ceiling", (0.91, 0.90, 0.89), 0.95, scale=3.2, acg="PaintedPlaster016", neutral=True),
        "skirting-shadow": _m("skirting-shadow", "Shadow-gap skirting", (0.84, 0.83, 0.82), 0.9, scale=1.0, acg="PaintedPlaster016", neutral=True),
        "render-grey": _m("render-grey", "Grey render", (0.76, 0.76, 0.75), 0.82, scale=3.0, acg="Concrete034", neutral=True),
        "stone-outdoor": _m("stone-outdoor", "Basalt paving", (0.52, 0.52, 0.52), 0.7, scale=0.9, acg="Concrete031", neutral=True),
        "stone-black": _m("stone-black", "Honed black granite", (0.22, 0.22, 0.225), 0.22, scale=2.2, acg="Marble016"),
        "joinery-ash": _m("joinery-ash", "Ash joinery", (0.76, 0.69, 0.58), 0.48, scale=1.1, acg="Wood094"),
        "wool-grey": _m("wool-grey", "Grey wool", (0.60, 0.60, 0.59), 0.88, sheen=0.4, scale=0.4, acg="Fabric061"),
    },
    slots={
        MaterialSlot.FLOOR_LIVING.value: "ash-pale",
        MaterialSlot.FLOOR_BEDROOM.value: "ash-bedroom",
        MaterialSlot.FLOOR_KITCHEN.value: "microcement",
        MaterialSlot.FLOOR_WET.value: "tile-slate",
        MaterialSlot.FLOOR_OUTDOOR.value: "stone-outdoor",
        MaterialSlot.FLOOR_CIRCULATION.value: "microcement",
        MaterialSlot.WALL_INTERIOR.value: "plaster-bone",
        MaterialSlot.WALL_ACCENT.value: "shou-sugi",
        MaterialSlot.WALL_WET.value: "microcement",
        MaterialSlot.WALL_EXTERIOR.value: "render-grey",
        MaterialSlot.CEILING.value: "ceiling-bone",
        MaterialSlot.SKIRTING.value: "skirting-shadow",
        MaterialSlot.DOOR_LEAF.value: "joinery-ash",
        MaterialSlot.DOOR_FRAME.value: "plaster-bone",
        MaterialSlot.WINDOW_FRAME.value: "steel-dark",
        MaterialSlot.GLAZING.value: "glass-clear",
        MaterialSlot.PARAPET.value: "render-grey",
        MaterialSlot.COUNTER.value: "stone-black",
        MaterialSlot.JOINERY.value: "joinery-ash",
        MaterialSlot.METAL.value: "steel-dark",
        MaterialSlot.FABRIC.value: "wool-grey",
        MaterialSlot.WOOD.value: "ash-pale",
    },
    accent_colour=(0.16, 0.15, 0.145),
    ambient_intensity=0.4,
    sun_intensity=2.6,
    sun_temperature_k=6000.0,
    interior_temperature_k=3200.0,
)


_MODERN_INDIAN = DesignScheme(
    id="modern-indian",
    name="Modern Indian",
    description=(
        "Kota stone, teak joinery and white-washed walls with a jade accent. "
        "Tuned to the Indian residential market these plans come from: hard floors "
        "for the climate, deep-toned timber, brass fittings."
    ),
    materials={
        **_common(),
        "kota-grey": _m("kota-grey", "Kota stone, honed", (0.54, 0.56, 0.54), 0.4, scale=0.9, acg="Concrete031", neutral=True),
        "marble-makrana": _m("marble-makrana", "Makrana white marble", (0.89, 0.88, 0.86), 0.14, scale=1.4, acg="Marble012"),
        "vitrified-ivory": _m("vitrified-ivory", "Ivory vitrified tile", (0.86, 0.84, 0.80), 0.16, scale=0.8, acg="Tiles139"),
        "teak-deep": _m("teak-deep", "Burma teak", (0.52, 0.38, 0.25), 0.4, scale=1.3, acg="Wood060"),
        "lime-wash": _m("lime-wash", "Lime wash white", (0.91, 0.90, 0.885), 0.9, scale=2.6, acg="PaintedPlaster017", neutral=True),
        "jade-accent": _m("jade-accent", "Jade green distemper", (0.40, 0.50, 0.46), 0.85, scale=2.4, acg="Plaster002", neutral=True),
        "tile-ceramic": _m("tile-ceramic", "Glazed ceramic, wet area", (0.85, 0.84, 0.81), 0.2, scale=0.6, acg="Tiles140"),
        "ceiling-pop": _m("ceiling-pop", "POP ceiling white", (0.92, 0.915, 0.905), 0.95),
        "skirting-granite": _m("skirting-granite", "Granite skirting", (0.40, 0.40, 0.41), 0.35, scale=0.8, acg="Marble016"),
        "render-cement": _m("render-cement", "Cement plaster", (0.81, 0.80, 0.78), 0.85, scale=3.0, acg="Concrete034", neutral=True),
        "ips-outdoor": _m("ips-outdoor", "IPS finish, balcony", (0.70, 0.69, 0.66), 0.7, scale=1.2, acg="Concrete036", neutral=True),
        "granite-black": _m("granite-black", "Black granite counter", (0.20, 0.20, 0.21), 0.2, scale=2.0, acg="Marble016"),
        "brass": _m("brass", "Antique brass", (0.62, 0.50, 0.28), 0.32, metal=1.0),
        "cotton-ivory": _m("cotton-ivory", "Ivory cotton", (0.85, 0.82, 0.76), 0.86, sheen=0.3, scale=0.45, acg="Fabric066"),
    },
    slots={
        MaterialSlot.FLOOR_LIVING.value: "marble-makrana",
        MaterialSlot.FLOOR_BEDROOM.value: "teak-deep",
        MaterialSlot.FLOOR_KITCHEN.value: "vitrified-ivory",
        MaterialSlot.FLOOR_WET.value: "tile-ceramic",
        MaterialSlot.FLOOR_OUTDOOR.value: "ips-outdoor",
        MaterialSlot.FLOOR_CIRCULATION.value: "kota-grey",
        MaterialSlot.WALL_INTERIOR.value: "lime-wash",
        MaterialSlot.WALL_ACCENT.value: "jade-accent",
        MaterialSlot.WALL_WET.value: "tile-ceramic",
        MaterialSlot.WALL_EXTERIOR.value: "render-cement",
        MaterialSlot.CEILING.value: "ceiling-pop",
        MaterialSlot.SKIRTING.value: "skirting-granite",
        MaterialSlot.DOOR_LEAF.value: "teak-deep",
        MaterialSlot.DOOR_FRAME.value: "teak-deep",
        MaterialSlot.WINDOW_FRAME.value: "steel-brushed",
        MaterialSlot.GLAZING.value: "glass-clear",
        MaterialSlot.PARAPET.value: "render-cement",
        MaterialSlot.COUNTER.value: "granite-black",
        MaterialSlot.JOINERY.value: "teak-deep",
        MaterialSlot.METAL.value: "brass",
        MaterialSlot.FABRIC.value: "cotton-ivory",
        MaterialSlot.WOOD.value: "teak-deep",
    },
    accent_colour=(0.31, 0.42, 0.38),
    ambient_intensity=0.34,
    sun_intensity=4.0,
    sun_temperature_k=5000.0,
    interior_temperature_k=3000.0,
)


_MONOCHROME = DesignScheme(
    id="monochrome",
    name="Monochrome Gallery",
    description=(
        "Poured concrete, white walls and blackened steel. High contrast, "
        "art-forward — deliberately cool where the others are warm."
    ),
    materials={
        **_common(),
        "concrete-poured": _m("concrete-poured", "Poured concrete, sealed", (0.70, 0.70, 0.69), 0.55, scale=3.2, acg="Concrete046", neutral=True),
        "concrete-bedroom": _m("concrete-bedroom", "Concrete, matt seal", (0.68, 0.68, 0.67), 0.68, scale=3.0, acg="Concrete047A", neutral=True),
        "resin-white": _m("resin-white", "White resin", (0.87, 0.87, 0.87), 0.25, scale=2.6, acg="Concrete045", neutral=True),
        "tile-graphite": _m("tile-graphite", "Graphite tile", (0.36, 0.365, 0.37), 0.35, scale=0.7, acg="Tiles142"),
        "paint-gallery": _m("paint-gallery", "Gallery white", (0.92, 0.92, 0.918), 0.93, scale=2.8, acg="PaintedPlaster016", neutral=True),
        "paint-graphite": _m("paint-graphite", "Graphite wall", (0.31, 0.315, 0.32), 0.88, scale=2.6, acg="Plaster003", neutral=True),
        "ceiling-flat": _m("ceiling-flat", "Flat white ceiling", (0.925, 0.925, 0.92), 0.96),
        "skirting-flush": _m("skirting-flush", "Flush white skirting", (0.89, 0.89, 0.885), 0.9, scale=1.0, acg="PaintedPlaster016", neutral=True),
        "render-white": _m("render-white", "White render", (0.845, 0.845, 0.84), 0.8, neutral=True),
        "granite-paving": _m("granite-paving", "Granite paving", (0.48, 0.48, 0.485), 0.68, scale=0.9, acg="Concrete031", neutral=True),
        "corian-white": _m("corian-white", "White solid surface", (0.90, 0.90, 0.895), 0.2, scale=2.4, acg="Marble021"),
        "oak-smoked": _m("oak-smoked", "Smoked oak joinery", (0.44, 0.38, 0.33), 0.45, scale=1.2, acg="Wood092"),
        "canvas-grey": _m("canvas-grey", "Grey canvas", (0.53, 0.535, 0.54), 0.9, sheen=0.25, scale=0.45, acg="Fabric080"),
    },
    slots={
        MaterialSlot.FLOOR_LIVING.value: "concrete-poured",
        MaterialSlot.FLOOR_BEDROOM.value: "concrete-bedroom",
        MaterialSlot.FLOOR_KITCHEN.value: "resin-white",
        MaterialSlot.FLOOR_WET.value: "tile-graphite",
        MaterialSlot.FLOOR_OUTDOOR.value: "granite-paving",
        MaterialSlot.FLOOR_CIRCULATION.value: "concrete-poured",
        MaterialSlot.WALL_INTERIOR.value: "paint-gallery",
        MaterialSlot.WALL_ACCENT.value: "paint-graphite",
        MaterialSlot.WALL_WET.value: "resin-white",
        MaterialSlot.WALL_EXTERIOR.value: "render-white",
        MaterialSlot.CEILING.value: "ceiling-flat",
        MaterialSlot.SKIRTING.value: "skirting-flush",
        MaterialSlot.DOOR_LEAF.value: "paint-gallery",
        MaterialSlot.DOOR_FRAME.value: "steel-dark",
        MaterialSlot.WINDOW_FRAME.value: "steel-dark",
        MaterialSlot.GLAZING.value: "glass-clear",
        MaterialSlot.PARAPET.value: "render-white",
        MaterialSlot.COUNTER.value: "corian-white",
        MaterialSlot.JOINERY.value: "oak-smoked",
        MaterialSlot.METAL.value: "steel-dark",
        MaterialSlot.FABRIC.value: "canvas-grey",
        MaterialSlot.WOOD.value: "oak-smoked",
    },
    accent_colour=(0.22, 0.225, 0.23),
    ambient_intensity=0.42,
    sun_intensity=2.8,
    sun_temperature_k=6200.0,
    interior_temperature_k=3400.0,
)


SCHEMES: dict[str, DesignScheme] = {
    s.id: s for s in (_WARM_MINIMAL, _JAPANDI, _MODERN_INDIAN, _MONOCHROME)
}


def default_scheme() -> DesignScheme:
    return _WARM_MINIMAL


def get_scheme(scheme_id: str) -> DesignScheme:
    return SCHEMES.get(scheme_id, _WARM_MINIMAL)


def scheme_summaries() -> list[dict[str, str]]:
    """Scheme metadata for the configurator's picker."""
    return [
        {"id": s.id, "name": s.name, "description": s.description}
        for s in SCHEMES.values()
    ]


def _validate() -> list[str]:
    """Every scheme must fill every slot with a material it actually defines.

    Run as a self-check: a missing slot silently renders a surface default grey,
    and one grey wall in an otherwise finished render is worse than an error.
    """
    problems: list[str] = []
    required = {s.value for s in MaterialSlot}
    for scheme in SCHEMES.values():
        missing = required - set(scheme.slots)
        for slot in sorted(missing):
            problems.append(f"{scheme.id}: slot {slot!r} unassigned")
        for slot, mid in scheme.slots.items():
            if mid not in scheme.materials:
                problems.append(f"{scheme.id}: slot {slot!r} -> unknown material {mid!r}")
    return problems
