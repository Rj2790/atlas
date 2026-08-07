"""Flatten a DXF's nested block hierarchy into world-space primitives.

Real architectural DWGs are not flat. The demo file in ``data/samples`` has 33
entities in modelspace and 8,104 block definitions: every wall, door and piece of
furniture lives inside a block that is INSERTed by another block that is INSERTed
by modelspace. Reading modelspace alone finds nothing. Every floor-plan-to-3D
project we surveyed reads the top level only, which is why they work on toy
exports and fail on drawings that came out of a practice.

Two problems have to be solved together:

**Transform composition.** Each INSERT carries a translation, per-axis scale
(which may be negative — the demo file mirrors a unit with ``xscale=-1``),
rotation, and an extrusion/OCS. A child's coordinates are meaningless until every
ancestor transform is applied. ezdxf exposes ``virtual_entities()`` which does
this correctly one level at a time, so we recurse with it rather than
reimplementing the matrix maths.

**Layer identity.** When an external reference is bound, AutoCAD rewrites its
layer names to ``<xref name>$<n>$<original layer>``. The demo file contains
``23] 29.05.2023_PLOT B CONCEPT 11_100 OS$0$ABA_02_wall``, which is the same
physical layer as ``ABA_02_wall`` but will not match any naming rule. We
normalise these back to the original name so downstream classification sees one
layer, not two.

There is also the ``BYBLOCK`` layer-0 convention: an entity drawn on layer ``0``
inside a block inherits the layer of the INSERT that placed it. We honour that,
because in practice it is how a lot of furniture blocks are authored.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterator

__all__ = ["FlatEntity", "FlattenStats", "normalise_layer", "flatten_layout"]

# Xref-bound layers look like "<xref>$<digit>$<real layer>", and can nest
# ("a$0$b$0$c") when an xref itself contained an xref. Taking the text after the
# final "$<digits>$" recovers the original name in both cases.
_XREF_LAYER_RE = re.compile(r"^.*\$\d+\$(?P<name>.+)$")

# Blocks whose names start with "*" are AutoCAD anonymous blocks (hatch patterns,
# dimension geometry). "zw$" is ZWCAD's equivalent. We still traverse them —
# they carry real geometry in this file — but flag them so callers can weight
# layer evidence from them lower if needed.
_ANON_PREFIXES = ("*", "zw$", "_")


def normalise_layer(name: str) -> str:
    """Strip xref binding prefixes and normalise whitespace/case-insensitively.

    ``"PLOT B ... OS$0$ABA_02_wall"`` -> ``"ABA_02_wall"``.
    """
    if not name:
        return "0"
    m = _XREF_LAYER_RE.match(name)
    if m:
        name = m.group("name")
    return name.strip()


@dataclass(slots=True)
class FlatEntity:
    """A DXF entity resolved into world space, with its provenance kept.

    Provenance matters: knowing an entity arrived via the ``A-FLOR-FURN`` block
    ``pallet2200tv`` lets the furniture stage recognise a sofa block by name even
    when its layer is uninformative.
    """

    dxftype: str
    layer: str
    """Normalised layer name."""

    points: list[tuple[float, float]]
    """World-space 2D footprint of the entity, in drawing units."""

    closed: bool = False
    text: str | None = None
    block_path: tuple[str, ...] = ()
    """Chain of block names traversed to reach this entity."""

    depth: int = 0
    raw: Any = None
    """The virtual entity, for attribute access the caller may need."""

    @property
    def block_name(self) -> str | None:
        return self.block_path[-1] if self.block_path else None


@dataclass
class FlattenStats:
    visited_inserts: int = 0
    emitted: int = 0
    max_depth_reached: int = 0
    truncated_at_depth: int = 0
    skipped_cycles: int = 0
    errors: int = 0
    error_samples: list[str] = field(default_factory=list)
    layer_counts: dict[str, int] = field(default_factory=dict)
    block_counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"flattened {self.emitted} entities from {self.visited_inserts} INSERTs "
            f"(max depth {self.max_depth_reached}, {self.truncated_at_depth} depth-truncated, "
            f"{self.skipped_cycles} cycles skipped, {self.errors} errors)"
        )


# Entity types we can reduce to a 2D footprint. Everything else (3D solids,
# OLE frames, viewports) is irrelevant to plan extraction.
_GEOMETRIC = frozenset(
    {"LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE", "SOLID", "HATCH"}
)
_TEXTUAL = frozenset({"TEXT", "MTEXT", "ATTRIB"})


def _mtext_plain(raw: str) -> str:
    """Strip MTEXT inline formatting codes down to readable text."""
    s = raw
    s = re.sub(r"\\P", "\n", s)          # paragraph break
    s = re.sub(r"\\[Ff][^;]*;", "", s)   # font
    s = re.sub(r"\\[A-Za-z][^;]*;", "", s)  # other \X...; codes
    s = re.sub(r"[{}]", "", s)           # grouping braces
    s = re.sub(r"\\(.)", r"\1", s)       # escaped literals
    return s.strip()


def _arc_points(cx: float, cy: float, r: float, a0: float, a1: float, segs: int = 16):
    """Sample an arc. Plan geometry needs shape fidelity, not analytic curves."""
    if a1 < a0:
        a1 += 360.0
    span = math.radians(a1 - a0)
    start = math.radians(a0)
    n = max(2, min(64, int(segs * max(1.0, span / math.pi))))
    return [
        (cx + r * math.cos(start + span * i / n), cy + r * math.sin(start + span * i / n))
        for i in range(n + 1)
    ]


def _footprint(e: Any) -> tuple[list[tuple[float, float]], bool]:
    """Reduce a world-space entity to (2D points, is_closed)."""
    t = e.dxftype()
    try:
        if t == "LINE":
            s, x = e.dxf.start, e.dxf.end
            return [(s.x, s.y), (x.x, x.y)], False

        if t == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in e.get_points("xy")]
            return pts, bool(e.closed)

        if t == "POLYLINE":
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
            return pts, bool(e.is_closed)

        if t == "CIRCLE":
            c, r = e.dxf.center, e.dxf.radius
            return _arc_points(c.x, c.y, r, 0.0, 360.0), True

        if t == "ARC":
            c = e.dxf.center
            return _arc_points(c.x, c.y, e.dxf.radius, e.dxf.start_angle, e.dxf.end_angle), False

        if t in ("ELLIPSE", "SPLINE"):
            # flattening() yields a polyline approximation at a given sagitta.
            pts = [(p.x, p.y) for p in e.flattening(distance=8.0)]
            closed = t == "ELLIPSE" or bool(getattr(e, "closed", False))
            return pts, closed

        if t == "SOLID":
            pts = []
            for name in ("vtx0", "vtx1", "vtx2", "vtx3"):
                v = e.dxf.get(name, None)
                if v is not None:
                    pts.append((v.x, v.y))
            return pts, True

        if t == "HATCH":
            # Hatches mark filled regions — a room fill or a wall poché. The outer
            # boundary is the useful part and is strong evidence of a region.
            pts: list[tuple[float, float]] = []
            for path in e.paths:
                if hasattr(path, "vertices"):
                    pts.extend((v[0], v[1]) for v in path.vertices)
                    break
            return pts, True
    except Exception:
        return [], False
    return [], False


def flatten_layout(
    layout,
    *,
    max_depth: int = 12,
    max_entities: int = 4_000_000,
    include_anonymous: bool = True,
) -> tuple[list[FlatEntity], FlattenStats]:
    """Recursively resolve ``layout`` into world-space :class:`FlatEntity` items.

    ``max_depth`` guards against pathological nesting; ``max_entities`` against
    a combinatorial block explosion. Both are reported in the stats so a caller
    can tell "nothing there" from "we stopped early".
    """
    stats = FlattenStats()
    out: list[FlatEntity] = []

    def emit(e: Any, layer: str, path: tuple[str, ...], depth: int) -> None:
        t = e.dxftype()
        if t in _TEXTUAL:
            raw = e.dxf.text if t in ("TEXT", "ATTRIB") else e.text
            txt = _mtext_plain(raw or "")
            if not txt:
                return
            try:
                ip = e.dxf.insert if t == "MTEXT" else e.dxf.get("insert", e.dxf.get("align_point"))
                pt = [(ip.x, ip.y)]
            except Exception:
                pt = []
            out.append(FlatEntity(t, layer, pt, False, txt, path, depth, e))
        elif t in _GEOMETRIC:
            pts, closed = _footprint(e)
            if len(pts) < 2:
                return
            out.append(FlatEntity(t, layer, pts, closed, None, path, depth, e))
        else:
            return

        stats.emitted += 1
        stats.layer_counts[layer] = stats.layer_counts.get(layer, 0) + 1
        if path:
            stats.block_counts[path[-1]] = stats.block_counts.get(path[-1], 0) + 1

    def walk(entities: Iterator[Any], path: tuple[str, ...], depth: int, parent_layer: str) -> None:
        stats.max_depth_reached = max(stats.max_depth_reached, depth)
        if depth > max_depth:
            stats.truncated_at_depth += 1
            return

        for e in entities:
            if stats.emitted >= max_entities:
                return
            try:
                t = e.dxftype()
                raw_layer = getattr(e.dxf, "layer", "0") or "0"
                # Layer 0 inside a block inherits the INSERT's layer (BYBLOCK).
                layer = parent_layer if (raw_layer == "0" and depth > 0) else normalise_layer(raw_layer)

                if t == "INSERT":
                    name = e.dxf.name or ""
                    if not include_anonymous and name.startswith(_ANON_PREFIXES):
                        continue
                    # A block that INSERTs an ancestor would recurse forever.
                    if name in path:
                        stats.skipped_cycles += 1
                        continue
                    stats.visited_inserts += 1
                    # Attributes (ATTRIB) often hold the room name or a tag.
                    try:
                        for attrib in e.attribs:
                            emit(attrib, layer, path + (name,), depth + 1)
                    except Exception:
                        pass
                    # virtual_entities() applies this INSERT's full transform to
                    # its children, so recursion composes transforms correctly.
                    walk(e.virtual_entities(), path + (name,), depth + 1, layer)
                else:
                    emit(e, layer, path, depth)
            except Exception as exc:  # noqa: BLE001 - one bad entity must not kill ingest
                stats.errors += 1
                if len(stats.error_samples) < 8:
                    stats.error_samples.append(f"{type(exc).__name__}: {exc}")

    walk(iter(layout), (), 0, "0")
    return out, stats
