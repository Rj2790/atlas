"""Render a plan to SVG, for verification by eye.

Extraction quality is a geometric property, and geometric bugs are close to
invisible in numbers but obvious in a picture: a wall that missed its pair, a room
that leaked through a doorway, an opening attached to the wall behind it. This
renderer exists so both the developer and the end user can check the machine's
reading of the drawing against the drawing itself.

Two modes:

* ``render_plan`` — a clean presentation drawing of the extracted PlanIR.
* ``render_debug`` — the same, over the raw input line work, so you can see
  exactly what was consumed and what was ignored.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..planir import OpeningKind, PlanIR, Room, RoomKind, Wall
from ..units import format_feet_inches

__all__ = ["SvgStyle", "render_plan", "render_debug"]


# A restrained architectural palette: rooms tinted by programme, walls solid
# graphite. Reads as a drawing rather than a debug dump.
_ROOM_FILL: dict[RoomKind, str] = {
    RoomKind.LIVING: "#e8e2d9",
    RoomKind.DINING: "#e5e0d6",
    RoomKind.KITCHEN: "#dfe4e2",
    RoomKind.BEDROOM: "#e9e4dc",
    RoomKind.MASTER_BEDROOM: "#eae3d7",
    RoomKind.KIDS_BEDROOM: "#e7e5dd",
    RoomKind.BATHROOM: "#dde5e8",
    RoomKind.POWDER: "#dde5e8",
    RoomKind.BALCONY: "#e3e7e0",
    RoomKind.DRY_BALCONY: "#e0e4de",
    RoomKind.UTILITY: "#e0e2e0",
    RoomKind.STORE: "#e4e2de",
    RoomKind.WARDROBE: "#e6e1d8",
    RoomKind.LOBBY: "#eceae4",
    RoomKind.CORRIDOR: "#eceae4",
    RoomKind.STAIRS: "#dcdcdc",
    RoomKind.LIFT: "#d6d6d6",
    RoomKind.DUCT: "#cfcfcf",
    RoomKind.TERRACE: "#e3e7e0",
    RoomKind.STUDY: "#e8e4db",
    RoomKind.PUJA: "#ece5d5",
    RoomKind.UNKNOWN: "#f4dede",
}


@dataclass(slots=True)
class SvgStyle:
    width_px: int = 1400
    margin_px: int = 48
    wall_fill: str = "#2f3336"
    parapet_fill: str = "#8b9096"
    background: str = "#fbfaf8"
    label_colour: str = "#33383b"
    dim_colour: str = "#7d8489"
    font: str = "Inter, 'Helvetica Neue', Arial, sans-serif"
    show_labels: bool = True
    show_dimensions: bool = True
    show_openings: bool = True
    show_ids: bool = False
    show_furniture: bool = False
    """Overlay furniture footprints. The fastest way to judge a layout: a sofa in a
    doorway or a bed through a wall is invisible in a table of coordinates and obvious
    in plan."""


def _bounds(plan: PlanIR, extra: Iterable[Sequence[tuple[float, float]]] = ()) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for w in plan.walls:
        xs += [w.start[0], w.end[0]]
        ys += [w.start[1], w.end[1]]
    for r in plan.rooms:
        for x, y in r.polygon:
            xs.append(x)
            ys.append(y)
    for seq in extra:
        for x, y in seq:
            xs.append(x)
            ys.append(y)
    if not xs:
        return (0.0, 0.0, 1000.0, 1000.0)
    return (min(xs), min(ys), max(xs), max(ys))


class _Canvas:
    """Maps drawing millimetres to SVG pixels, flipping Y (CAD is Y-up)."""

    def __init__(self, bounds: tuple[float, float, float, float], style: SvgStyle):
        self.style = style
        x0, y0, x1, y1 = bounds
        w_mm = max(x1 - x0, 1.0)
        h_mm = max(y1 - y0, 1.0)
        inner = style.width_px - 2 * style.margin_px
        self.k = inner / w_mm
        self.w = style.width_px
        self.h = int(h_mm * self.k) + 2 * style.margin_px
        self.x0, self.y1 = x0, y1

    def p(self, pt: tuple[float, float]) -> tuple[float, float]:
        return (
            self.style.margin_px + (pt[0] - self.x0) * self.k,
            self.style.margin_px + (self.y1 - pt[1]) * self.k,
        )

    def path(self, pts: Sequence[tuple[float, float]], close: bool = True) -> str:
        if not pts:
            return ""
        d = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in (self.p(q) for q in pts))
        return d + (" Z" if close else "")


def _wall_quad(w: Wall) -> list[tuple[float, float]]:
    """The wall's footprint rectangle from its centreline and thickness."""
    nx, ny = w.normal()
    h = w.thickness_mm / 2.0
    return [
        (w.start[0] + nx * h, w.start[1] + ny * h),
        (w.end[0] + nx * h, w.end[1] + ny * h),
        (w.end[0] - nx * h, w.end[1] - ny * h),
        (w.start[0] - nx * h, w.start[1] - ny * h),
    ]


def _room_label(room: Room, imperial: bool) -> tuple[str, str]:
    name = (room.label or room.kind.value.replace("_", " ")).upper()
    if room.stated_dimension is not None:
        d = room.stated_dimension
        if imperial:
            sub = f"{format_feet_inches(d.width_mm)}×{format_feet_inches(d.depth_mm)}"
        else:
            sub = f"{d.width_mm/1000:.2f}×{d.depth_mm/1000:.2f} m"
    else:
        sub = f"{room.area_sqft:.0f} sq ft"
    return name, sub


def render_plan(
    plan: PlanIR, style: SvgStyle | None = None, scene: object | None = None
) -> str:
    """Render the extracted plan as a presentation-quality SVG.

    Pass ``scene`` (a SceneIR) with ``style.show_furniture`` to overlay the layout.
    """
    style = style or SvgStyle()
    cv = _Canvas(_bounds(plan), style)
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{cv.w}" height="{cv.h}" '
        f'viewBox="0 0 {cv.w} {cv.h}" font-family="{style.font}">',
        f'<rect width="{cv.w}" height="{cv.h}" fill="{style.background}"/>',
    ]

    parts.append('<g id="rooms">')
    for r in plan.rooms:
        if len(r.polygon) < 3:
            continue
        fill = _ROOM_FILL.get(r.kind, "#eeeeee")
        parts.append(
            f'<path d="{cv.path(r.polygon)}" fill="{fill}" stroke="#c9c3b8" stroke-width="0.6"/>'
        )
    parts.append("</g>")

    parts.append('<g id="walls">')
    for w in plan.walls:
        fill = style.parapet_fill if w.kind == "parapet" else style.wall_fill
        op = "0.55" if w.kind == "parapet" else ("1" if w.confidence >= 0.7 else "0.45")
        parts.append(f'<path d="{cv.path(_wall_quad(w))}" fill="{fill}" opacity="{op}"/>')
    parts.append("</g>")

    if style.show_openings:
        parts.append('<g id="openings">')
        for o in plan.openings:
            w = plan.wall_by_id(o.wall_id)
            if w is None or w.length_mm <= 0:
                continue
            half_t = w.thickness_mm / 2.0 + 30.0
            nx, ny = w.normal()
            dx, dy = w.direction()
            c = w.point_at(o.t_center)
            hw = o.width_mm / 2.0
            quad = [
                (c[0] - dx * hw + nx * half_t, c[1] - dy * hw + ny * half_t),
                (c[0] + dx * hw + nx * half_t, c[1] + dy * hw + ny * half_t),
                (c[0] + dx * hw - nx * half_t, c[1] + dy * hw - ny * half_t),
                (c[0] - dx * hw - nx * half_t, c[1] - dy * hw - ny * half_t),
            ]
            colour = "#f2efe9" if o.kind.is_door else "#bfd4e0"
            parts.append(
                f'<path d="{cv.path(quad)}" fill="{colour}" stroke="#6b7378" stroke-width="0.7"/>'
            )
            if o.kind is OpeningKind.WINDOW or o.kind is OpeningKind.FRENCH_WINDOW:
                a = cv.p((c[0] - dx * hw, c[1] - dy * hw))
                b = cv.p((c[0] + dx * hw, c[1] + dy * hw))
                parts.append(
                    f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" '
                    'stroke="#5c8fa8" stroke-width="1.2"/>'
                )
        parts.append("</g>")

    if style.show_labels:
        parts.append('<g id="labels" text-anchor="middle">')
        for r in plan.rooms:
            if len(r.polygon) < 3 or r.area_mm2 < 1_200_000:
                continue
            cx, cy = cv.p(r.centroid)
            name, sub = _room_label(r, plan.imperial_display)
            size = max(7.0, min(12.0, cv.k * 900))
            parts.append(
                f'<text x="{cx:.1f}" y="{cy:.1f}" font-size="{size:.1f}" font-weight="600" '
                f'letter-spacing="0.6" fill="{style.label_colour}">{html.escape(name)}</text>'
            )
            if style.show_dimensions:
                parts.append(
                    f'<text x="{cx:.1f}" y="{cy + size * 1.25:.1f}" font-size="{size * 0.82:.1f}" '
                    f'fill="{style.dim_colour}">{html.escape(sub)}</text>'
                )
            if style.show_ids:
                parts.append(
                    f'<text x="{cx:.1f}" y="{cy - size * 1.2:.1f}" font-size="{size*0.7:.1f}" '
                    f'fill="#b04a4a">{html.escape(r.id)}</text>'
                )
        parts.append("</g>")

    if style.show_furniture and scene is not None:
        import math as _m

        ox, oy = getattr(scene, "origin_offset", (0.0, 0.0))

        def back(x_m: float, z_m: float) -> tuple[float, float]:
            """Scene metres -> plan millimetres."""
            return (x_m * 1000.0 + ox, -(z_m * 1000.0) + oy)

        parts.append('<g id="furniture">')
        for pl in getattr(scene, "furniture", []):
            w, d = pl.footprint
            if w <= 0 or d <= 0:
                continue
            cx, _, cz = pl.position
            cos_y, sin_y = _m.cos(pl.rotation_y), _m.sin(pl.rotation_y)
            corners = []
            for lx, lz in ((-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)):
                corners.append(back(cx + lx * cos_y + lz * sin_y, cz - lx * sin_y + lz * cos_y))
            colour = "#c2185b" if pl.source == "dressing" else "#1565c0"
            parts.append(
                f'<path d="{cv.path(corners)}" fill="{colour}" fill-opacity="0.16" '
                f'stroke="{colour}" stroke-width="1.1"/>'
            )
            # A tick showing which way it faces — a bed facing the wall is a bug you
            # can only see if the direction is drawn.
            fx, fz = _m.sin(pl.rotation_y), _m.cos(pl.rotation_y)
            a = cv.p(back(cx, cz))
            b = cv.p(back(cx + fx * (d / 2 + 0.25), cz + fz * (d / 2 + 0.25)))
            parts.append(
                f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" '
                f'stroke="{colour}" stroke-width="1.4"/>'
            )
        parts.append("</g>")

    parts.append("</svg>")
    return "\n".join(parts)


def render_debug(
    plan: PlanIR,
    *,
    raw_segments: Sequence[tuple[tuple[float, float], tuple[float, float], str]] = (),
    label_points: Sequence[tuple[tuple[float, float], str]] = (),
    boundary: Sequence[tuple[float, float]] = (),
    style: SvgStyle | None = None,
) -> str:
    """Overlay the extraction on the raw input, to see what was consumed.

    Raw line work is drawn thin and coloured by layer role; solved walls are drawn
    as translucent bands over it. Where a magenta raw line has no band over it,
    the solver missed a wall — which is precisely the failure that is impossible to
    spot in a summary table.
    """
    style = style or SvgStyle()
    extra: list[Sequence[tuple[float, float]]] = [[a, b] for a, b, _ in raw_segments]
    if boundary:
        extra.append(list(boundary))
    cv = _Canvas(_bounds(plan, extra), style)

    role_colour = {
        "wall": "#d81b60",
        "wall_low": "#8e24aa",
        "door": "#1e88e5",
        "window": "#00897b",
        "window_sill": "#00acc1",
        "column": "#5e35b1",
    }

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{cv.w}" height="{cv.h}" '
        f'viewBox="0 0 {cv.w} {cv.h}" font-family="{style.font}">',
        f'<rect width="{cv.w}" height="{cv.h}" fill="#ffffff"/>',
    ]

    if boundary:
        parts.append(
            f'<path d="{cv.path(list(boundary))}" fill="#fff8e1" stroke="#f9a825" '
            'stroke-width="1.5" stroke-dasharray="6 3"/>'
        )

    parts.append('<g id="rooms" opacity="0.55">')
    for r in plan.rooms:
        if len(r.polygon) < 3:
            continue
        parts.append(
            f'<path d="{cv.path(r.polygon)}" fill="{_ROOM_FILL.get(r.kind, "#eee")}" '
            'stroke="#8d6e63" stroke-width="1"/>'
        )
    parts.append("</g>")

    parts.append('<g id="raw" stroke-width="0.9">')
    for a, b, role in raw_segments:
        pa, pb = cv.p(a), cv.p(b)
        c = role_colour.get(role, "#bdbdbd")
        parts.append(
            f'<line x1="{pa[0]:.1f}" y1="{pa[1]:.1f}" x2="{pb[0]:.1f}" y2="{pb[1]:.1f}" stroke="{c}"/>'
        )
    parts.append("</g>")

    parts.append('<g id="solved">')
    for w in plan.walls:
        colour = "#2e7d32" if w.confidence >= 0.7 else "#ef6c00"
        parts.append(
            f'<path d="{cv.path(_wall_quad(w))}" fill="{colour}" opacity="0.35" '
            f'stroke="{colour}" stroke-width="0.5"/>'
        )
        a, b = cv.p(w.start), cv.p(w.end)
        parts.append(
            f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" '
            'stroke="#1b5e20" stroke-width="0.8" stroke-dasharray="3 2"/>'
        )
    parts.append("</g>")

    parts.append('<g id="label-pts">')
    for pt, txt in label_points:
        x, y = cv.p(pt)
        first = txt.strip().splitlines()[0][:16]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#c62828"/>')
        parts.append(
            f'<text x="{x + 5:.1f}" y="{y - 4:.1f}" font-size="9" fill="#b71c1c">{html.escape(first)}</text>'
        )
    parts.append("</g>")

    legend = [
        ("raw wall line work", "#d81b60"),
        ("raw door", "#1e88e5"),
        ("raw window", "#00897b"),
        ("solved wall (confident)", "#2e7d32"),
        ("solved wall (assumed)", "#ef6c00"),
        ("room label", "#c62828"),
    ]
    parts.append(f'<g id="legend" transform="translate(12,{cv.h - 14 * len(legend) - 12})">')
    for i, (text, colour) in enumerate(legend):
        parts.append(f'<rect x="0" y="{i*14}" width="10" height="10" fill="{colour}"/>')
        parts.append(f'<text x="15" y="{i*14 + 9}" font-size="10" fill="#333">{text}</text>')
    parts.append("</g></svg>")
    return "\n".join(parts)
