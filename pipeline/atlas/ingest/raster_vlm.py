"""Extract a PlanIR from a raster floor plan using a vision-language model.

The CAD path measures geometry directly. An uploaded PNG or JPEG has no geometry at
all, so something has to *read* the drawing, and the choice of what does that
reading is the main design decision here.

**Why a general vision model rather than a segmentation network.** The obvious
option is CubiCasa5K or DeepFloorplan — pixel-wise segmentation trained on floor
plans. They work, but they share one disqualifying weakness: they recover *shape*
and not *scale*. A segmentation mask says "wall here"; it cannot say the room is
11'0" × 13'6". You then have to guess the scale from a supposed drawing standard,
and every dimension in the output inherits that guess.

The plans this is built for solve that problem themselves — they print the room
name and its dimensions inside every room, and the total carpet area on the sheet:

    BEDROOM          LIVING           3 BHK
    11'0"X13'6"      11'0"X17'0"      CARPET AREA: 1358 SQFT

A vision model *reads text*, so it recovers the dimensions the designer stated, and
those are ground truth rather than inference. That is also the direction the
research has moved: FloorplanVLM (arXiv 2602.06507) reframes vectorisation as
image-to-sequence for the same reason. We use a frontier general model rather than
a fine-tuned 3B one because no training is needed to read a label, and generality
is what survives contact with an unfamiliar drawing convention.

**Structure is derived, not asked for.** The model is asked for rooms, labels,
printed dimensions, adjacency and openings — things visible in the image. It is
*not* asked for a wall centreline graph. Language models are unreliable at
producing metrically consistent coordinate sets, and a plan assembled from
hallucinated coordinates is confidently wrong in a way that is hard to detect. So
the model reports what it sees and this module solves the geometry: rectangles are
sized from the printed dimensions and packed according to the reported adjacency.

**Every number is checked.** Room areas are summed and compared against the stated
carpet area; disagreement is recorded as a warning on the PlanIR rather than
smoothed over. A pipeline that silently invents a building is worse than one that
says it is unsure.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..planir import (
    Level,
    Opening,
    OpeningKind,
    PlanIR,
    PlanUnit,
    Room,
    RoomKind,
    Source,
    Wall,
    classify_room_name,
)
from ..units import mm_to_sqft, parse_area_sqft, parse_dimension_pair

__all__ = ["VlmConfig", "VlmReading", "read_plan_image", "extract_plan_from_image"]

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass(slots=True)
class VlmConfig:
    model: str = "anthropic/claude-opus-4.5"
    """Default is a frontier vision model. The task is reading small text in a
    dense technical drawing and reasoning about adjacency — the hardest part is
    legibility, where model capability translates directly into fewer wrong rooms."""

    fallback_models: tuple[str, ...] = (
        "google/gemini-2.5-pro",
        "openai/gpt-5",
    )
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = OPENROUTER_URL
    timeout_s: int = 240
    max_tokens: int = 8000
    temperature: float = 0.0
    """Zero temperature: this is an extraction task with one correct answer, and
    sampling variety would only produce inconsistent plans between runs."""

    wall_thickness_mm: float = 150.0
    exterior_thickness_mm: float = 230.0
    wall_height_mm: float = 2900.0
    door_width_mm: float = 900.0
    door_height_mm: float = 2100.0
    window_width_mm: float = 1500.0
    window_height_mm: float = 1200.0
    window_sill_mm: float = 900.0


SYSTEM_PROMPT = """You are an architectural drawing analyst. You read 2D residential \
floor plan images and report exactly what is drawn, as structured JSON.

You will be given a floor plan image. Report only what you can actually see. \
Never invent a room, a dimension or an opening that is not visible in the image.

Return a single JSON object with this exact shape:

{
  "unit_name": string,            // e.g. "3 BHK" or "" if not printed
  "stated_area_text": string,     // the carpet/built-up area caption verbatim, or ""
  "imperial": boolean,            // true if dimensions are printed in feet/inches
  "confidence": number,           // 0..1, your overall confidence
  "rooms": [
    {
      "name": string,             // the label exactly as printed, e.g. "MASTER BEDROOM"
      "dimension_text": string,   // exactly as printed, e.g. "11'0\\"X13'6\\"" or ""
      "grid_row": integer,        // 0 = topmost band of the plan, increasing downward
      "grid_col": integer,        // 0 = leftmost, increasing rightward
      "neighbours": [string],     // names of rooms sharing a wall with this one
      "has_window": boolean,      // an exterior window is drawn in this room
      "door_to": [string],        // names of rooms this one has a doorway to
      "is_outdoor": boolean       // balcony, terrace, open deck
    }
  ]
}

Guidance:
- Read every printed dimension character by character. These are the most \
important values in your reply: they are the true size of each room.
- "grid_row"/"grid_col" place rooms approximately relative to each other. Rooms \
side by side share a row; rooms stacked vertically share a column. Approximate \
integers are fine, they only need to preserve the arrangement.
- Indian plan conventions you may meet: "BHK" counts bedrooms; "TOILET" is a \
bathroom; "DRY BAL"/"DRY BALCONY" is a service balcony; "W.I.W" is a walk-in \
wardrobe; "PUJA" is a prayer room; "C.TOILET" is a common bathroom.
- If a room has no printed dimension, still list it with "dimension_text": "".
- Output only the JSON object. No prose, no markdown fences."""


@dataclass
class VlmReading:
    """The model's structured reading, before geometry is solved."""

    unit_name: str = ""
    stated_area_sqft: float | None = None
    imperial: bool = False
    confidence: float = 0.0
    rooms: list[dict[str, Any]] = field(default_factory=list)
    model_used: str = ""
    raw: str = ""
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# model call
# ---------------------------------------------------------------------------

def _encode_image(path: Path) -> tuple[str, str]:
    """Return ``(mime, base64)`` for an image file."""
    ext = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "image/png")
    return mime, base64.b64encode(path.read_bytes()).decode("ascii")


def _strip_fences(text: str) -> str:
    """Pull the JSON object out of a reply that may be fenced or prefaced."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        return t[start : end + 1]
    return t


def _call_openrouter(cfg: VlmConfig, model: str, image_path: Path, extra: str = "") -> str:
    key = os.environ.get(cfg.api_key_env, "").strip()
    if not key:
        raise RuntimeError(
            f"{cfg.api_key_env} is not set. Vision ingest needs an OpenRouter API key; "
            "the CAD path (DWG/DXF) needs no key at all."
        )
    mime, b64 = _encode_image(image_path)
    body = {
        "model": model,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Read this floor plan and return the JSON object described. "
                            + extra
                        ).strip(),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            },
        ],
    }
    req = urllib.request.Request(
        cfg.base_url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/atlas-plan",
            "X-Title": "Atlas floor plan ingest",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if "choices" not in payload:
        raise RuntimeError(f"unexpected API response: {str(payload)[:400]}")
    return payload["choices"][0]["message"]["content"]


def read_plan_image(image_path: Path | str, cfg: VlmConfig | None = None) -> VlmReading:
    """Ask a vision model to read a floor plan image into structured data.

    Tries the configured model, then the fallbacks. Model failure is reported
    rather than silently producing an empty plan.
    """
    cfg = cfg or VlmConfig()
    image_path = Path(image_path)
    reading = VlmReading()

    last_error: str = ""
    for model in (cfg.model, *cfg.fallback_models):
        try:
            raw = _call_openrouter(cfg, model, image_path)
            data = json.loads(_strip_fences(raw))
            reading.raw = raw
            reading.model_used = model
            reading.unit_name = str(data.get("unit_name", "") or "")
            reading.imperial = bool(data.get("imperial", False))
            reading.confidence = float(data.get("confidence", 0.0) or 0.0)
            reading.rooms = [r for r in data.get("rooms", []) if isinstance(r, dict)]
            area_text = str(data.get("stated_area_text", "") or "")
            reading.stated_area_sqft = parse_area_sqft(area_text)
            reading.notes.append(
                f"{model} read {len(reading.rooms)} rooms "
                f"(self-reported confidence {reading.confidence:.2f})"
            )
            if area_text:
                reading.notes.append(f"stated area caption: {area_text!r}")
            return reading
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            detail = ""
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    detail = exc.read().decode("utf-8", "replace")[:300]
                except Exception:
                    detail = ""
            last_error = f"{model}: {type(exc).__name__} {exc} {detail}"
        except (json.JSONDecodeError, KeyError, ValueError, RuntimeError) as exc:
            last_error = f"{model}: {type(exc).__name__}: {exc}"
        reading.notes.append(f"failed: {last_error}")

    raise RuntimeError(f"all vision models failed. last error: {last_error}")


# ---------------------------------------------------------------------------
# geometry solving
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _Cell:
    name: str
    row: int
    col: int
    w: float
    d: float
    kind: RoomKind
    dim_text: str
    outdoor: bool
    has_window: bool
    doors: list[str]
    x: float = 0.0
    y: float = 0.0


def _solve_layout(cells: list[_Cell], cfg: VlmConfig) -> None:
    """Position rooms from their printed sizes and reported arrangement.

    Deliberately not an attempt to reproduce the drawing's exact outline. The model
    gives reliable *sizes* (it read them off the page) and a reliable *relative
    arrangement* (which room is left of which), but not reliable coordinates. The
    packing consumes exactly the information that is trustworthy.

    Rooms are packed **edge to edge within each row**, each at its own width, with one
    wall thickness between neighbours. The first version instead placed every room at a
    fixed column offset whose width was the widest room in that column, which left a
    gap beside every narrower room — and the resulting model read as a scatter of
    disconnected boxes rather than a flat. Consecutive packing removes those gaps, so
    rooms actually share walls.

    Rows are then centred on the plan's overall width rather than left-aligned, because
    a residential plan is roughly a compact block: centring keeps the silhouette read as
    one building instead of a staircase.
    """
    if not cells:
        return
    t = cfg.wall_thickness_mm

    rows = sorted({c.row for c in cells})
    row_cells: dict[int, list[_Cell]] = {
        r: sorted([c for c in cells if c.row == r], key=lambda c: c.col) for r in rows
    }

    # Row widths, so rows can be centred against the widest one.
    row_width: dict[int, float] = {}
    for r in rows:
        group = row_cells[r]
        row_width[r] = sum(c.w for c in group) + t * max(0, len(group) - 1)
    plan_width = max(row_width.values()) if row_width else 0.0

    row_depth = {r: max((c.d for c in row_cells[r]), default=3000.0) for r in rows}

    y_top = 0.0
    for r in rows:
        x = (plan_width - row_width[r]) / 2.0
        for c in row_cells[r]:
            c.x = x
            # Plan Y increases upward, so rows march downward from the top.
            # Shorter rooms hang from the top of their band, which keeps their
            # street-facing edge aligned with their neighbours'.
            c.y = y_top - c.d
            x += c.w + t
        y_top -= row_depth[r] + t


def _rect(x: float, y: float, w: float, d: float) -> list[tuple[float, float]]:
    return [(x, y), (x + w, y), (x + w, y + d), (x, y + d)]


def extract_plan_from_image(
    image_path: Path | str,
    *,
    cfg: VlmConfig | None = None,
    reading: VlmReading | None = None,
) -> PlanIR:
    """Read a raster floor plan and build a PlanIR from it.

    Pass ``reading`` to reuse an earlier model response instead of paying for
    another call.
    """
    cfg = cfg or VlmConfig()
    image_path = Path(image_path)
    reading = reading or read_plan_image(image_path, cfg)

    plan = PlanIR(
        source_file=str(image_path),
        source_kind="raster",
        levels=[Level("L0", "Ground Floor", 0.0, cfg.wall_height_mm)],
        imperial_display=reading.imperial,
    )
    plan.ingest_notes.append(f"vision ingest via {reading.model_used}")
    plan.ingest_notes.extend(reading.notes)

    cells: list[_Cell] = []
    no_dim: list[str] = []
    for i, r in enumerate(reading.rooms):
        name = str(r.get("name", "") or f"Room {i+1}").strip()
        dim_text = str(r.get("dimension_text", "") or "")
        dim = parse_dimension_pair(dim_text)
        kind = classify_room_name(name)
        if dim is None:
            no_dim.append(name)
            # Fall back to a typical size for the programme rather than dropping the
            # room: an unlabelled bathroom is still a bathroom.
            w, d = _typical_size(kind)
        else:
            w, d = dim.width_mm, dim.depth_mm
        cells.append(
            _Cell(
                name=name,
                row=int(r.get("grid_row", 0) or 0),
                col=int(r.get("grid_col", 0) or 0),
                w=w,
                d=d,
                kind=kind,
                dim_text=dim_text,
                outdoor=bool(r.get("is_outdoor", False)) or kind.is_outdoor,
                has_window=bool(r.get("has_window", False)),
                doors=[str(x) for x in (r.get("door_to") or [])],
                y=0.0,
            )
        )

    # Drop common-area service cores. A lift or a stair shaft appears on the sheet
    # because the sheet shows the floor, but it is not part of the dwelling — it is
    # shared, it is not the buyer's carpet area, and including it inflates the plan
    # with rooms that have no doorway into the flat.
    dropped_service = [c.name for c in cells if c.kind in (RoomKind.LIFT, RoomKind.STAIRS)]
    cells = [c for c in cells if c.kind not in (RoomKind.LIFT, RoomKind.STAIRS)]

    if not cells:
        plan.warnings.append(
            "the vision model reported no rooms; the image may not be a floor plan, "
            "or may be too low-resolution to read"
        )
        return plan

    if dropped_service:
        plan.ingest_notes.append(
            f"excluded {len(dropped_service)} common-area core(s) from the dwelling "
            f"({', '.join(dropped_service)})"
        )

    if no_dim:
        plan.warnings.append(
            f"{len(no_dim)} room(s) had no readable printed dimension and were given "
            f"typical sizes ({', '.join(no_dim[:6])}) — check these against the drawing"
        )

    _solve_layout(cells, cfg)

    # --- rooms ---
    for i, c in enumerate(cells, 1):
        poly = _rect(c.x, c.y, c.w, c.d)
        cx = c.x + c.w / 2.0
        cy = c.y + c.d / 2.0
        dim = parse_dimension_pair(c.dim_text)
        plan.rooms.append(
            Room(
                id=f"v-r{i}",
                kind=c.kind,
                label=c.name,
                polygon=poly,
                centroid=(cx, cy),
                stated_dimension=dim,
                ceiling_height_mm=(
                    1050.0 if c.outdoor else cfg.wall_height_mm - (300.0 if c.kind.is_wet else 0.0)
                ),
                floor_level_mm=-50.0 if c.outdoor else 0.0,
                source=Source.VISION,
                confidence=reading.confidence if dim else reading.confidence * 0.6,
                notes=[] if dim else ["size assumed; no printed dimension read"],
            )
        )

    # --- walls: one per room edge, deduplicated where two rooms share a line ---
    seen: dict[tuple[int, int, int, int], str] = {}
    for c in cells:
        poly = _rect(c.x, c.y, c.w, c.d)
        for k in range(4):
            a, b = poly[k], poly[(k + 1) % 4]
            key = tuple(sorted([(round(a[0]), round(a[1])), (round(b[0]), round(b[1]))]))
            flat = (key[0][0], key[0][1], key[1][0], key[1][1])
            if flat in seen:
                continue
            wid = f"v-w{len(plan.walls) + 1}"
            seen[flat] = wid
            plan.walls.append(
                Wall(
                    id=wid,
                    start=a,
                    end=b,
                    thickness_mm=cfg.wall_thickness_mm,
                    height_mm=1050.0 if c.outdoor else cfg.wall_height_mm,
                    kind="parapet" if c.outdoor else "partition",
                    source=Source.VISION,
                    confidence=reading.confidence * 0.8,
                )
            )

    # --- exterior detection: a wall with only one room touching it faces outside ---
    from shapely.geometry import LineString, Polygon as ShpPolygon
    from shapely.ops import unary_union

    room_union = unary_union([ShpPolygon(r.polygon).buffer(1.0) for r in plan.rooms])
    for w in plan.walls:
        mid = w.midpoint
        n = w.normal()
        probe = cfg.wall_thickness_mm / 2.0 + 120.0
        from shapely.geometry import Point as ShpPoint

        left = room_union.contains(ShpPoint(mid[0] + n[0] * probe, mid[1] + n[1] * probe))
        right = room_union.contains(ShpPoint(mid[0] - n[0] * probe, mid[1] - n[1] * probe))
        w.exterior = left != right
        if w.exterior and w.kind != "parapet":
            w.thickness_mm = cfg.exterior_thickness_mm

    # --- openings: doors on shared walls, windows on exterior walls ---
    name_to_room = {c.name.upper(): r for c, r in zip(cells, plan.rooms)}
    placed: set[tuple[str, str]] = set()
    for c in cells:
        src = name_to_room.get(c.name.upper())
        if src is None:
            continue
        for target in c.doors:
            dst = name_to_room.get(target.strip().upper())
            if dst is None or dst is src:
                continue
            pair = tuple(sorted([src.id, dst.id]))
            if pair in placed:
                continue
            wall = _shared_wall(plan, src, dst)
            if wall is None:
                continue
            placed.add(pair)  # type: ignore[arg-type]
            plan.openings.append(
                Opening(
                    id=f"v-o{len(plan.openings) + 1}",
                    wall_id=wall.id,
                    kind=OpeningKind.DOOR,
                    t_center=0.5,
                    width_mm=min(cfg.door_width_mm, wall.length_mm * 0.8),
                    height_mm=cfg.door_height_mm,
                    sill_mm=0.0,
                    swing="left",
                    source=Source.VISION,
                    confidence=reading.confidence * 0.7,
                    connects=(src.id, dst.id),
                )
            )

    for c, room in zip(cells, plan.rooms):
        if not c.has_window:
            continue
        ext = [
            w
            for w in plan.walls
            if w.exterior
            and w.kind != "parapet"
            and LineString([w.start, w.end]).distance(ShpPolygon(room.polygon).exterior) < 200.0
        ]
        if not ext:
            continue
        wall = max(ext, key=lambda w: w.length_mm)
        plan.openings.append(
            Opening(
                id=f"v-o{len(plan.openings) + 1}",
                wall_id=wall.id,
                kind=OpeningKind.WINDOW,
                t_center=0.5,
                width_mm=min(cfg.window_width_mm, wall.length_mm * 0.7),
                height_mm=cfg.window_height_mm,
                sill_mm=cfg.window_sill_mm,
                source=Source.VISION,
                confidence=reading.confidence * 0.6,
            )
        )

    # --- unit and reconciliation against the stated area ---
    measured = sum(r.area_sqft for r in plan.rooms)
    plan.units.append(
        PlanUnit(
            id="v-u1",
            name=reading.unit_name or "Unit 1",
            room_ids=[r.id for r in plan.rooms],
            wall_ids=[w.id for w in plan.walls],
            stated_area_sqft=reading.stated_area_sqft or round(measured, 1),
            bounds=_bounds(plan),
            confidence=reading.confidence,
        )
    )

    if reading.stated_area_sqft:
        ratio = min(measured, reading.stated_area_sqft) / max(measured, reading.stated_area_sqft)
        plan.ingest_notes.append(
            f"room areas total {measured:.0f} sqft against a printed carpet area of "
            f"{reading.stated_area_sqft:.0f} sqft ({ratio:.0%} agreement)"
        )
        # Labelled rooms *undercount* carpet area, because carpet area includes the
        # circulation and wall footprints that carry no room label. So a sum that
        # *exceeds* the printed figure is the suspicious direction, not the reassuring
        # one: it means the sheet showed more than one dwelling and rooms from a
        # neighbour were read in. Reporting a plain ratio hid that — on the 2.5 BHK
        # sample, two lift cores and an adjacent flat's bedroom pushed the sum up to
        # almost exactly the printed figure and looked like a 98% pass.
        if measured > reading.stated_area_sqft * 1.02:
            plan.warnings.append(
                f"labelled rooms total {measured:.0f} sqft, MORE than the printed carpet "
                f"area of {reading.stated_area_sqft:.0f} sqft. Carpet area includes "
                "unlabelled circulation, so the labels should sum to less. The sheet "
                "probably shows more than one dwelling and rooms from a neighbouring "
                "unit have been included — check the room list."
            )
        elif ratio < 0.7:
            plan.warnings.append(
                f"labelled rooms total {measured:.0f} sqft against a printed carpet area "
                f"of {reading.stated_area_sqft:.0f} sqft ({ratio:.0%}) — a dimension may "
                "have been misread or a room missed"
            )

    plan.ingest_notes.append(
        f"solved {len(plan.rooms)} rooms, {len(plan.walls)} walls, {len(plan.openings)} openings "
        "on a grid packed from the printed dimensions"
    )
    plan.warnings.append(
        "geometry from a raster image is reconstructed, not measured: room sizes come from "
        "the printed dimensions but the arrangement is approximated on a grid. Supply DWG/DXF "
        "for exact geometry."
    )
    return plan


def _typical_size(kind: RoomKind) -> tuple[float, float]:
    """Fallback room size when no dimension is printed, in mm.

    Values are ordinary Indian residential sizes so an unlabelled room comes out
    plausible rather than arbitrary.
    """
    table = {
        RoomKind.LIVING: (3350, 4600),
        RoomKind.DINING: (1830, 3050),
        RoomKind.KITCHEN: (2440, 2740),
        RoomKind.BEDROOM: (3050, 3350),
        RoomKind.MASTER_BEDROOM: (3350, 3960),
        RoomKind.KIDS_BEDROOM: (2740, 3050),
        RoomKind.BATHROOM: (1370, 2130),
        RoomKind.POWDER: (1220, 1520),
        RoomKind.BALCONY: (1220, 3050),
        RoomKind.DRY_BALCONY: (1070, 1830),
        RoomKind.UTILITY: (1520, 1830),
        RoomKind.WARDROBE: (1520, 1830),
        RoomKind.LOBBY: (1520, 1830),
        RoomKind.CORRIDOR: (1070, 2440),
        RoomKind.STUDY: (2440, 2740),
        RoomKind.PUJA: (1070, 1220),
        RoomKind.STORE: (1220, 1520),
    }
    return table.get(kind, (2440, 2740))


def _shared_wall(plan: PlanIR, a: Room, b: Room) -> Wall | None:
    """The wall segment lying between two rooms, if they touch."""
    from shapely.geometry import LineString, Polygon as ShpPolygon

    ra = ShpPolygon(a.polygon).exterior
    rb = ShpPolygon(b.polygon).exterior
    best: tuple[float, Wall] | None = None
    for w in plan.walls:
        line = LineString([w.start, w.end])
        da, db = line.distance(ra), line.distance(rb)
        if da < 250.0 and db < 250.0:
            score = da + db
            if best is None or score < best[0]:
                best = (score, w)
    return best[1] if best else None


def _bounds(plan: PlanIR) -> tuple[float, float, float, float]:
    xs = [p[0] for r in plan.rooms for p in r.polygon]
    ys = [p[1] for r in plan.rooms for p in r.polygon]
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))
