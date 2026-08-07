"""Decide the art direction for a plan, autonomously.

Everything upstream of this module is measurement and construction: the walls are
where the drawing says, the materials are physically correct, the composition rules
are sound. What was still missing is the thing a practice supplies before any of
that matters — a *point of view*. Which scheme suits this flat. Whether it should
read warm and low-sun or cool and bright. Which wall in which room gets the accent.
Which room is the hero shot. Without those decisions the pipeline produces a
correct, evenly-lit, unopinionated model, and "unopinionated" is the difference
between a model and a project.

So this module makes those calls, and it makes them without asking.

**Why an LLM here specifically.** This is the one stage that is a judgement rather
than a computation. There is no correct answer to "should this 4 BHK be Japandi or
Modern Indian" — there is a defensible one, argued from the plan: how many bedrooms,
how much glazing, which aspect, the market it sits in. A language model is good at
exactly that kind of argued preference, and it is the only part of this pipeline
where its output cannot be checked against a measurement. Everywhere else it would
be a liability; here it is the right tool.

**It is not trusted blindly.** Every id it returns — scheme, material, catalogue
item, room — is validated against what actually exists, and anything unrecognised is
dropped rather than passed on. Numeric choices are clamped to sane ranges. A model
that invents a material cannot produce a magenta wall; it can only fail to change
one.

**It is not required.** :func:`direct` falls back to a deterministic rule-based
director when no API key is present or the call fails. The CAD path must never
depend on a network service, so the rules encode the same reasoning in a fixed form:
larger flats and more glazing lean warm and minimal, wet-heavy compact plans lean to
the harder-wearing Indian palette, and the hero room is the largest well-proportioned
habitable space.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ..planir import PlanIR, RoomKind
from ..sceneir import SceneIR

__all__ = ["ArtDirection", "direct", "DirectorConfig"]

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass(slots=True)
class DirectorConfig:
    model: str = "anthropic/claude-opus-4.5"
    fallback_models: tuple[str, ...] = ("google/gemini-2.5-pro", "openai/gpt-5")
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = OPENROUTER_URL
    timeout_s: int = 120
    max_tokens: int = 3000
    temperature: float = 0.4
    """Warmer than the extraction stages on purpose. This is a taste decision, not a
    measurement, and a little variety across runs is a feature rather than a defect."""

    enabled: bool = True


@dataclass
class ArtDirection:
    """A complete set of design decisions, ready to apply to a scene."""

    scheme_id: str = ""
    rationale: str = ""
    mood: str = ""
    slot_overrides: dict[str, str] = field(default_factory=dict)
    """Global material slot overrides on top of the chosen scheme."""

    accent_by_room: dict[str, str] = field(default_factory=dict)
    """Room id -> material id for that room's accent wall."""

    hero_room_id: str | None = None
    sun_azimuth_deg: float = 215.0
    sun_elevation_deg: float = 38.0
    sun_temperature_k: float = 5200.0
    interior_temperature_k: float = 3000.0
    exposure: float = -0.1
    source: str = "rules"
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# the brief
# ---------------------------------------------------------------------------

def _brief(plan: PlanIR, scene: SceneIR) -> dict[str, Any]:
    """Compact, factual description of the plan for the director to reason from.

    Deliberately small. Handing over the whole SceneIR would bury the decision in
    vertex buffers; what a designer needs is the programme, the sizes, the aspect and
    the glazing.
    """
    from .catalog import CATALOG
    from ..build.schemes import SCHEMES

    windows_by_room: dict[str, int] = {}
    for op in scene.openings:
        if not op.glazed:
            continue
        # Attribute the window to the nearest room centroid; approximate is fine, we
        # only need "which rooms have daylight".
        best: tuple[float, str] | None = None
        for z in scene.rooms:
            d = math.dist((op.centre[0], op.centre[2]), (z.centroid[0], z.centroid[2]))
            if best is None or d < best[0]:
                best = (d, z.id)
        if best:
            windows_by_room[best[1]] = windows_by_room.get(best[1], 0) + 1

    rooms = []
    for z in sorted(scene.rooms, key=lambda r: -r.area_m2):
        if z.area_m2 < 1.5:
            continue
        rooms.append(
            {
                "id": z.id,
                "name": z.name,
                "kind": z.kind,
                "area_m2": round(z.area_m2, 1),
                "printed_size_m": (
                    [round(z.stated_dimension_m[0], 2), round(z.stated_dimension_m[1], 2)]
                    if z.stated_dimension_m
                    else None
                ),
                "windows": windows_by_room.get(z.id, 0),
                "has_accent_wall": z.accent_wall_id is not None,
            }
        )

    return {
        "unit": plan.units[0].name if plan.units else scene.name,
        "total_area_sqft": round(sum(r.area_sqft for r in plan.rooms), 0),
        "imperial_market": plan.imperial_display,
        "rooms": rooms,
        "schemes": {
            sid: {
                "name": s.name,
                "description": s.description,
                "materials": sorted(s.materials),
            }
            for sid, s in SCHEMES.items()
        },
        "material_slots": sorted({s.slot for s in scene.surfaces}),
        "catalog_items": sorted(CATALOG),
    }


SYSTEM_PROMPT = """You are the design director at an architecture practice known for \
restrained, materially-literate residential interiors. You are handed the measured plan \
of one dwelling and you decide how it should be presented. You decide alone — there is \
no client brief and you must not ask for one.

Return a single JSON object:

{
  "scheme_id": string,            // one of the offered scheme ids
  "rationale": string,            // 1-2 sentences arguing the choice FROM THE PLAN
  "mood": string,                 // short phrase, e.g. "late afternoon, low warm sun"
  "slot_overrides": {             // optional; material slot -> material id
     "<slot>": "<material id from the chosen scheme>"
  },
  "accent_by_room": {             // optional; room id -> material id for its accent wall
     "<room id>": "<material id from the chosen scheme>"
  },
  "hero_room_id": string,         // the room that should lead the presentation
  "sun_azimuth_deg": number,      // 0-360
  "sun_elevation_deg": number,    // 5-70; low sun rakes and models form, high sun flattens
  "sun_temperature_k": number,    // 4000-6800
  "interior_temperature_k": number, // 2500-3600
  "exposure": number              // -1.0 to 0.6 stops
}

How to decide, as you would on a real job:
- Argue the scheme from the plan's own facts: number of bedrooms, total area, how much \
glazing, whether it is a compact family flat or a large one, whether it is an imperial \
(Indian/US) market.
- A low sun elevation (15-30 deg) rakes across surfaces and models relief; a high one \
flattens everything. Choose deliberately and say so in the mood.
- Use slot_overrides sparingly, only where you would genuinely deviate from the scheme \
you picked. An unchanged scheme is a valid decision.
- Only ever use material ids listed under the scheme you chose, and room ids from the \
plan. Do not invent ids.
- hero_room_id should be a generous, well-lit habitable room.

Output only the JSON object. No prose, no markdown fences."""


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    a, b = t.find("{"), t.rfind("}")
    return t[a : b + 1] if a >= 0 and b > a else t


def _call(cfg: DirectorConfig, model: str, brief: dict[str, Any]) -> str:
    key = os.environ.get(cfg.api_key_env, "").strip()
    if not key:
        raise RuntimeError(f"{cfg.api_key_env} not set")
    body = {
        "model": model,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(brief, ensure_ascii=False)},
        ],
    }
    req = urllib.request.Request(
        cfg.base_url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/atlas-plan",
            "X-Title": "Atlas art director",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# rule-based fallback
# ---------------------------------------------------------------------------

def _rules(plan: PlanIR, scene: SceneIR) -> ArtDirection:
    """Deterministic art direction, encoding the same reasoning as fixed rules.

    Used when there is no API key or the model call fails. The CAD path must never
    require a network service, and a defensible default is better than a grey box.
    """
    from ..build.schemes import SCHEMES

    total_sqft = sum(r.area_sqft for r in plan.rooms)
    glazed = sum(1 for o in scene.openings if o.glazed)
    beds = sum(
        1
        for r in plan.rooms
        if r.kind in (RoomKind.BEDROOM, RoomKind.MASTER_BEDROOM, RoomKind.KIDS_BEDROOM)
    )
    wet = sum(1 for r in plan.rooms if r.kind.is_wet)

    if plan.imperial_display and wet >= 3 and total_sqft < 1600:
        scheme_id, why = (
            "modern-indian",
            f"A {beds}-bedroom, {wet}-bathroom plan of {total_sqft:.0f} sqft in an "
            "imperial market: hard stone floors and deep teak suit both the climate and "
            "the buyer.",
        )
    elif total_sqft >= 1600 and glazed >= 6:
        scheme_id, why = (
            "warm-minimal",
            f"{total_sqft:.0f} sqft with {glazed} glazed openings — generous and "
            "daylight-led, so wide oak and lime plaster let the light do the work.",
        )
    elif glazed <= 3:
        scheme_id, why = (
            "japandi",
            f"Only {glazed} glazed opening(s): a low-contrast pale palette keeps the "
            "interior from reading as dim.",
        )
    else:
        scheme_id, why = (
            "warm-minimal",
            f"A {beds}-bedroom plan of {total_sqft:.0f} sqft; the restrained warm "
            "palette is the safe, saleable choice.",
        )

    scheme = SCHEMES[scheme_id]
    hero = None
    best = -1.0
    for z in scene.rooms:
        if z.kind not in (
            RoomKind.LIVING.value,
            RoomKind.BEDROOM.value,
            RoomKind.MASTER_BEDROOM.value,
            RoomKind.DINING.value,
        ):
            continue
        score = z.area_m2 + (2.0 if z.accent_wall_id else 0.0)
        if score > best:
            best, hero = score, z.id

    # Low sun rakes across relief and models form; that is the whole reason to choose
    # an afternoon over a noon.
    return ArtDirection(
        scheme_id=scheme_id,
        rationale=why,
        mood="late afternoon, low raking sun",
        hero_room_id=hero,
        sun_azimuth_deg=232.0,
        sun_elevation_deg=22.0,
        sun_temperature_k=scheme.sun_temperature_k - 200.0,
        interior_temperature_k=scheme.interior_temperature_k,
        exposure=-0.05,
        source="rules",
        notes=["art direction by rules (no model call)"],
    )


# ---------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------

def direct(plan: PlanIR, scene: SceneIR, cfg: DirectorConfig | None = None) -> ArtDirection:
    """Decide the art direction. Never raises; always returns a usable result."""
    cfg = cfg or DirectorConfig()
    fallback = _rules(plan, scene)
    if not cfg.enabled or not os.environ.get(cfg.api_key_env, "").strip():
        return fallback

    from ..build.schemes import SCHEMES

    brief = _brief(plan, scene)
    room_ids = {z.id for z in scene.rooms}
    slots = set(brief["material_slots"])

    last = ""
    for model in (cfg.model, *cfg.fallback_models):
        try:
            data = json.loads(_strip_fences(_call(cfg, model, brief)))

            scheme_id = str(data.get("scheme_id", "")).strip()
            if scheme_id not in SCHEMES:
                raise ValueError(f"unknown scheme {scheme_id!r}")
            known_materials = set(SCHEMES[scheme_id].materials)

            def clamp(key: str, lo: float, hi: float, default: float) -> float:
                try:
                    return max(lo, min(hi, float(data.get(key, default))))
                except (TypeError, ValueError):
                    return default

            # Validate every id against what exists. A model that invents a material
            # must be unable to produce a magenta wall — only to fail to change one.
            overrides = {
                str(k): str(v)
                for k, v in (data.get("slot_overrides") or {}).items()
                if str(k) in slots and str(v) in known_materials
            }
            accents = {
                str(k): str(v)
                for k, v in (data.get("accent_by_room") or {}).items()
                if str(k) in room_ids and str(v) in known_materials
            }
            hero = str(data.get("hero_room_id", "") or "")

            ad = ArtDirection(
                scheme_id=scheme_id,
                rationale=str(data.get("rationale", ""))[:400],
                mood=str(data.get("mood", ""))[:120],
                slot_overrides=overrides,
                accent_by_room=accents,
                hero_room_id=hero if hero in room_ids else fallback.hero_room_id,
                sun_azimuth_deg=clamp("sun_azimuth_deg", 0.0, 360.0, 232.0),
                sun_elevation_deg=clamp("sun_elevation_deg", 5.0, 70.0, 22.0),
                sun_temperature_k=clamp("sun_temperature_k", 4000.0, 6800.0, 5200.0),
                interior_temperature_k=clamp("interior_temperature_k", 2500.0, 3600.0, 3000.0),
                exposure=clamp("exposure", -1.0, 0.6, -0.05),
                source=model,
            )
            dropped = len(data.get("slot_overrides") or {}) - len(overrides)
            ad.notes.append(
                f"art direction by {model}: {ad.scheme_id}, {len(overrides)} slot "
                f"override(s), {len(accents)} room accent(s)"
                + (f", {dropped} unrecognised id(s) dropped" if dropped > 0 else "")
            )
            return ad
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            last = f"{model}: {type(exc).__name__} {exc}"
        except (json.JSONDecodeError, KeyError, ValueError, RuntimeError, TypeError) as exc:
            last = f"{model}: {type(exc).__name__}: {exc}"

    fallback.notes.append(f"model art direction unavailable ({last}); used rules")
    return fallback
