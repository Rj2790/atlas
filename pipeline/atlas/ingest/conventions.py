"""Infer a drawing's layer conventions with an LLM — only when heuristics fail.

Every geometric decision in this pipeline is deterministic, and stays so. What an
LLM is genuinely good at is the one step that is *linguistic*: reading layer names
written by a stranger. The rule table in :mod:`atlas.ingest.layers` knows AIA/NCS
and the common Indian-office conventions, but a drawing whose author named the
wall layer ``MUR_PORTEUR`` or ``x-arch-w1`` classifies as UNKNOWN, and everything
downstream starves.

So this stage:

* engages only when a meaningful share of a drawing's geometry sits on layers the
  rule table cannot classify — a drawing whose conventions we already know never
  makes a network call and never risks an LLM overruling a working heuristic;
* sends layer *names and statistics* (entity counts, total length), never
  entities — the model maps vocabulary to roles, it does not draw;
* returns a strict JSON mapping onto the existing :class:`LayerRole` enum,
  validated before use, recorded in the plan's ingest notes with the model name;
* degrades to a note when there is no API key, no network, or no parseable
  answer. The CAD path must keep working on an air-gapped machine.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .layers import LayerRole, classify_layer

__all__ = ["ConventionConfig", "infer_layer_conventions"]

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass(slots=True)
class ConventionConfig:
    model: str = "anthropic/claude-sonnet-5"
    """A text-only classification over a few dozen layer names — a mid-tier model
    is fully adequate and an order of magnitude cheaper than a frontier one."""

    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = OPENROUTER_URL
    timeout_s: int = 60
    max_tokens: int = 2000

    engage_unknown_share: float = 0.25
    """Ask only when at least this share of geometric entities sits on layers the
    rule table cannot classify. Below it, the heuristics are doing their job."""

    min_confidence: float = 0.6
    """Mappings the model itself marks below this are discarded."""


_ROLE_NAMES = sorted(r.value for r in LayerRole)

_PROMPT = """You are reading the layer table of an architectural CAD floor plan.
Map each layer name to exactly one role from this fixed list:

{roles}

Rules:
- "wall" is load-bearing or partition wall linework; "hatch_wall" is wall poché/fill.
- "door" and "window" are opening symbols; "furniture" and "sanitary" are fittings.
- Use "ignore" for title blocks, viewports, revision clouds, grids, and plot junk.
- Use "unknown" when the name plus statistics genuinely do not tell you.
- Do not guess a structural role from decoration-sounding names.

Layers (name, then entity statistics):
{table}

Answer with ONLY a JSON object: {{"layers": [{{"layer": str, "role": str,
"confidence": float 0..1, "why": str}}]}}.
"""


def infer_layer_conventions(
    entities: Iterable[Any],
    *,
    cfg: ConventionConfig | None = None,
    transport: Callable[[str, dict[str, Any]], str] | None = None,
) -> tuple[dict[str, LayerRole], list[str]]:
    """Return {layer name -> role} overrides for unknown layers, plus notes.

    ``transport`` exists for tests: it receives (url, payload) and returns the raw
    response body, so the prompt construction and answer validation are testable
    without a network or a key.
    """
    cfg = cfg or ConventionConfig()
    notes: list[str] = []

    stats: dict[str, Counter[str]] = defaultdict(Counter)
    length: dict[str, float] = defaultdict(float)
    unknown_count = 0
    total = 0
    for e in entities:
        pts = getattr(e, "points", None) or []
        if getattr(e, "text", None) is not None or len(pts) < 2:
            continue
        total += 1
        layer = getattr(e, "layer", "") or ""
        stats[layer][getattr(e, "dxftype", "?")] += 1
        for a, b in zip(pts, pts[1:]):
            length[layer] += ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        if classify_layer(layer).role is LayerRole.UNKNOWN:
            unknown_count += 1

    if total == 0:
        return {}, notes
    share = unknown_count / total
    if share < cfg.engage_unknown_share:
        return {}, notes

    unknown_layers = [
        lay for lay in stats if classify_layer(lay).role is LayerRole.UNKNOWN
    ]
    if not unknown_layers:
        return {}, notes

    api_key = os.environ.get(cfg.api_key_env, "")
    if transport is None and not api_key:
        notes.append(
            f"{share:.0%} of geometry sits on layers with unrecognised names "
            f"({', '.join(sorted(unknown_layers)[:6])}); convention inference "
            "skipped — no OPENROUTER_API_KEY. Extraction proceeds on heuristics."
        )
        return {}, notes

    rows = []
    for lay in sorted(stats):
        mark = " (UNKNOWN)" if lay in unknown_layers else ""
        rows.append(
            f"- {lay!r}{mark}: {dict(stats[lay])}, total length {length[lay]:.0f}"
        )
    prompt = _PROMPT.format(roles=", ".join(_ROLE_NAMES), table="\n".join(rows))
    payload = {
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        if transport is not None:
            body = transport(cfg.base_url, payload)
        else:
            req = urllib.request.Request(
                cfg.base_url,
                data=json.dumps(payload).encode(),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                body = resp.read().decode()
        data = json.loads(body)
        content = data["choices"][0]["message"]["content"]
        # Models occasionally fence the JSON despite instructions.
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            content = content[content.index("{") :]
        parsed = json.loads(content)
    except Exception as exc:  # noqa: BLE001 — this stage must never fail extraction
        notes.append(f"convention inference failed ({type(exc).__name__}: {exc}); heuristics only")
        return {}, notes

    overrides: dict[str, LayerRole] = {}
    valid = {r.value: r for r in LayerRole}
    for row in parsed.get("layers", []):
        lay = row.get("layer")
        role = valid.get(str(row.get("role", "")).strip().lower())
        conf = float(row.get("confidence", 0.0) or 0.0)
        if (
            lay in unknown_layers  # only ever overrides what the rules could NOT read
            and role is not None
            and role is not LayerRole.UNKNOWN
            and conf >= cfg.min_confidence
        ):
            overrides[lay] = role
    if overrides:
        notes.append(
            f"layer conventions inferred by {cfg.model} for "
            f"{len(overrides)} unrecognised layer(s): "
            + ", ".join(f"{k}→{v.value}" for k, v in sorted(overrides.items()))
        )
    return overrides, notes
