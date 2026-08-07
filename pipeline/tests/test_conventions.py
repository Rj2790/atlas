"""Convention inference engages only when heuristics fail, and validates hard.

The stage's contract has three legs — dormancy on known conventions, strict
validation of what the model returns, and graceful degradation without a key —
and each leg failing would be invisible in production: a dormant-when-it-
shouldn't-be stage just extracts badly, an over-trusting one lets a language
model reclassify working layers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from atlas.ingest.conventions import ConventionConfig, infer_layer_conventions
from atlas.ingest.layers import LayerRole


@dataclass
class _Ent:
    layer: str
    dxftype: str = "LINE"
    points: list = field(default_factory=lambda: [(0.0, 0.0), (1000.0, 0.0)])
    text: str | None = None


def _answer(mapping: dict[str, tuple[str, float]]) -> str:
    return json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "layers": [
                                    {"layer": k, "role": r, "confidence": c, "why": "test"}
                                    for k, (r, c) in mapping.items()
                                ]
                            }
                        )
                    }
                }
            ]
        }
    )


def test_dormant_when_conventions_are_known() -> None:
    """A drawing on recognised layers never engages — no call, no overrides."""
    ents = [_Ent("A-WALL") for _ in range(50)] + [_Ent("A-DOOR") for _ in range(10)]
    calls: list[str] = []

    def transport(url: str, payload: dict) -> str:
        calls.append(url)
        return _answer({})

    overrides, notes = infer_layer_conventions(ents, transport=transport)
    assert overrides == {}
    assert calls == [], "made a network call for a drawing the rules already read"


def test_maps_unknown_layers_and_only_unknown_layers() -> None:
    ents = [_Ent("MUR_PORTEUR") for _ in range(40)] + [_Ent("A-WALL") for _ in range(10)]

    def transport(url: str, payload: dict) -> str:
        # The model tries to reclassify the *known* layer too — must be ignored.
        return _answer({"MUR_PORTEUR": ("wall", 0.95), "A-WALL": ("ignore", 0.99)})

    overrides, notes = infer_layer_conventions(ents, transport=transport)
    assert overrides == {"MUR_PORTEUR": LayerRole.WALL}
    assert any("MUR_PORTEUR" in n for n in notes)


def test_low_confidence_and_invalid_roles_are_discarded() -> None:
    ents = [_Ent("x1") for _ in range(30)] + [_Ent("x2") for _ in range(30)]

    def transport(url: str, payload: dict) -> str:
        return _answer({"x1": ("wall", 0.3), "x2": ("marzipan", 0.99)})

    overrides, _ = infer_layer_conventions(ents, transport=transport)
    assert overrides == {}


def test_degrades_on_garbage_response() -> None:
    ents = [_Ent("x1") for _ in range(30)]

    def transport(url: str, payload: dict) -> str:
        return "<html>502 Bad Gateway</html>"

    overrides, notes = infer_layer_conventions(ents, transport=transport)
    assert overrides == {}
    assert any("failed" in n for n in notes)


def test_no_key_no_transport_degrades_with_a_note(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    ents = [_Ent("x1") for _ in range(30)]
    overrides, notes = infer_layer_conventions(ents)
    assert overrides == {}
    assert any("OPENROUTER_API_KEY" in n for n in notes)
