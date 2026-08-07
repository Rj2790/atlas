"""The IFC exporter round-trips a minimal dwelling.

Synthetic rather than extracted: the demo drawing takes minutes to extract, and
this test exists to catch API drift in ifcopenshell (a 0.7→0.8 rename broke the
`void` module into `feature`) and regressions in the exporter's own geometry,
neither of which needs sixteen real dwellings.
"""

from __future__ import annotations

import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")
import ifcopenshell.geom  # noqa: E402 — after the skip guard

from atlas.build.ifc import export_ifc
from atlas.planir import (
    Level,
    Opening,
    OpeningKind,
    PlanIR,
    PlanUnit,
    Room,
    RoomKind,
    Source,
    Wall,
)


def _mini_plan() -> PlanIR:
    walls = [
        Wall("w1", (0.0, 0.0), (4000.0, 0.0), 230.0, 2900.0, kind="structural", exterior=True),
        Wall("w2", (4000.0, 0.0), (4000.0, 3000.0), 230.0, 2900.0, kind="structural", exterior=True),
        Wall("w3", (4000.0, 3000.0), (0.0, 3000.0), 230.0, 2900.0, kind="structural", exterior=True),
        Wall("w4", (0.0, 3000.0), (0.0, 0.0), 230.0, 2900.0, kind="structural", exterior=True),
    ]
    room = Room(
        id="r1",
        kind=RoomKind.LIVING,
        label="LIVING",
        polygon=[(115.0, 115.0), (3885.0, 115.0), (3885.0, 2885.0), (115.0, 2885.0)],
        centroid=(2000.0, 1500.0),
    )
    door = Opening(
        id="o1",
        wall_id="w1",
        kind=OpeningKind.DOOR,
        t_center=0.5,
        width_mm=900.0,
        height_mm=2100.0,
        sill_mm=0.0,
        swing="left",
        source=Source.CAD_VECTOR,
    )
    window = Opening(
        id="o2",
        wall_id="w3",
        kind=OpeningKind.WINDOW,
        t_center=0.5,
        width_mm=1200.0,
        height_mm=1200.0,
        sill_mm=900.0,
        source=Source.CAD_VECTOR,
    )
    return PlanIR(
        source_file="synthetic",
        source_kind="dxf",
        levels=[Level("L0", "Ground Floor", 0.0, 2900.0, 150.0)],
        walls=walls,
        rooms=[room],
        openings=[door, window],
        units=[
            PlanUnit(
                id="u1",
                name="Test flat",
                room_ids=["r1"],
                wall_ids=[w.id for w in walls],
                bounds=(0.0, 0.0, 4000.0, 3000.0),
            )
        ],
    )


def test_export_round_trips(tmp_path) -> None:
    out = export_ifc(_mini_plan(), tmp_path / "mini.ifc", name="Test flat")
    g = ifcopenshell.open(str(out))

    assert g.schema == "IFC4"
    assert len(g.by_type("IfcWall")) == 4
    assert len(g.by_type("IfcSpace")) == 1
    assert len(g.by_type("IfcDoor")) == 1
    assert len(g.by_type("IfcWindow")) == 1
    assert len(g.by_type("IfcOpeningElement")) == 2

    # The door fills an opening that voids a wall — the relationship chain is the
    # point of IFC; a door floating beside its wall is just a box with a name.
    door = g.by_type("IfcDoor")[0]
    fills = door.FillsVoids
    assert fills, "door is not related to its opening"
    opening = fills[0].RelatingOpeningElement
    assert opening.VoidsElements[0].RelatingBuildingElement.is_a("IfcWall")

    # Provenance travels with every product.
    space = g.by_type("IfcSpace")[0]
    pset_names = {
        rel.RelatingPropertyDefinition.Name
        for rel in space.IsDefinedBy
        if rel.is_a("IfcRelDefinesByProperties")
    }
    assert "Atlas_Provenance" in pset_names

    # Geometry actually builds — a file full of unresolvable representations
    # opens fine and renders nothing.
    settings = ifcopenshell.geom.settings()
    for product in g.by_type("IfcWall") + g.by_type("IfcSpace") + g.by_type("IfcDoor"):
        ifcopenshell.geom.create_shape(settings, product)
