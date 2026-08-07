"""PlanIR → IFC4, so the extraction is consumable by BIM tooling.

The GLB is a picture; the IFC is a statement. Revit, ArchiCAD, Solibri and every
IFC viewer read walls as walls, doors as doors filling openings, and rooms as
IfcSpace with names and areas — which turns this pipeline's output from "a viewer"
into an interchange artefact an architect's office can take downstream.

Scope choices, deliberate:

* Geometry is the simple, honest kind — walls are extruded boxes on their
  centrelines, openings are voiding boxes, doors/windows are filling boxes, and
  spaces are the measured floor polygons extruded to ceiling height. No parametric
  door leafs: the source drawing does not state leaf geometry, and inventing it in
  an interchange format is worse than a plain box with correct position and size.
* Everything carries an ``Atlas_Provenance`` property set: source (drawn vs
  inferred), confidence, and the extraction's measured figures. A consumer can
  filter drawn fact from reasoned inference — the same distinction the audit runs on.
* Coordinates are metres in a local origin at the dwelling's bounds corner, so the
  numbers stay small and the file opens where viewers expect it.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

import ifcopenshell
import ifcopenshell.api.aggregate
import ifcopenshell.api.context
import ifcopenshell.api.feature
import ifcopenshell.api.geometry
import ifcopenshell.api.project
import ifcopenshell.api.pset
import ifcopenshell.api.root
import ifcopenshell.api.spatial
import ifcopenshell.api.unit

from ..planir import OpeningKind, PlanIR

__all__ = ["export_ifc"]

_MM = 0.001


def export_ifc(plan: PlanIR, out_path: Path | str, *, name: str | None = None) -> Path:
    """Write ``plan`` (one dwelling's subset) as an IFC4 file."""
    out_path = Path(out_path)

    f = ifcopenshell.api.project.create_file(version="IFC4")
    project = ifcopenshell.api.root.create_entity(
        f, ifc_class="IfcProject", name=name or (plan.units[0].name if plan.units else "Dwelling")
    )
    ifcopenshell.api.unit.assign_unit(f)
    model = ifcopenshell.api.context.add_context(f, context_type="Model")
    body = ifcopenshell.api.context.add_context(
        f,
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
        parent=model,
    )

    site = ifcopenshell.api.root.create_entity(f, ifc_class="IfcSite", name="Site")
    building = ifcopenshell.api.root.create_entity(f, ifc_class="IfcBuilding", name="Building")
    storey = ifcopenshell.api.root.create_entity(
        f, ifc_class="IfcBuildingStorey", name=plan.levels[0].name if plan.levels else "Ground Floor"
    )
    ifcopenshell.api.aggregate.assign_object(f, products=[site], relating_object=project)
    ifcopenshell.api.aggregate.assign_object(f, products=[building], relating_object=site)
    ifcopenshell.api.aggregate.assign_object(f, products=[storey], relating_object=building)

    # Local origin: dwelling bounds corner, so coordinates stay small.
    if plan.units and plan.units[0].bounds != (0, 0, 0, 0):
        ox, oy = plan.units[0].bounds[0], plan.units[0].bounds[1]
    elif plan.rooms:
        ox = min(x for r in plan.rooms for x, _ in r.polygon)
        oy = min(y for r in plan.rooms for _, y in r.polygon)
    else:
        ox = oy = 0.0

    def _m(pt: tuple[float, float]) -> tuple[float, float]:
        return ((pt[0] - ox) * _MM, (pt[1] - oy) * _MM)

    def _pset(product: Any, props: dict[str, Any]) -> None:
        ps = ifcopenshell.api.pset.add_pset(f, product=product, name="Atlas_Provenance")
        ifcopenshell.api.pset.edit_pset(
            f, pset=ps, properties={k: v for k, v in props.items() if v is not None}
        )

    # --- walls -----------------------------------------------------------------
    wall_products: dict[str, Any] = {}
    wall_geo: dict[str, tuple[tuple[float, float], float, float, float]] = {}
    for w in plan.walls:
        # Plain floats at the API boundary: extraction values arrive as numpy
        # scalars through pickling, and the IFC wrapper's aggregate setters
        # reject anything that is not exactly float.
        sx, sy = (float(v) for v in _m(w.start))
        ex, ey = (float(v) for v in _m(w.end))
        length = float(math.hypot(ex - sx, ey - sy))
        if length < 1e-6:
            continue
        ang = math.atan2(ey - sy, ex - sx)
        height = float(w.height_mm) * _MM
        thick = float(max(w.thickness_mm, 40.0)) * _MM
        wall = ifcopenshell.api.root.create_entity(f, ifc_class="IfcWall", name=w.id)
        rep = ifcopenshell.api.geometry.add_wall_representation(
            f, context=body, length=length, height=height, thickness=thick, offset=-thick / 2.0
        )
        ifcopenshell.api.geometry.assign_representation(f, product=wall, representation=rep)
        matrix = np.eye(4)
        matrix[0][0], matrix[0][1] = math.cos(ang), -math.sin(ang)
        matrix[1][0], matrix[1][1] = math.sin(ang), math.cos(ang)
        matrix[0][3], matrix[1][3], matrix[2][3] = sx, sy, 0.0
        ifcopenshell.api.geometry.edit_object_placement(f, product=wall, matrix=matrix)
        ifcopenshell.api.spatial.assign_container(
            f, products=[wall], relating_structure=storey
        )
        _pset(
            wall,
            {
                "AtlasId": w.id,
                "Source": str(w.source),
                "Confidence": float(w.confidence),
                "Exterior": bool(w.exterior),
                "Kind": str(w.kind),
            },
        )
        wall_products[w.id] = wall
        wall_geo[w.id] = ((sx, sy), ang, length, height)

    # --- openings, and the doors/windows that fill them -------------------------
    for o in plan.openings:
        host = wall_products.get(o.wall_id)
        geo = wall_geo.get(o.wall_id)
        w = plan.wall_by_id(o.wall_id)
        if host is None or geo is None or w is None:
            continue
        (sx, sy), ang, length, _ = geo
        width = float(min(float(o.width_mm) * _MM, length * 0.97))
        o_height = float(o.height_mm) * _MM
        sill = float(o.sill_mm) * _MM
        thick = float(max(w.thickness_mm, 40.0)) * _MM
        along = float(max(0.0, min(float(o.t_center) * length - width / 2.0, length - width)))

        def _place(product: Any, dz: float, depth_pad: float) -> None:
            matrix = np.eye(4)
            matrix[0][0], matrix[0][1] = math.cos(ang), -math.sin(ang)
            matrix[1][0], matrix[1][1] = math.sin(ang), math.cos(ang)
            matrix[0][3] = sx + math.cos(ang) * along
            matrix[1][3] = sy + math.sin(ang) * along
            matrix[2][3] = dz
            ifcopenshell.api.geometry.edit_object_placement(f, product=product, matrix=matrix)
            del depth_pad

        opening = ifcopenshell.api.root.create_entity(
            f, ifc_class="IfcOpeningElement", name=f"{o.id}-void"
        )
        rep = ifcopenshell.api.geometry.add_wall_representation(
            f,
            context=body,
            length=width,
            height=o_height,
            thickness=thick + 0.05,
            offset=-(thick + 0.05) / 2.0,
        )
        ifcopenshell.api.geometry.assign_representation(f, product=opening, representation=rep)
        _place(opening, sill, 0.05)
        ifcopenshell.api.feature.add_feature(f, feature=opening, element=host)

        is_door = o.kind.is_door or o.kind is OpeningKind.OPENING
        if o.kind is OpeningKind.OPENING:
            continue  # a cased opening is a void with no filling
        fill_class = "IfcDoor" if is_door else "IfcWindow"
        fill = ifcopenshell.api.root.create_entity(f, ifc_class=fill_class, name=o.id)
        rep = ifcopenshell.api.geometry.add_wall_representation(
            f,
            context=body,
            length=width,
            height=o_height,
            thickness=0.05,
            offset=-0.025,
        )
        ifcopenshell.api.geometry.assign_representation(f, product=fill, representation=rep)
        _place(fill, sill, 0.0)
        ifcopenshell.api.feature.add_filling(f, opening=opening, element=fill)
        ifcopenshell.api.spatial.assign_container(
            f, products=[fill], relating_structure=storey
        )
        _pset(
            fill,
            {
                "AtlasId": o.id,
                "Source": str(o.source),
                "Confidence": float(o.confidence),
                "OpeningKind": str(o.kind),
                "WidthMm": float(o.width_mm),
            },
        )

    # --- spaces ------------------------------------------------------------------
    for r in plan.rooms:
        if len(r.polygon) < 3:
            continue
        pts = [_m(p) for p in r.polygon]
        # Local profile about the room's own corner, placement carries it back.
        rx = min(x for x, _ in pts)
        ry = min(y for _, y in pts)
        local = [(x - rx, y - ry) for x, y in pts]
        if local[0] != local[-1]:
            local.append(local[0])
        curve = f.create_entity(
            "IfcIndexedPolyCurve",
            Points=f.create_entity(
                "IfcCartesianPointList2D", CoordList=[(float(x), float(y)) for x, y in local]
            ),
        )
        profile = f.create_entity(
            "IfcArbitraryClosedProfileDef", ProfileType="AREA", OuterCurve=curve
        )
        space = ifcopenshell.api.root.create_entity(f, ifc_class="IfcSpace", name=r.label or r.kind.value)
        space.LongName = r.kind.value
        rep = ifcopenshell.api.geometry.add_profile_representation(
            f,
            context=body,
            profile=profile,
            depth=float(r.ceiling_height_mm) * _MM,
            cardinal_point=None,
        )
        ifcopenshell.api.geometry.assign_representation(f, product=space, representation=rep)
        matrix = np.eye(4)
        matrix[0][3], matrix[1][3] = float(rx), float(ry)
        matrix[2][3] = float(r.floor_level_mm) * _MM
        ifcopenshell.api.geometry.edit_object_placement(f, product=space, matrix=matrix)
        ifcopenshell.api.aggregate.assign_object(f, products=[space], relating_object=storey)
        _pset(
            space,
            {
                "AtlasId": r.id,
                "Source": str(r.source),
                "Confidence": float(r.confidence),
                "MeasuredAreaSqft": float(r.area_sqft),
                "StatedDimension": (
                    f"{r.stated_dimension.width_mm:.0f}x{r.stated_dimension.depth_mm:.0f}"
                    if r.stated_dimension
                    else None
                ),
            },
        )

    f.write(str(out_path))
    return out_path
