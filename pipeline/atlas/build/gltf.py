"""Export a SceneIR to glTF/GLB.

glTF is the right target: it is the only 3D format that is simultaneously a web
standard, natively understood by three.js, importable into Blender for offline
rendering, and physically-based by definition. One file therefore serves the
browser configurator, the hero render and any downstream consumer, with no
material translation step where a look can drift between them.

Two choices worth stating:

**Merge surfaces by material slot.** SceneIR keeps one surface per wall and per
room, which is what the configurator wants for selection and per-room swapping. A
renderer wants the opposite: each distinct mesh/material pair is a draw call, and
seventy of them for a flat is wasteful. Merging by slot collapses that to about a
dozen while preserving exactly the granularity the schemes address, and the node
names retain the original ids so selection still works.

**Bake furniture and joinery, don't instance them.** glTF supports reusing a mesh
across nodes, which would be smaller. But a swap in the configurator replaces one
item's geometry, and shared meshes make that a graph surgery problem instead of a
node replacement. At the sizes involved — a furnished flat is well under a
megabyte of boxes — the simplicity is worth more than the bytes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..furnish.catalog import CATALOG
from ..sceneir import DesignScheme, Material, MaterialSlot, OpeningSpec, SceneIR
from .models import ModelCache
from .textures import TextureCache
from .schemes import get_scheme

__all__ = ["ExportResult", "export_glb", "scene_to_trimesh"]


@dataclass
class ExportResult:
    path: Path
    size_bytes: int
    mesh_count: int
    triangle_count: int
    materials: int
    notes: list[str]

    def summary(self) -> str:
        return (
            f"{self.path.name}: {self.size_bytes/1e6:.2f} MB, {self.mesh_count} meshes, "
            f"{self.triangle_count} triangles, {self.materials} materials"
        )


def _pbr(mat: Material, textures: Any | None = None) -> Any:
    """Build a trimesh PBRMaterial from our material description.

    When ``textures`` supplies maps for this material they are attached, packed the
    way glTF specifies:

    * ``baseColorTexture`` — the colour map, *multiplied* by ``baseColorFactor``, so
      one oak asset serves warm oak and smoked oak by tint alone.
    * ``metallicRoughnessTexture`` — a single RGB image with **G = roughness** and
      **B = metallic**. glTF mandates this packing; supplying a plain greyscale
      roughness map instead puts roughness in R where nothing reads it, and every
      surface renders mirror-smooth.
    * ``occlusionTexture`` — R channel, per spec, so it can share the packed image.
    """
    from trimesh.visual.material import PBRMaterial

    r, g, b = mat.base_color
    kwargs: dict[str, Any] = dict(
        name=mat.name,
        baseColorFactor=[r, g, b, mat.opacity],
        metallicFactor=mat.metallic,
        roughnessFactor=mat.roughness,
        emissiveFactor=list(mat.emissive),
        alphaMode="BLEND" if mat.opacity < 0.999 else "OPAQUE",
        doubleSided=False,
    )

    ts = textures.get(mat.acg_asset) if textures else None
    if ts is not None:
        try:
            from PIL import Image

            colour_role = (
                "base_color_neutral"
                if mat.texture_neutralize and "base_color_neutral" in ts.embed
                else "base_color"
            )
            if colour_role in ts.embed:
                kwargs["baseColorTexture"] = Image.open(ts.embed[colour_role]).convert("RGB")
            if "normal" in ts.embed:
                kwargs["normalTexture"] = Image.open(ts.embed["normal"]).convert("RGB")

            rough = Image.open(ts.embed["roughness"]).convert("L") if "roughness" in ts.embed else None
            ao = Image.open(ts.embed["ao"]).convert("L") if "ao" in ts.embed else None
            if rough is not None:
                size = rough.size
                red = (ao.resize(size) if ao is not None else Image.new("L", size, 255))
                blue = Image.new("L", size, int(round(mat.metallic * 255)))
                packed = Image.merge("RGB", (red, rough, blue))
                kwargs["metallicRoughnessTexture"] = packed
                # Scalars multiply the map, so they must be neutral or the map's
                # variation is scaled away to nothing.
                kwargs["roughnessFactor"] = 1.0
                kwargs["metallicFactor"] = 1.0 if mat.metallic > 0.5 else 1.0
                if ao is not None:
                    kwargs["occlusionTexture"] = packed
        except Exception:
            # A texture problem must degrade to a scalar material, never fail a build.
            pass

    return PBRMaterial(**kwargs)


def _fallback_material(slot: str) -> Material:
    """A visibly wrong magenta, if a slot is somehow unfilled.

    Deliberately garish. A missing material that renders as plausible grey ships
    unnoticed; one that renders as magenta gets fixed.
    """
    return Material(id=f"missing-{slot}", name=f"MISSING {slot}", base_color=(1.0, 0.0, 0.8), roughness=0.5)


def _boxes_to_arrays(
    boxes: Sequence[tuple[tuple[float, float, float], tuple[float, float, float], float]],
    uv_scale: float = 1.0,
) -> tuple[list[list[float]], list[list[int]], list[list[float]]]:
    """Convert (centre, size, yaw) boxes into vertex/face/UV arrays.

    UVs are generated per box face from that face's real size in metres, then
    divided by ``uv_scale`` (the material's ``texture_scale_m``). Without this the
    furniture and door leaves carried no UVs at all, so every texture collapsed to
    a single pixel and the joinery rendered as flat colour while the walls around it
    were correctly grained.
    """
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    uvs: list[list[float]] = []
    # Unit cube corners and its 12 triangles, wound outward.
    corners = [
        (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5),
    ]
    tris = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
        (1, 2, 6), (1, 6, 5), (0, 4, 7), (0, 7, 3),
    ]
    # Which of the box's dimensions each corner's (u, v) should follow. Using the
    # two largest dimensions keeps the mapping sensible on thin panels.
    for (cx, cy, cz), (sx, sy, sz), yaw in boxes:
        base = len(verts)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        for lx, ly, lz in corners:
            px, py, pz = lx * sx, ly * sy, lz * sz
            verts.append([cx + px * cos_y + pz * sin_y, cy + py, cz - px * sin_y + pz * cos_y])
            # Project onto the box's two dominant axes so the grain runs along the
            # long dimension of a plank or panel.
            if sy <= sx and sy <= sz:          # flat-ish: map in x/z
                u, v = (lx + 0.5) * sx, (lz + 0.5) * sz
            elif sx <= sy and sx <= sz:        # thin in x: map in z/y
                u, v = (lz + 0.5) * sz, (ly + 0.5) * sy
            else:                              # thin in z: map in x/y
                u, v = (lx + 0.5) * sx, (ly + 0.5) * sy
            uvs.append([u / uv_scale, v / uv_scale])
        for a, b, c in tris:
            faces.append([base + a, base + b, base + c])
    return verts, faces, uvs


def _fit_external(mesh_scene: Any, item: Any, placement: Any) -> Any:
    """Scale and place a downloaded model to match a catalogue item's declared size.

    Fitting rather than stretching. The scale is **uniform**, so the model keeps its
    designer's proportions; a non-uniform fit to an arbitrary declared box turns a
    coffee table into a squashed caricature and is instantly readable as wrong.

    The model's longer horizontal axis is aligned with the item's longer declared
    axis before fitting, because glTF authors do not agree on which way an object
    faces — Poly Haven's coffee table is 1.2 m along Z and 0.6 m along X, so a naive
    fit would place it across the room instead of along the wall.

    Finally the model is dropped so its lowest point sits on the floor. Models are
    authored with origins anywhere, and one authored around its centre would
    otherwise be buried to the waist.
    """
    import numpy as np
    import trimesh

    combined = mesh_scene.copy()
    bounds = combined.bounds
    if bounds is None:
        return None
    size = bounds[1] - bounds[0]
    ex, ey, ez = (float(v) for v in size)
    if min(ex, ey, ez) <= 1e-6:
        return None

    want_w = item.width_mm / 1000.0
    want_d = item.depth_mm / 1000.0
    want_h = item.height_mm / 1000.0

    # Align the longer horizontal axis of the model with the longer declared axis.
    swap = (ex >= ez) != (want_w >= want_d)
    model_w, model_d = (ez, ex) if swap else (ex, ez)

    scale = min(want_w / model_w, want_d / model_d, want_h / ey)
    if not (scale > 0):
        return None

    T = np.eye(4)
    T[:3, :3] *= scale
    combined.apply_transform(T)

    if swap:
        combined.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))

    b = combined.bounds
    centre = (b[0] + b[1]) / 2.0
    # Centre in plan, sit on the floor.
    combined.apply_translation([-centre[0], -b[0][1], -centre[2]])

    combined.apply_transform(
        trimesh.transformations.rotation_matrix(placement.rotation_y, [0, 1, 0])
    )
    combined.apply_translation(list(placement.position))
    return combined


def _furniture_boxes(
    scene: SceneIR, skip_ids: frozenset[str] = frozenset()
) -> dict[str, list[tuple[tuple[float, float, float], tuple[float, float, float], float]]]:
    """Expand furniture placements into world-space boxes, grouped by slot."""
    out: dict[str, list[Any]] = {}
    for pl in scene.furniture:
        if pl.id in skip_ids:
            continue
        item = CATALOG.get(pl.catalog_id)
        if item is None:
            continue
        px, py, pz = pl.position
        yaw = pl.rotation_y
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        # Plan dimensions honour the placement's scale; height does not.
        #
        # The solver may place an item smaller than its catalogue size when the room is
        # tight, and it reserves floor for the size it chose. Building the parts at
        # catalogue size regardless would draw a full-size desk in the space reserved for a
        # 70% one, so the mesh would overlap its neighbours while the collision model
        # believed everything fitted. Height is left alone for the same reason the solver
        # leaves it alone: a work surface is set by the human body, not the spare floor.
        sx, _sy, sz = pl.scale
        for part in item.parts:
            slot = pl.material_overrides.get(part.slot, part.slot)
            # Part local mm -> metres, then rotate into world.
            lx = part.x / 1000.0 * sx
            ly = part.y / 1000.0
            lz = part.z / 1000.0 * sz
            wx = px + lx * cos_y + lz * sin_y
            wz = pz - lx * sin_y + lz * cos_y
            out.setdefault(slot, []).append(
                (
                    (wx, py + ly, wz),
                    (part.w / 1000.0 * sx, part.h / 1000.0, part.d / 1000.0 * sz),
                    yaw,
                )
            )
    return out


def _opening_boxes(
    openings: Sequence[OpeningSpec],
) -> dict[str, list[tuple[tuple[float, float, float], tuple[float, float, float], float]]]:
    """Frames, leaves and glazing for each opening.

    A frame is four thin boxes around the reveal rather than one slab, so the
    opening reads as a built assembly. Glazing sits inside the frame at reduced
    height so the frame is visible around it — glass flush to the wall face is the
    single most common thing that makes a window look painted on.
    """
    out: dict[str, list[Any]] = {}
    for op in openings:
        w, h, t = op.width, op.height, op.thickness
        cx, cy, cz = op.centre
        yaw = op.rotation_y
        frame = 0.06
        depth = max(0.05, t * 0.9)

        def add(slot: str, lx: float, ly: float, sx: float, sy: float, sz: float) -> None:
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            out.setdefault(slot, []).append(
                ((cx + lx * cos_y, cy + ly, cz - lx * sin_y), (sx, sy, sz), yaw)
            )

        # Frame: two jambs, head, and (for windows) a sill.
        add(op.frame_slot, -(w / 2 - frame / 2), 0.0, frame, h, depth)
        add(op.frame_slot, (w / 2 - frame / 2), 0.0, frame, h, depth)
        add(op.frame_slot, 0.0, h / 2 - frame / 2, w, frame, depth)
        add(op.frame_slot, 0.0, -(h / 2 - frame / 2), w, frame, depth)

        if op.glazed:
            add(MaterialSlot.GLAZING.value, 0.0, 0.0, w - 2 * frame, h - 2 * frame, 0.012)
            # Central mullion on anything wide enough to need one.
            if w > 1.4:
                add(op.frame_slot, 0.0, 0.0, 0.05, h - 2 * frame, depth * 0.8)
        elif op.kind == "sliding_door" or op.swing is None:
            # A sliding door does not swing — it runs in the wall plane. Sending it down
            # the hinged branch stood a 2 m leaf 0.8 m out into the room at the balcony
            # threshold, which is both wrong and the largest single intrusion in the model.
            # One panel parked over half the opening leaves the other half clear, which is
            # what an open patio door looks like.
            panel = (w - 2 * frame) / 2.0
            add(op.leaf_slot, -(w / 2 - frame) + panel / 2.0, 0.0, panel, h - frame, 0.042)
        else:
            # Door leaf, swung ajar about its hinge jamb.
            #
            # A leaf modelled flat in the wall plane completely fills the opening, so
            # the hole cut in the wall is invisible and you cannot see from one room
            # into the next — the doorway reads as a panelled recess. Swinging it clear
            # is what makes the opening legible as an opening.
            #
            # But 78 degrees is nearly perpendicular to the wall, which stands the leaf
            # out into the middle of the room. With a door on every room that turned the
            # overview into a field of free-standing brown panels — from above they read
            # as screens, not doors, and they hid the rooms they were meant to reveal.
            # 24 degrees keeps the leaf against its own wall, still plainly ajar, and
            # leaves the opening clear to see through. It is also how a presentation plan
            # draws a door: enough to read the swing, not enough to block the room.
            leaf_w = w - 2 * frame
            ang = math.radians(24.0) * (1.0 if op.swing != "right" else -1.0)
            hinge_x = -(leaf_w / 2.0)
            # Rotate the leaf's far edge inward about the hinge.
            lx = hinge_x + (leaf_w / 2.0) * math.cos(ang)
            lz = (leaf_w / 2.0) * math.sin(ang) * (1.0 if op.swing_inward else -1.0)
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            out.setdefault(MaterialSlot.DOOR_LEAF.value, []).append(
                (
                    (cx + lx * cos_y + lz * sin_y, cy, cz - lx * sin_y + lz * cos_y),
                    (leaf_w, h - frame, 0.042),
                    yaw + ang * (1.0 if op.swing_inward else -1.0),
                )
            )
    return out


def scene_to_trimesh(
    scene: SceneIR,
    scheme: DesignScheme | None = None,
    textures: TextureCache | None = None,
    models: ModelCache | None = None,
) -> Any:
    """Build a ``trimesh.Scene`` from a SceneIR, with materials and maps resolved."""
    import numpy as np
    import trimesh

    scheme = scheme or get_scheme(scene.scheme_id)
    ts = trimesh.Scene()

    def material_for(slot: str, wall_id: str | None = None) -> Material:
        """Resolve a slot through the scheme and the art director's overrides."""
        if wall_id:
            keyed = scene.slot_overrides.get(f"{slot}@{wall_id}")
            if keyed and keyed in scheme.materials:
                return scheme.materials[keyed]
        mid = scene.slot_overrides.get(slot)
        if mid and mid in scheme.materials:
            return scheme.materials[mid]
        return scheme.material_for(slot) or _fallback_material(slot)

    # --- architectural surfaces, merged by slot ---
    by_slot: dict[str, tuple[list[float], list[int], list[float], list[float]]] = {}
    for s in scene.surfaces:
        v, i, n, u = by_slot.setdefault(s.slot, ([], [], [], []))
        offset = len(v) // 3
        nvert = len(s.vertices) // 3
        v.extend(s.vertices)
        n.extend(s.normals if s.normals else [0.0, 1.0, 0.0] * nvert)
        u.extend(s.uvs if len(s.uvs) == nvert * 2 else [0.0, 0.0] * nvert)
        i.extend(idx + offset for idx in s.indices)

    added = 0
    tri_total = 0
    for slot, (v, i, n, u) in by_slot.items():
        if not i:
            continue
        mat = material_for(slot)
        # World-metre UVs divided by the material's real-world tile size. This is
        # what keeps a 600 mm tile 600 mm and a 1.9 m oak plank 1.9 m, instead of
        # every surface getting one arbitrary repeat.
        k = max(mat.texture_scale_m, 1e-6)
        uv = np.asarray(u, dtype=np.float64).reshape(-1, 2) / k
        mesh = trimesh.Trimesh(
            vertices=np.asarray(v, dtype=np.float64).reshape(-1, 3),
            faces=np.asarray(i, dtype=np.int64).reshape(-1, 3),
            vertex_normals=np.asarray(n, dtype=np.float64).reshape(-1, 3),
            process=False,
        )
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=_pbr(mat, textures))
        ts.add_geometry(mesh, geom_name=f"arch::{slot}", node_name=f"arch::{slot}")
        added += 1
        tri_total += len(mesh.faces)

    # --- real downloaded models, each keeping its own authored materials ---
    external_ids: set[str] = set()
    if models is not None:
        for pl in scene.furniture:
            item = CATALOG.get(pl.catalog_id)
            asset = models.get(item.external_model) if item else None
            if asset is None:
                continue
            try:
                loaded = trimesh.load(str(asset.gltf_path), force="scene")
                fitted = _fit_external(loaded, item, pl)
                if fitted is None:
                    continue
                name = f"model::{pl.id}"
                ts.add_geometry(fitted, geom_name=name, node_name=name)
                external_ids.add(pl.id)
                added += 1
                tri_total += sum(
                    len(g.faces) for g in getattr(fitted, "geometry", {}).values()
                ) or 0
            except Exception:
                # Fall through to the parametric version rather than lose the item.
                continue

    # --- openings and remaining furniture, merged by slot ---
    for prefix, groups in (
        ("opening", _opening_boxes(scene.openings)),
        ("furniture", _furniture_boxes(scene, frozenset(external_ids))),
    ):
        for slot, boxes in groups.items():
            mat = material_for(slot)
            verts, faces, uvs = _boxes_to_arrays(boxes, max(mat.texture_scale_m, 1e-6))
            if not faces:
                continue
            mesh = trimesh.Trimesh(
                vertices=np.asarray(verts, dtype=np.float64),
                faces=np.asarray(faces, dtype=np.int64),
                process=False,
            )
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=np.asarray(uvs, dtype=np.float64), material=_pbr(mat, textures)
            )
            name = f"{prefix}::{slot}"
            ts.add_geometry(mesh, geom_name=name, node_name=name)
            added += 1
            tri_total += len(mesh.faces)

    ts.metadata["atlas"] = {
        "external_models": len(external_ids),
        "scheme": scheme.id,
        "rooms": len(scene.rooms),
        "furniture": len(scene.furniture),
        "triangles": tri_total,
        "meshes": added,
    }
    return ts


def export_glb(
    scene: SceneIR,
    out_path: Path | str,
    *,
    scheme_id: str | None = None,
    textures: TextureCache | None = None,
    models: ModelCache | None = None,
) -> ExportResult:
    """Write ``scene`` to a .glb file, with PBR maps embedded when available."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scheme = get_scheme(scheme_id or scene.scheme_id)
    ts = scene_to_trimesh(scene, scheme, textures, models)

    data = ts.export(file_type="glb")
    out_path.write_bytes(data)

    meta = ts.metadata.get("atlas", {})
    notes = [
        f"scheme '{scheme.name}' applied to {len(ts.geometry)} merged meshes",
        f"slots present: {', '.join(sorted({n.split('::')[-1] for n in ts.geometry}))}",
    ]
    missing = [n for n in ts.geometry if "MISSING" in str(n)]
    if missing:
        notes.append(f"WARNING: {len(missing)} surface group(s) had no material in the scheme")
    return ExportResult(
        path=out_path,
        size_bytes=len(data),
        mesh_count=len(ts.geometry),
        triangle_count=int(meta.get("triangles", 0)),
        materials=len(ts.geometry),
        notes=notes,
    )
