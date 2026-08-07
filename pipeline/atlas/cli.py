"""Atlas command line: floor plan in, 3D out.

    python -m atlas build data/samples/demo-cad-1.dwg --out out/ --unit 1
    python -m atlas build data/samples/3d-2.png --out out/
    python -m atlas render out/scene.json out/model.glb --out out/renders
    python -m atlas export-data --out apps/web/public/data

The schemes and the furniture catalogue are defined once, in Python, and exported
as JSON for the web app. Mirroring them by hand in TypeScript would guarantee they
drift, and a configurator offering a material the renderer does not have is worse
than one offering fewer.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .build.assemble import AssembleConfig, assemble_scene
from .build.gltf import export_glb
from .build.models import POLYHAVEN_MODELS, ModelCache
from .build.schemes import SCHEMES, get_scheme
from .build.textures import TextureCache
from .furnish.catalog import CATALOG
from .planir import PlanIR
from .render.plan_svg import SvgStyle, render_plan


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _ingest(src: Path, work: Path, unit: int | None, max_units: int) -> PlanIR:
    suffix = src.suffix.lower()
    if suffix in (".dwg", ".dxf"):
        from .ingest.dxf_plan import extract_plan

        _log(f"[atlas] reading CAD {src.name}")
        plan = extract_plan(src, work_dir=work, max_units=max_units)
        if unit is not None and 1 <= unit <= len(plan.units):
            chosen = plan.units[unit - 1]
            _log(f"[atlas] selecting dwelling {unit}: {chosen.name}")
            plan = plan.subset(chosen)
        return plan

    if suffix in (".png", ".jpg", ".jpeg", ".webp"):
        from .ingest.raster_vlm import VlmConfig, extract_plan_from_image

        _log(f"[atlas] reading raster {src.name} with a vision model")
        return extract_plan_from_image(src, cfg=VlmConfig())

    raise SystemExit(
        f"unsupported input {src.suffix!r}. Supported: .dwg, .dxf (measured geometry), "
        ".png, .jpg (vision-read geometry)."
    )


def cmd_build(args: argparse.Namespace) -> int:
    src = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(exist_ok=True)

    plan = _ingest(src, work, args.unit, args.max_units)

    _log("")
    _log("[atlas] ingest notes:")
    for n in plan.ingest_notes:
        _log(f"        - {n}")
    if plan.warnings:
        _log("[atlas] warnings:")
        for w in plan.warnings:
            _log(f"        ! {w}")

    q = plan.quality_report()
    _log("")
    _log("[atlas] quality:")
    for k, v in q.items():
        if isinstance(v, float):
            v = round(v, 3)
        _log(f"        {k}: {v}")

    if not plan.rooms:
        _log("[atlas] no rooms recovered; nothing to build")
        (out / "plan.json").write_text(plan.to_json())
        return 1

    (out / "plan.json").write_text(plan.to_json())
    (out / "plan.svg").write_text(render_plan(plan, SvgStyle(width_px=1600)))
    _log(f"[atlas] wrote {out/'plan.json'} and {out/'plan.svg'}")

    scene = assemble_scene(
        plan,
        config=AssembleConfig(
            scheme_id=args.scheme,
            furnish=not args.no_furniture,
            direct_art=not args.no_director,
        ),
    )
    _log("")
    _log("[atlas] scene:")
    for n in scene.notes:
        _log(f"        - {n}")

    (out / "scene.json").write_text(scene.to_json())
    # A layout check drawing. Judging furniture from coordinates is hopeless; in plan a
    # sofa in a doorway is immediately obvious.
    (out / "layout.svg").write_text(
        render_plan(plan, SvgStyle(width_px=1600, show_furniture=True, show_ids=False), scene)
    )
    _log(f"[atlas] wrote {out/'scene.json'} and {out/'layout.svg'}")

    # The director may have chosen a different scheme than the one requested.
    schemes = [scene.scheme_id] if args.single_scheme else list(SCHEMES)

    # Acquire PBR maps for exactly the schemes being exported. Flat base colours
    # are the difference between a massing model and something that reads as built,
    # so this is on by default; --no-textures is for a fast geometry-only check.
    textures: TextureCache | None = None
    if not args.no_textures:
        wanted = sorted(
            {
                m.acg_asset
                for sid in schemes
                for m in get_scheme(sid).materials.values()
                if m.acg_asset
            }
        )
        textures = TextureCache(args.texture_cache)
        _log(f"[atlas] preparing {len(wanted)} CC0 texture set(s) from ambientCG…")
        textures.ensure(wanted)
        for n in textures.notes:
            _log(f"        - {n}")

    # Real CC0 meshes for the detail items. Small download, disproportionate effect:
    # a modelled armchair and a real plant give the eye organic silhouettes to land
    # on, which generated boxes cannot.
    models: ModelCache | None = None
    if not args.no_models:
        models = ModelCache(cache_dir=args.model_cache)
        used = sorted({
            i.external_model for i in CATALOG.values() if i.external_model
        })
        if used:
            _log(f"[atlas] preparing {len(used)} CC0 furniture model(s) from Poly Haven…")
            models.ensure(used)
            for n in models.notes:
                _log(f"        - {n}")

    for sid in schemes:
        res = export_glb(
            scene, out / f"model-{sid}.glb", scheme_id=sid, textures=textures, models=models
        )
        _log(f"[atlas] {res.summary()}")

    # The web app reads the same data the renderer does.
    if args.export_web:
        web = Path(args.export_web)
        _export_data(web)
        (web / "scene.json").write_text(scene.to_json())
        if models is not None:
            import shutil

            dest = web / "models"
            copied = 0
            for slug in {i.external_model for i in CATALOG.values() if i.external_model}:
                asset = models.get(slug)
                if asset is None:
                    continue
                src_dir = asset.gltf_path.parent
                # Copy the whole asset directory: the glTF references its .bin and
                # textures by relative path, so the layout must survive the copy.
                out_dir = dest / slug
                if not out_dir.exists():
                    shutil.copytree(src_dir, out_dir)
                copied += 1
            manifest = {
                slug: f"models/{slug}/{models.get(slug).gltf_path.name}"
                for slug in {i.external_model for i in CATALOG.values() if i.external_model}
                if models.get(slug) is not None
            }
            (web / "models.json").write_text(json.dumps(manifest, indent=1))
            _log(f"[atlas] published {copied} furniture model(s) to {dest}")

        if textures is not None:
            n = textures.publish_web(web / "textures")
            textures.write_manifest(web / "textures.json")
            _log(f"[atlas] published {n} texture file(s) to {web/'textures'}")
            lines = [
                "Textures: ambientCG (https://ambientcg.com), CC0 / public domain.",
                "Models:   Poly Haven (https://polyhaven.com), CC0 / public domain.",
                "No attribution is required; recorded for provenance.",
                "",
                *textures.credits(),
                *(models.credits() if models is not None else []),
            ]
            (out / "CREDITS.txt").write_text("\n".join(lines) + "\n")
        _log(f"[atlas] copied scene + catalogue data to {web}")

    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from .render.blender_render import render

    res = render(
        args.glb,
        args.scene,
        args.out,
        samples=args.samples,
        resolution=(args.width, args.height),
        views=args.views,
    )
    if res["returncode"] != 0:
        _log("[atlas] Blender exited non-zero")
        _log(res["err_tail"])
        return res["returncode"]
    # Surface Blender's own progress lines. They report what the render setup actually
    # did — glass materials rebuilt, bevel applied — and discarding them meant those
    # steps could silently no-op with no way to tell from the outside.
    for line in (res.get("log_tail") or "").splitlines():
        if line.startswith("[atlas]") and "rendered" not in line:
            _log(f"        {line}")
    for p in res["images"]:
        _log(f"[atlas] rendered {p}")
    return 0


def cmd_units(args: argparse.Namespace) -> int:
    """List the dwellings a drawing contains, so one can be chosen to build.

    A production CAD file is a whole floor, not a flat: the sample here holds 16
    apartments across several towers at survey coordinates. Extracting "the plan"
    from it is meaningless — you have to say which dwelling. This lists them with
    the information needed to choose: the unit type the drawing prints, its declared
    carpet area, its measured area and its room mix.
    """
    from .ingest.dxf_plan import extract_plan

    src = Path(args.source)
    if src.suffix.lower() not in (".dwg", ".dxf"):
        raise SystemExit("`units` reads CAD files; a raster image is a single dwelling.")

    work = Path(args.work or (src.parent / "work"))
    work.mkdir(parents=True, exist_ok=True)
    _log(f"[atlas] scanning {src.name} for dwellings…")
    plan = extract_plan(src, work_dir=work, max_units=args.max_units)

    if not plan.units:
        _log("[atlas] no dwellings found")
        return 1

    print(f"{'#':>3}  {'type':<10} {'declared':>10} {'measured':>10} {'rooms':>6}  mix")
    print("-" * 78)
    for i, u in enumerate(plan.units, 1):
        rooms = [r for r in plan.rooms if r.id in set(u.room_ids)]
        measured = sum(r.area_sqft for r in rooms)
        mix: dict[str, int] = {}
        for r in rooms:
            if r.kind.is_habitable or r.kind.is_wet:
                mix[r.kind.value] = mix.get(r.kind.value, 0) + 1
        summary = ", ".join(f"{n}×{k}" for k, n in sorted(mix.items(), key=lambda kv: -kv[1])[:5])
        declared = f"{u.stated_area_sqft:.0f} sf" if u.stated_area_sqft else "-"
        print(
            f"{i:>3}  {u.name[:10]:<10} {declared:>10} {measured:>9.0f} sf {len(rooms):>6}  {summary}"
        )
    print()
    print(f"Build one with:  python -m atlas build {src} --unit <#>")
    return 0


def _export_data(dest: Path) -> None:
    """Write schemes and catalogue JSON for the web app."""
    dest.mkdir(parents=True, exist_ok=True)

    schemes_out = {}
    for sid, scheme in SCHEMES.items():
        schemes_out[sid] = {
            "id": scheme.id,
            "name": scheme.name,
            "description": scheme.description,
            "slots": scheme.slots,
            "accent_colour": list(scheme.accent_colour),
            "ambient_intensity": scheme.ambient_intensity,
            "sun_intensity": scheme.sun_intensity,
            "sun_temperature_k": scheme.sun_temperature_k,
            "interior_temperature_k": scheme.interior_temperature_k,
            "materials": {
                mid: {
                    "id": m.id,
                    "name": m.name,
                    "base_color": list(m.base_color),
                    "roughness": m.roughness,
                    "metallic": m.metallic,
                    "opacity": m.opacity,
                    "sheen": m.sheen,
                    "texture_scale_m": m.texture_scale_m,
                    "acg_asset": m.acg_asset,
                }
                for mid, m in scheme.materials.items()
            },
        }
    (dest / "schemes.json").write_text(json.dumps(schemes_out, indent=1))

    catalog_out = {}
    for cid, item in CATALOG.items():
        catalog_out[cid] = {
            "id": item.id,
            "name": item.name,
            "category": item.category,
            "width_mm": item.width_mm,
            "depth_mm": item.depth_mm,
            "height_mm": item.height_mm,
            "anchor": item.anchor.value,
            "rooms": [r.value for r in item.rooms],
            "priority": item.priority,
            "min_room_area_m2": item.min_room_area_m2,
            "tags": list(item.tags),
            "notes": item.notes,
            "external_model": item.external_model,
            "parts": [
                {
                    "x": p.x, "y": p.y, "z": p.z,
                    "w": p.w, "h": p.h, "d": p.d,
                    "slot": p.slot,
                }
                for p in item.parts
            ],
        }
    (dest / "catalog.json").write_text(json.dumps(catalog_out, indent=1))


def cmd_audit(args: argparse.Namespace) -> int:
    """Audit every dwelling in a drawing against geometric invariants.

    Exits non-zero when any invariant fails, so a threshold change that fixes one flat
    and breaks another shows up immediately rather than never.
    """
    from .audit import audit_file

    _log(f"[atlas] auditing {Path(args.source).name} …")
    res = audit_file(args.source, max_units=args.max_units)
    _log(res.table())
    failed = [u for u in res.units if not u.ok]
    if failed:
        _log(f"\n[atlas] {len(failed)} of {len(res.units)} dwellings failed:")
        _log(res.failure_report())
    else:
        _log(f"\n[atlas] all {len(res.units)} dwellings pass")
    notes = [(u.name, n) for u in res.units for n in u.notes]
    if notes:
        _log("\n[atlas] notes:")
        for n, txt in notes[:12]:
            _log(f"  {n}: {txt}")
    return 0 if res.ok else 1


def cmd_export_data(args: argparse.Namespace) -> int:
    dest = Path(args.out)
    _export_data(dest)
    _log(f"[atlas] wrote {dest/'schemes.json'} ({len(SCHEMES)} schemes)")
    _log(f"[atlas] wrote {dest/'catalog.json'} ({len(CATALOG)} items)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="atlas", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="floor plan -> PlanIR, SceneIR, GLB")
    b.add_argument("source", help="a .dwg/.dxf drawing or a .png/.jpg plan image")
    b.add_argument("--out", default="out", help="output directory")
    b.add_argument("--scheme", default="warm-minimal", choices=list(SCHEMES))
    b.add_argument("--unit", type=int, default=None, help="1-based dwelling to build")
    b.add_argument("--max-units", type=int, default=8, help="dwellings to extract from a CAD set")
    b.add_argument("--no-furniture", action="store_true")
    b.add_argument("--no-director", action="store_true",
                   help="use --scheme as given instead of letting the art director choose")
    b.add_argument("--single-scheme", action="store_true", help="export one GLB, not all schemes")
    b.add_argument("--export-web", default=None, help="also write data into a web public dir")
    b.add_argument("--no-textures", action="store_true", help="skip PBR maps (fast geometry check)")
    b.add_argument("--no-models", action="store_true", help="skip downloaded furniture meshes")
    b.add_argument("--model-cache", default="data/models", help="where to cache Poly Haven models")
    b.add_argument("--texture-cache", default="data/materials", help="where to cache ambientCG downloads")
    b.set_defaults(func=cmd_build)

    r = sub.add_parser("render", help="Cycles hero renders from a GLB + scene.json")
    r.add_argument("scene", help="scene.json")
    r.add_argument("glb", help="model .glb")
    r.add_argument("--out", default="out/renders")
    r.add_argument("--samples", type=int, default=450,
                   help="Cycles samples. Interiors need several hundred; adaptive sampling stops early on the smooth areas so the cost lands on the glass.")
    r.add_argument("--width", type=int, default=1920)
    r.add_argument("--height", type=int, default=1080)
    r.add_argument("--views", type=int, default=4)
    r.set_defaults(func=cmd_render)

    u = sub.add_parser("units", help="list the dwellings a CAD floor plan contains")
    u.add_argument("source", help="a .dwg/.dxf drawing")
    u.add_argument("--max-units", type=int, default=40)
    u.add_argument("--work", default=None, help="scratch dir for DWG conversion")
    u.set_defaults(func=cmd_units)

    a = sub.add_parser(
        "audit",
        help="check every dwelling in a drawing against geometric invariants",
    )
    a.add_argument("source", help="a .dwg/.dxf drawing")
    a.add_argument("--max-units", type=int, default=None)
    a.set_defaults(func=cmd_audit)

    e = sub.add_parser("export-data", help="write schemes.json and catalog.json")
    e.add_argument("--out", default="apps/web/public/data")
    e.set_defaults(func=cmd_export_data)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
