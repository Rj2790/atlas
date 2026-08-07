"""Offline hero renders via headless Blender Cycles.

The browser configurator is for exploring; this is for the image that goes in the
brochure. Cycles is a physically-correct path tracer, so it gives the things a
real-time renderer approximates or omits: multi-bounce indirect light (which is
most of the light in an interior), true soft shadows from area sources, and correct
glass.

Runs as a Blender-internal script — Blender is driven as ``blender --background
--python this_file -- args``, because ``bpy`` as a pip module is a large, fragile
dependency while a Blender install is a single download that ships its own Python.

The camera work matters as much as the renderer. Three deliberate choices:

* **A 24 mm lens, not a fisheye.** Interior renders are often shot at 12-16 mm to
  get the whole room in, which visibly bends verticals and inflates the space. 24 mm
  is what an architectural photographer actually uses.
* **Eye height, level.** The camera sits at 1.55 m and is kept horizontal, so
  verticals stay vertical. A tilted camera converging the walls is the classic
  amateur interior photograph.
* **Framed from the room's long diagonal.** Standing in the corner of the longest
  axis shows depth; standing square to a wall flattens the room.
"""

from __future__ import annotations

import json
import math
import os
import sys

# ---------------------------------------------------------------------------
# host side: launch Blender
# ---------------------------------------------------------------------------

def render(
    glb_path: str,
    scene_json: str,
    out_dir: str,
    *,
    blender: str = "/Applications/Blender.app/Contents/MacOS/Blender",
    samples: int = 450,
    resolution: tuple[int, int] = (1920, 1080),
    views: int = 3,
    timeout: int = 3600,
) -> dict:
    """Render hero stills. Returns a dict with the written files and any log tail."""
    import subprocess
    from pathlib import Path

    if not Path(blender).exists():
        raise RuntimeError(
            f"Blender not found at {blender}. Install Blender or pass blender=<path>."
        )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Expand each room's ceiling polygon outward so it overlaps the wall bodies.
    # Floor outlines stop at (or a raster-cell short of) the wall face, and a
    # ceiling cut to the same line leaves a horizontal skylight gap the sun pours
    # through — every wall-head in the first ceilinged render wore a bright slit.
    # Shapely lives host-side, so the buffering happens here and Blender just
    # receives polygons it can trust.
    import json as _json
    import tempfile as _tempfile

    try:
        from shapely.geometry import Polygon as _Poly

        data = _json.loads(Path(scene_json).read_text())
        for room in data.get("rooms") or []:
            poly = room.get("polygon") or []
            if len(poly) < 3:
                continue
            grown = _Poly(poly).buffer(0.2, join_style=2)
            if grown.is_empty or not hasattr(grown, "exterior"):
                continue
            room["render_ceiling_polygon"] = [
                [float(x), float(z)] for x, z in grown.exterior.coords[:-1]
            ]
        tmp = _tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, dir=out_dir
        )
        _json.dump(data, tmp)
        tmp.close()
        scene_json = tmp.name
    except Exception:  # noqa: BLE001 — renders must not die on ceiling prep
        pass

    proc = subprocess.run(
        [
            blender, "--background", "--factory-startup",
            "--python", os.path.abspath(__file__), "--",
            glb_path, scene_json, out_dir,
            str(samples), str(resolution[0]), str(resolution[1]), str(views),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    written = sorted(str(p) for p in Path(out_dir).glob("*.png"))
    return {
        "returncode": proc.returncode,
        "images": written,
        "log_tail": (proc.stdout or "")[-3000:],
        "err_tail": (proc.stderr or "")[-2000:],
    }


# ---------------------------------------------------------------------------
# Blender side
# ---------------------------------------------------------------------------

def _kelvin_to_rgb(k: float) -> tuple[float, float, float]:
    """Approximate a blackbody colour, normalised to peak 1.0.

    Warm interior fill against cooler daylight is what makes an interior read as
    lit rather than flooded; that contrast needs real colour temperatures.
    """
    t = k / 100.0
    if t <= 66:
        r = 1.0
        g = 0.39008157876901960784 * math.log(max(t, 1e-3)) - 0.63184144378862745098
    else:
        r = 1.29293618606274509804 * ((t - 60) ** -0.1332047592)
        g = 1.12989086089529411765 * ((t - 60) ** -0.0755148492)
    if t >= 66:
        b = 1.0
    elif t <= 19:
        b = 0.0
    else:
        b = 0.54320678911019607843 * math.log(t - 10) - 1.19625408914
    return (min(max(r, 0.0), 1.0), min(max(g, 0.0), 1.0), min(max(b, 0.0), 1.0))


def _blender_main() -> None:
    import bpy  # type: ignore[import-not-found]
    from mathutils import Vector  # type: ignore[import-not-found]

    argv = sys.argv[sys.argv.index("--") + 1 :]
    glb_path, scene_json, out_dir = argv[0], argv[1], argv[2]
    samples = int(argv[3])
    res_x, res_y = int(argv[4]), int(argv[5])
    n_views = int(argv[6])

    with open(scene_json) as fh:
        scene_data = json.load(fh)

    ad = scene_data.get("art_direction") or {}

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb_path)

    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = samples
    # An open-top model is lit almost entirely by the sky, which is a large, smooth
    # source — noise converges fast, but glass and its caustics do not, so the sample
    # count has to carry the transmission rather than the diffuse.
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.005
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 12
    # Indirect light is the majority of an interior's illumination, so the diffuse
    # bounce limit must be generous or rooms go muddy in the corners.
    sc.cycles.diffuse_bounces = 8
    sc.cycles.glossy_bounces = 6
    sc.cycles.transmission_bounces = 8
    sc.render.resolution_x = res_x
    sc.render.resolution_y = res_y
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX" if "AgX" in [
        t.identifier for t in sc.view_settings.bl_rna.properties["view_transform"].enum_items
    ] else "Filmic"
    # AgX/Filmic rather than Standard: Standard clips highlights hard, so a sunlit
    # floor blows out to flat white and the render reads as a screenshot.
    sc.view_settings.look = "None"

    # --- world: a simple sky, so windows look out onto something ---
    world = bpy.data.worlds.new("AtlasWorld")
    sc.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    sky = nt.nodes.new("ShaderNodeTexSky")
    try:
        sky.sky_type = "NISHITA"
        # The director's sun. A low elevation rakes across surfaces and models relief;
        # a high one flattens everything, which is why this is a decision and not a
        # constant.
        sky.sun_elevation = math.radians(float(ad.get("sun_elevation_deg", 38.0)))
        sky.sun_rotation = math.radians(float(ad.get("sun_azimuth_deg", 215.0)))
        sky.altitude = 120
    except Exception:
        pass
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(sky.outputs[0], bg.inputs[0])
    nt.links.new(bg.outputs[0], out.inputs[0])

    # --- 1. real glass -------------------------------------------------------
    #
    # glTF exports glazing as alphaMode BLEND at 16% opacity, which is a tinted film:
    # no refraction, no Fresnel, no reflection. Windows are the brightest thing in an
    # interior shot and that made them read as grey plastic. Rebuilt here as a genuine
    # transmissive BSDF with the IOR of glass.
    glass_fixed = 0
    for mat in bpy.data.materials:
        name = (mat.name or "").lower()
        if "glaz" not in name and "glass" not in name:
            continue
        mat.use_nodes = True
        nt_m = mat.node_tree
        nt_m.nodes.clear()
        out_n = nt_m.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt_m.nodes.new("ShaderNodeBsdfPrincipled")
        i = bsdf.inputs
        i["Base Color"].default_value = (0.94, 0.96, 0.97, 1.0)
        i["Roughness"].default_value = 0.02
        i["Metallic"].default_value = 0.0
        for key, val in (("Transmission Weight", 1.0), ("Transmission", 1.0)):
            if key in i:
                i[key].default_value = val
                break
        if "IOR" in i:
            i["IOR"].default_value = 1.52
        # Thin glass: a single pane modelled as a 12 mm slab would refract like a solid
        # block and darken the room behind it.
        for key in ("Thin Film Thickness", "Thin Film IOR"):
            if key in i:
                i[key].default_value = 0.0 if "Thickness" in key else 1.45
        nt_m.links.new(bsdf.outputs[0], out_n.inputs[0])
        mat.use_backface_culling = False
        try:
            mat.blend_method = "BLEND"
        except Exception:
            pass
        glass_fixed += 1
    print(f"[atlas] glass materials rebuilt: {glass_fixed}")

    # --- 2. bevelled arrises -------------------------------------------------
    #
    # Every edge in generated geometry is a mathematically perfect 90 degrees, and a
    # perfect edge catches no highlight. That single cue is what makes an otherwise
    # correct render read as CG. A 2 mm bevel is invisible as geometry and puts a thin
    # specular line on every arris, which is what real plaster and joinery do.
    bevelled = 0
    for ob in bpy.data.objects:
        if ob.type != "MESH" or not ob.data.polygons:
            continue
        bev = ob.modifiers.new(name="arris", type="BEVEL")
        bev.width = 0.002
        bev.segments = 2
        bev.limit_method = "ANGLE"
        bev.angle_limit = math.radians(35)
        bev.harden_normals = True
        # Harden normals needs smooth shading to have anything to blend.
        for poly in ob.data.polygons:
            poly.use_smooth = True
        try:
            ob.data.use_auto_smooth = True
            ob.data.auto_smooth_angle = math.radians(31)
        except AttributeError:
            pass  # Blender 4.1+ dropped auto-smooth for a modifier
        bevelled += 1
    print(f"[atlas] bevel applied to {bevelled} meshes")

    # --- 3. ground plane -----------------------------------------------------
    #
    # A cutaway model floating in void has no contact shadow, so it reads as a
    # weightless render rather than an object sitting somewhere. A large neutral plane
    # just below floor level gives the walls something to cast onto and grounds the
    # whole image.
    mn, mx = scene_data.get("bounds", [0, 0, 0, 0, 0, 0])[:3], scene_data.get("bounds", [0, 0, 0, 0, 0, 0])[3:]
    span = max(abs(mx[0] - mn[0]), abs(mx[2] - mn[2]), 8.0)
    bpy.ops.mesh.primitive_plane_add(size=span * 6.0, location=(0.0, 0.0, mn[1] - 0.02))
    ground = bpy.context.active_object
    ground.name = "ground"
    gmat = bpy.data.materials.new("ground")
    gmat.use_nodes = True
    gb = gmat.node_tree.nodes.get("Principled BSDF")
    if gb:
        gb.inputs["Base Color"].default_value = (0.20, 0.20, 0.21, 1.0)
        gb.inputs["Roughness"].default_value = 0.55
    ground.data.materials.append(gmat)

    # --- lights from SceneIR ---
    for spec in scene_data.get("lights", []):
        if spec.get("realtime_only"):
            # Cycles gets this daylight physically, through the same opening.
            continue
        kind = spec.get("kind")
        colour = _kelvin_to_rgb(float(spec.get("colour_k", 3000)))
        if kind == "sun":
            data = bpy.data.lights.new(spec["id"], type="SUN")
            data.energy = float(spec.get("intensity", 3.0))
            data.angle = math.radians(1.5)
            # Aim the directional light to match the sky's sun, or the cast shadows
            # contradict the light in the windows.
            az = math.radians(float(ad.get("sun_azimuth_deg", 215.0)))
            el = math.radians(float(ad.get("sun_elevation_deg", 38.0)))
            spec = dict(spec)
            spec["direction"] = [
                -math.cos(el) * math.sin(az),
                -math.sin(el),
                -math.cos(el) * math.cos(az),
            ]
        elif kind in ("area", "cove"):
            data = bpy.data.lights.new(spec["id"], type="AREA")
            data.shape = "RECTANGLE"
            sx, sy = spec.get("size", [0.4, 0.4])
            data.size = max(0.05, float(sx))
            data.size_y = max(0.05, float(sy))
            # Watts. Relative intensity is scaled modestly: an interior ceiling
            # fitting is tens of watts, not hundreds, and over-driving these was
            # what turned the first render into a white rectangle.
            data.energy = float(spec.get("intensity", 3.0)) * 6.0
        else:
            data = bpy.data.lights.new(spec["id"], type="POINT")
            data.energy = float(spec.get("intensity", 3.0)) * 6.0
        data.color = colour
        data.use_shadow = bool(spec.get("cast_shadow", True))
        obj = bpy.data.objects.new(spec["id"], data)
        obj["atlas_kind"] = "fill" if str(spec.get("id", "")).startswith("fill-") else kind
        sc.collection.objects.link(obj)
        pos = spec.get("position", [0, 0, 0])
        # glTF is Y-up, Blender is Z-up.
        obj.location = (pos[0], -pos[2], pos[1])
        d = spec.get("direction", [0, -1, 0])
        dv = Vector((d[0], -d[2], d[1]))
        if dv.length > 1e-6:
            obj.rotation_euler = dv.normalized().to_track_quat("-Z", "Y").to_euler()

    # --- render-only ceilings ------------------------------------------------
    # The delivered model is open-top so the web viewer can look in from above,
    # but a photographed interior is lit through its windows, not through a
    # missing roof. Ceilings are synthesised here per room from the scene's own
    # polygons and hidden again for the overview shot — the model on disk stays
    # ceiling-free.
    no_extras = os.environ.get("ATLAS_RENDER_NO_EXTRAS") == "1"  # diagnostic toggle
    no_ceil = no_extras or os.environ.get("ATLAS_RENDER_NO_CEILINGS") == "1"
    no_portal = no_extras or os.environ.get("ATLAS_RENDER_NO_PORTALS") == "1"
    ceil_mat = bpy.data.materials.new("atlas-ceiling")
    ceil_mat.use_nodes = True
    bsdf = ceil_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.93, 0.92, 0.90, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.92
    import mathutils
    import mathutils.geometry

    for ridx, room in enumerate([] if no_ceil else (scene_data.get("rooms") or [])):
        poly = room.get("render_ceiling_polygon") or room.get("polygon") or []
        if len(poly) < 3:
            continue
        # 50 mm below the wall tops. Coplanar with them, the sun grazes straight
        # through the seam and every wall-ceiling junction renders as a bright slit;
        # tucked below, the wall body itself blocks the gap. Nobody can see 50 mm of
        # ceiling height, everybody can see the leaks.
        #
        # The per-room stagger exists because expanded ceilings overlap their
        # neighbours', and two coplanar faces z-fight into black patches. A few
        # deterministic millimetres of separation is invisible from eye height.
        h = (
            float(room.get("floor_level", 0.0))
            + float(room.get("ceiling_height", 2.9))
            - 0.05
            - 0.004 * (ridx % 8)
        )
        verts = [(float(x), -float(z), h) for x, z in poly]
        # Room outlines are concave and raster-edged (hundreds of vertices); a single
        # n-gon over one renders as garbage that swallowed a whole camera view.
        # Tessellate to triangles instead.
        ring = [mathutils.Vector((v[0], v[1], 0.0)) for v in verts]
        faces = [tuple(tri) for tri in mathutils.geometry.tessellate_polygon([ring])]
        if not faces:
            continue
        mesh = bpy.data.meshes.new(f"ceil-{room['id']}")
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        ob = bpy.data.objects.new(f"ceil-{room['id']}", mesh)
        ob.data.materials.append(ceil_mat)
        ob["atlas_kind"] = "ceiling"
        sc.collection.objects.link(ob)

    # --- window portals --------------------------------------------------------
    # With real ceilings the interiors are lit through the glazing, which Cycles
    # samples badly blind: portals tell it where the light comes in, trading a
    # handful of area lights for a large drop in interior noise.
    rooms_xy = [
        (
            (float(r.get("centroid", [0, 0, 0])[0]), -float(r.get("centroid", [0, 0, 0])[2])),
            r,
        )
        for r in scene_data.get("rooms") or []
    ]
    portals = 0
    for op in [] if no_portal else (scene_data.get("openings") or []):
        if not op.get("glazed"):
            continue
        cx, cy, cz = (float(v) for v in op.get("centre", [0, 0, 0]))
        bx, by = cx, -cz
        w = float(op.get("width", 1.0))
        hgt = float(op.get("height", 1.2))
        zmid = float(op.get("sill", 0.0)) + hgt / 2.0
        data = bpy.data.lights.new(f"portal-{op['id']}", type="AREA")
        data.shape = "RECTANGLE"
        data.size = w
        data.size_y = hgt
        data.energy = 0.0
        data.cycles.is_portal = True
        ob = bpy.data.objects.new(f"portal-{op['id']}", data)
        ob["atlas_kind"] = "portal"
        sc.collection.objects.link(ob)
        ob.location = (bx, by, zmid)
        # An area light emits (and a portal admits) along its local -Z. Stand the
        # rectangle upright in the wall plane and face it at the nearest room's
        # centre — a portal facing outward degrades rather than helps.
        yaw = float(op.get("rotation_y", 0.0))
        near = min(rooms_xy, key=lambda t: (t[0][0] - bx) ** 2 + (t[0][1] - by) ** 2, default=None)
        import mathutils

        normal = mathutils.Vector((math.sin(yaw), math.cos(yaw), 0.0))
        if near is not None:
            to_room = mathutils.Vector((near[0][0] - bx, near[0][1] - by, 0.0))
            if normal.dot(to_room) < 0:
                normal = -normal
        ob.rotation_euler = normal.to_track_quat("-Z", "Z").to_euler()
        portals += 1
    if portals:
        print(f"[atlas] {portals} window portal(s) placed")

    # --- cameras: use the framing computed in assemble_scene ---
    # Framing is not recomputed here. It needs the room polygons and a containment
    # test to guarantee the camera stands inside an L-shaped room, which is exactly
    # what the build stage already did with Shapely in hand.
    cams = scene_data.get("cameras") or []
    if not cams:
        cx, _, cz = (scene_data.get("rooms") or [{}])[0].get("centroid", [0, 0, 0])
        cams = [
            {
                "id": "fallback",
                "name": "view",
                "position": [cx + 2.0, 1.55, cz + 2.0],
                "target": [cx, 1.55, cz],
                "focal_mm": 24.0,
            }
        ]
    cams = cams[: max(1, n_views)]

    written = 0
    for i, spec in enumerate(cams):
        cam_data = bpy.data.cameras.new(f"cam{i}")
        cam_data.lens = float(spec.get("focal_mm", 24.0))
        cam_data.sensor_width = 36.0
        cam_data.clip_start = 0.02
        cam = bpy.data.objects.new(f"cam{i}", cam_data)
        sc.collection.objects.link(cam)

        px, py, pz = spec["position"]
        tx, ty, tz = spec["target"]
        is_overview = spec.get("id") == "cam-overview"
        # The assemble-stage interior cameras float at ~3.25 m for the open-top
        # viewer — through a real ceiling that is a view of plaster from above,
        # which is exactly what the first ceilinged render produced: a white
        # frame. Interiors here are photographs, and the doctrine at the top of
        # this file already says how photographs are made: eye height, level.
        # The x/z stay as assemble placed them (guaranteed inside the room);
        # only the tripod height changes.
        if not is_overview and not no_ceil:
            py = 1.55
            ty = 1.45
        # glTF Y-up -> Blender Z-up.
        cam.location = (px, -pz, py)
        direction = Vector((tx, -tz, ty)) - Vector(cam.location)
        if direction.length > 1e-6:
            cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        sc.camera = cam

        # Two view types, two lighting worlds. The overview looks into an open-top
        # dollhouse: ceilings and portals hidden, sky pulled right down or it floods
        # everything, fill lights off because they hang below a ceiling that is not
        # there. An interior is a photograph inside a closed room: ceiling on, lit
        # through the glazing (hence the stronger sky and the portals), fills on.
        for ob in bpy.data.objects:
            kind_tag = ob.get("atlas_kind")
            if kind_tag in ("ceiling", "portal"):
                ob.hide_render = is_overview
            elif ob.type == "LIGHT" and kind_tag == "fill":
                ob.hide_render = is_overview
        bg.inputs[1].default_value = 0.5 if is_overview else 1.15
        base_exposure = float(ad.get("exposure", -0.1))
        sc.view_settings.exposure = -1.05 if is_overview else base_exposure

        name = str(spec.get("name") or spec.get("id") or f"view{i}").replace("/", "-")
        sc.render.filepath = os.path.join(out_dir, f"{i+1:02d}-{name}.png")
        bpy.ops.render.render(write_still=True)
        written += 1
        print(f"[atlas] rendered {sc.render.filepath}")

    print(f"[atlas] done, {written} image(s)")


if __name__ == "__main__" and "bpy" in sys.modules or (
    __name__ == "__main__" and any(a.endswith("Blender") or "blender" in a.lower() for a in sys.argv[:1])
):
    _blender_main()
elif __name__ == "__main__":
    try:
        import bpy  # type: ignore[import-not-found]  # noqa: F401

        _blender_main()
    except ImportError:
        print("This module must run inside Blender; use render() from the host side.")
