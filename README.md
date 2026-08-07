# Atlas — 2D floor plans to modular, furnished 3D

Upload a floor plan as **DWG/DXF** or as an **image**, get a dimensionally-accurate
3D model with furniture placed under real spatial constraints, where flooring, wall
finishes, joinery and individual furniture pieces are all swappable at runtime.

Verified end to end on a real 10 MB AutoCAD 2018 project drawing and on marketing
plan images.

---

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install ezdxf shapely numpy trimesh mapbox_earcut pillow pytest
brew install libredwg                      # DWG reading; not needed for DXF
```

On the first `build`, Atlas downloads ~35 CC0 PBR material sets from ambientCG
(~120 MB, cached in `data/materials/`) and 5 CC0 furniture models from Poly Haven
(~15 MB, `data/models/`). Both are cached; `--no-textures` and `--no-models` skip
them for a fast geometry-only run.

A production CAD file is a **whole floor**, not a flat — the sample holds 16
apartments. So list them first, then build the one you want:

```bash
# What dwellings does this floor contain?
PYTHONPATH=pipeline .venv/bin/python -m atlas units data/samples/demo-cad-1.dwg
#   #  type         declared   measured  rooms  mix
#   1  4 BHK         1558 sf      1364 sf     15  4x bedroom, 4x bathroom, 1x living...
#   5  3 BHK         1358 sf      1232 sf     16  3x bedroom, 1x living, 1x kitchen...
#  16  2 BHK          710 sf       622 sf     11  2x bedroom, 1x living, 1x dining...

# CAD path — exact measured geometry, no API key needed
PYTHONPATH=pipeline .venv/bin/python -m atlas build data/samples/demo-cad-1.dwg \
    --out out --unit 1 --export-web apps/web/public/data

# Image path — needs OPENROUTER_API_KEY in .env
PYTHONPATH=pipeline .venv/bin/python -m atlas build data/samples/3d-2.png --out out

# Photoreal hero renders (Cycles)
PYTHONPATH=pipeline .venv/bin/python -m atlas render out/scene.json out/model-warm-minimal.glb --out out/renders

# Interactive configurator
cd apps/web && npm install && npm run dev      # http://localhost:5188
```

Outputs: `plan.json` (PlanIR), `plan.svg` (extraction check drawing), `scene.json`
(SceneIR), `model-<scheme>.glb` per design scheme, `renders/*.png`.

---

## Research: the open-source landscape, and what we used

You asked whether any existing project could be used wholly or in part. Short
answer: **the vectorisation research is worth learning from, none of the
implementations are worth adopting**, and the genuinely reusable pieces are the
boring infrastructure — LibreDWG, ezdxf, Shapely, trimesh, Blender.

### Floor plan recognition

| Project | What it does | Verdict |
|---|---|---|
| [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) | 5,000-plan dataset + multi-task segmentation, 80+ classes, pretrained weights | **Not used.** PyTorch 1.0 / Python 3.6 era. Fatal issue is conceptual, not stale deps: it recovers *shape* but not *scale* |
| [DeepFloorplan](https://github.com/zlzeng/DeepFloorplan) | Room-boundary-guided attention, wall/opening/room maps | **Not used.** TF 1.10 + Python 2.7. GPL-3.0 |
| [FloorplanTransformation](https://github.com/art-programmer/FloorplanTransformation) (Raster-to-Vector) | The paper CubiCasa builds on | **Not used.** Lua/Torch |
| [RoomFormer](https://github.com/ywyue/RoomFormer) (CVPR'23) | Transformer, two-level polygon/corner queries, MIT | **Not used** — input is point-cloud density maps from RGB-D scans, not raster plans |
| [PolyRoom](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06708.pdf) (ECCV'24), [HEAT](https://github.com/woodfrog/heat) (CVPR'22) | Successors, better metrics | Same modality mismatch |
| [FloorplanVLM](https://arxiv.org/abs/2602.06507) (2026) | Reframes vectorisation as image→JSON sequence; 92.5% external-wall IoU | **Paradigm adopted, code not used.** This is the right idea and we implement it with a frontier general VLM instead of a fine-tuned 3B model |
| [FloorPlanTo3D-API](https://github.com/fadyazizz/FloorPlanTo3D-API) | Mask R-CNN, 3 classes (wall/window/door) | **Not used.** Only 3 classes, conversion to 3D undocumented |

**Why we didn't use any segmentation model.** They return a mask that says "wall
here". They cannot say the room is 11'0"×13'6". You then guess the scale from an
assumed drawing standard and *every dimension inherits the guess*. Your sample
plans print the room name and its dimensions inside every room, plus the carpet
area on the sheet. Reading that text gives ground truth instead of inference — so
the raster path is a vision model reading labels, and the CAD path measures
geometry directly and needs no ML at all.

### 2D→3D generation

| Project | Verdict |
|---|---|
| [FloorplanToBlender3d](https://github.com/grebtsew/FloorplanToBlender3d) (609★) | **Not used.** GPL-3.0 (blocks commercial use), no furniture, no scale calibration, and README states "images need to be quite small for detections to work" |
| [Plan2Scene](https://github.com/3dlg-hcvc/plan2scene) | **Not used.** Needs photos of the actual rooms to synthesise textures — not available for an unbuilt flat |
| [Infinigen Indoors](https://github.com/princeton-vl/infinigen) (BSD-3) | **Not used, most interesting.** Procedural Blender interiors with a constraint DSL — but it *generates* layouts; it cannot take your floor plan as input. Its constraint approach informed our solver |
| [Holodeck](https://github.com/allenai/Holodeck) (Apache-2.0) | **Not used, informed the solver.** LLM + DFS constraint solving over Objaverse. Generates from text; cannot accept a given plan. Its DFS-over-constraints approach is what our solver does |
| [ATISS](https://github.com/nv-tlabs/ATISS), [DiffuScene](https://github.com/tangjiapeng/DiffuScene), [LayoutGPT](https://github.com/weixi-feng/LayoutGPT) | **Not used.** Trained on 3D-FRONT room types; no architectural elements, and licence-encumbered assets |

### Web editors

| Project | Verdict |
|---|---|
| [blueprint3d](https://github.com/furnishup/blueprint3d) / [blueprint-js](https://github.com/aalavandhaann/blueprint-js) / [blueprint3d-modern](https://github.com/charmlinn/blueprint3d-modern) (MIT) | **Not used as a base.** Modern rewrite is genuinely nice (TS, three r181, MIT) but it is a *draw-your-own-plan* tool: no importer, OBJ-only loading, and its data model has no place for provenance or confidence |
| [arcada](https://github.com/mehanix/arcada), [homeRoughEditor](https://github.com/lcorcodilos/homeRoughEditor) | **Not used.** 2D editors, no 3D pipeline |
| Sweet Home 3D | **Not used.** GPL Java desktop |

### Infrastructure we *do* use

| Library | Role | Licence |
|---|---|---|
| **LibreDWG** `dwg2dxf` | The only free, redistributable DWG reader that handles AC1032 (AutoCAD 2018) | GPL-3 (separate binary, invoked as a subprocess) |
| **ezdxf** | DXF parsing, block flattening with transform composition | MIT |
| **Shapely** | Planar subdivision, polygonisation, constraint geometry | BSD-3 |
| **trimesh** | glTF/GLB assembly | MIT |
| **Blender 5.1 / Cycles** | Offline path-traced renders | GPL (separate binary) |
| **three.js + react-three-fiber** | Real-time configurator | MIT |
| **ambientCG** | PBR material maps (colour, normal, roughness, AO) | **CC0** — commercial use, no attribution needed, no per-asset audit |
| **Poly Haven** | CC0 furniture meshes for the detail items | **CC0** |
| **IfcOpenShell** | *Not yet wired* — the obvious route to BIM/IFC export | LGPL |

Asset sources evaluated: **ambientCG** (CC0) is what we use for materials — 35 sets
mapped to the scheme materials, fetched and cached by `atlas.build.textures`.
**Poly Haven** (also CC0) is an equally safe alternative. **Objaverse** and
**3D-FUTURE** are per-object licensed, which means auditing every asset before
shipping, so the furniture catalogue is parametric instead (see below).

### LLM use

Used in exactly one place, where it is genuinely better than the alternative:
**reading a raster plan into structured data.** It is *not* used for geometry —
language models do not produce metrically consistent coordinates, and a plan built
from hallucinated coordinates is confidently wrong in a way that is hard to detect.
The model reports what it can see (labels, printed dimensions, adjacency, openings)
and Python solves the geometry.

---

## Architecture

```
DWG ──dwg2dxf──> DXF ──repair──> ezdxf ──flatten──> layer classify ─┐
                                                                    ├─> PlanIR ─> SceneIR ─> GLB
Image ──────────> vision model ─> rooms + printed dims + adjacency ─┘         │        │
                                                                              │        └─> Cycles renders
                                                                              └─> web configurator
```

Two intermediate representations do the work:

**PlanIR** (`pipeline/atlas/planir.py`) — the metric 2D plan. Millimetres always.
Walls are **centrelines plus thickness**, not rectangles, because a rectangle-per-wall
representation cannot express a T-junction without a gap or an overlap. Every
element carries `source` and `confidence`, and `Room.stated_dimension` keeps the
*printed* dimension separately from the measured polygon so disagreement is a
reported fact rather than something averaged away.

**SceneIR** (`pipeline/atlas/sceneir.py`) — the 3D scene. Nothing visual is baked
in: every surface carries a **material slot** (`floor.living`, `wall.accent`, …)
resolved through a separate `DesignScheme`, and furniture is a catalogue reference
plus a transform. That is what makes swapping instant rather than a rebuild — the
data design is the product feature.

### Stage order: map the house, then build it

The per-dwelling stages run in this order, and the order is the design:

| # | Stage | Module | Produces |
|---|-------|--------|----------|
| 1 | **Map the complete house** | `ingest/spaces.py` | every space, gap-free, plus the adjacency graph |
| 2 | **Extrude walls** | `_walls_from_boundaries` | one wall per space boundary |
| 3 | **Place openings** | `_attach_openings`, `_doors_from_topology` | doors on the boundaries that need them |
| 4 | **Furnish by space type** | `furnish/` | layout constrained per `RoomKind` |

It used to run **walls-first**: pair the drawn wall faces, then take the leftover
cycles of that network as the rooms. That made rooms a derivative of the noisiest step
in the pipeline, and it failed the way you would expect — measured on the demo flat,
**24% of the footprint (39.6 m²) belonged to no room at all**, four fragments came out
unlabelled (one of them 154 sqft), and neighbouring spaces sat 150–240 mm apart with
nothing in between. Living and dining were two rooms with a mystery strip between them.

The evidence points the other way. The drawing *declares* its spaces — sixteen labels,
each with a printed size — and that is ground truth; wall geometry is exact where it was
recovered and simply missing where it was not. So space is now the primitive:

- Each label is grown to its **printed area** over a raster of the dwelling footprint,
  with recovered walls acting as **barriers**. The partition is complete by
  construction: there is nowhere for an "unknown" to hide.
- Growth uses **Chebyshev distance in the drawing's own wall directions**, so rooms come
  out rectangular. Euclidean distance grows discs — the living room was a rounded blob
  and the walls derived from it ran diagonally across the plan.
- Printed sizes are **absolute** targets, not ratios scaled onto the footprint. Scaling
  them up turned a 40 sqft toilet into 121 sqft, because the footprint also holds wall
  thickness and unlabelled passage.
- A large leftover region is **named as circulation**, not absorbed. Absorbing it gave a
  40 sqft toilet a 2.8 × 4.8 m footprint. A large leftover touching a room that measured
  *short of its printed size* is returned to that room first — that recovered the living
  room from 135 sqft to 190 against a printed 204.
- **A wall is built on its boundary line, and takes only its *thickness* from the matched
  CAD wall — never its position.** This is the single most important rule in the module.
  Adopting the recovered centreline instead (a match is accepted up to 260 mm away) put
  every wall off the boundary it divided, so the floor stopped in one place and the wall
  stood in another: the kitchen measured **52% enclosed**, a toilet 69%, a bedroom 77%.
  In the render that is a maze of disconnected wall stubs. Position from the partition is
  not a compromise — growth is blocked *at* the recovered walls, so a boundary already
  sits within one cell (90 mm) of the real wall, and it is guaranteed to coincide with the
  floors either side of it.
- Adjacency is read off the **cell partition**, not from polygon outlines with a distance
  tolerance. Tolerance tuning made the metrics oscillate rather than converge, which is
  the signature of a wrong approach rather than a wrong constant.
- Collinear runs are **fused into one wall** — a wall with a door in it is one wall, not
  three. The merge keys on height and parapet-ness as well as position and thickness, or a
  1.05 m balcony guard fuses into a collinear partition and comes out 2.9 m tall.
- A **45° interface**, where two growth fronts meet, is re-split along one straight line.
  No neighbourhood filter can remove it: every tooth of a staircase has exactly two
  neighbours of each owner, so majority rules see nothing to fix.
- Doors are placed by **walking the adjacency graph** out from the front door, weighted
  so rooms are entered from circulation. Choosing the longest shared wall alone put a
  door directly between two bedrooms.

Open plan needs positive evidence, and the only honest evidence is the pair of space
types: living into dining may be continuous, a bedroom into a corridor may not. Treating
every unmatched boundary as open left 41 of 109 boundaries wall-less, which made every
room trivially reachable and meant **not one door was placed in the whole flat**.

### Things worth knowing

**Real CAD is not flat.** The sample drawing has 33 entities in modelspace and
**8,104 block definitions** — every wall lives inside a block inside a block.
Reading modelspace alone finds nothing. `dxf_flatten.py` recurses with transform
composition (handling the `xscale=-1` mirrored units), and normalises xref-bound
layer names (`...OS$0$ABA_02_wall` → `ABA_02_wall`).

**Draughtsmen draw wall faces, not walls.** `wall_solver.py` inverts that: cluster
line work by direction, rotate into wall-local coordinates, pair opposing faces
thinnest-first (so a 150 mm partition abutting a 230 mm structural wall doesn't
marry the outer two rails into a 380 mm wall that doesn't exist), then merge and
snap junctions.

**Rooms are the planar subdivision of the wall graph, plus openings.** Subtracting
wall solids from the unit boundary *does not work* — one gap in the envelope merges
every room in the flat into a single region, silently. Noding the centrelines and
taking minimal cycles has no such dependence. Openings must be part of the
enclosure too: a bedroom opens onto its balcony through a full-width sliding door
with only 100 mm wall stubs, and without a barrier there the two are one room.

**Openings are cut analytically, not with CSG.** A wall minus its openings
decomposes exactly into piers, lintels and aprons — all boxes. Clean quads,
predictable UVs, no coplanar-face artifacts, fast.

**Textures carry the material, base colour carries the design.** The eye reads a
material from its microstructure — oak grain, tile grout, linen weave — before it
reads colour, so scalar-only materials always look like a massing study. Maps come
from ambientCG at 1K, served as separate files to the browser and embedded at 512 px
in the GLB. Two details matter: UVs are in **world metres** and divided by each
material's `texture_scale_m`, so a 600 mm tile stays 600 mm and a 1.9 m oak plank
stays 1.9 m; and for painted or plastered surfaces the colour map is **neutralised**
to luminance only, because multiplying a paint tint by a photograph's own hue
compounds the two (an off-white tint over warm plaster produced visibly pink walls).

**A region larger than its own label gets split by the printed dimension.** One face
came out at 287 sqft carrying a single BEDROOM label the drawing prints as 132 sqft — it
had swallowed an adjacent space whose label was clipped away. Since the drawing states
how big the labelled room is, that footprint is carved off and the remainder reported as
an *unnamed* space rather than as a bedroom two and a half times its stated size. That
one fix moved mean agreement from 0.851 to 0.889.

**A room with no doorway gets one inferred.** A room you cannot enter is not a room.
Where the drawing neither cuts its wall faces nor draws a door block, the doorway is
still certainly there — someone has to get in. Three were added on the longest wall each
room shares with a neighbour, centred on the overlap where both rooms meet it, marked
`Source.INFERRED` at 0.35 confidence with a warning on the plan. Their swing is a *soft*
constraint for the layout solver: a guessed door must not evict a real sofa, whereas a
measured door's swing is inviolable.

**Openings are recovered from the wall faces, not just from door symbols.** The demo
drawing contains **three door blocks for a sixteen-room flat** — symbol detection
alone cuts a hole for barely a fifth of the doorways and leaves the rest of the flat
as sealed boxes. But every doorway *is* cut through both wall faces, and the wall
solver already locates those cuts in order to bridge them. It now reports them, and
a nearby symbol only supplies the *kind* (door / window / cased opening). Requiring
both faces to be absent is what distinguishes an opening from a recess or a chase.

**Art direction is encoded, not left to the viewer.** Correct geometry, physical
materials and path-traced light still produced images that read as models rather than
rooms, and the reason was not fidelity — it was that nobody had *composed* them. Three
things a practice does, written down:

*Composition.* Camera views are **scored**, not merely placed legally. Candidates are
generated along both principal axes, and each is rewarded for what falls inside the
frame: a window (weighted highest — it brings the light and the depth), furniture (scale
and occupancy), and view depth. Subjects slightly off-axis score higher than
dead-centre ones. Views whose direction leaves the building are rejected outright — the
earlier version produced a frame that was 60% sky gradient through a missing envelope
wall, and a perimeter-enclosure test alone does not catch it, because a room can be
walled on three sides with the camera aimed at the fourth.

*Dressing.* Curtains at every window in a habitable room, and framed art on each accent
wall. A bare window reads as a hole, and blank walls read as unfinished; both are among
the strongest tells that an image is a model. Folds are alternating slabs at two depths
so a panel catches a gradient rather than rendering as a flat card, and art frames are
deep with the canvas set back so they catch a shadow line. Art is never hung on a wall
that has an opening in it.

Crucially these are emitted as **ordinary catalogue placements**, so the browser viewer
and the Cycles render both pick them up with no renderer-specific code. Their position
comes from the architecture (the window, the accent wall) rather than from the layout
solver, because neither is a search problem.

**Furniture is parametric where size is the requirement, modelled where detail is.**
A queen bed is 1524 mm because that is what a queen mattress is; a counter is 900 mm
high and 600 mm deep because that is the standard. A downloaded mesh is whatever size
its author made it, usually unstated, and scaling one to fit a slot makes it no longer
the object it claims to be. So beds, wardrobes, counters and sanitaryware stay
parametric — and they are box-like anyway, so a real mesh buys little.

The small, detailed, organically-shaped items are the opposite case: their silhouette
is what reads as "real" and none needs an exact dimension. Coffee table, side table,
armchair, cabinet and plant are **real CC0 meshes from Poly Haven**, fitted to their
declared footprint with a *uniform* scale (a non-uniform fit turns a coffee table into
a squashed caricature) and with the model's longer horizontal axis aligned to the
item's, because glTF authors do not agree on which way an object faces.

Most of Poly Haven's 85 furniture models are deliberately **not** used: they are
vintage, gothic, farmhouse and industrial. A `GothicBed_01` in the Japandi scheme
would look worse than clean massing, because that failure is one of taste rather
than of fidelity.

**The solver falls back within a category.** Filtering by area only asks "is the room
big enough?", and a room can have ample area in a shape that defeats the item: the
demo flat's living zone is 12.3 m² but shaped as a ribbon, and a 2.1 m three-seat sofa
with its 900 mm clearance has nowhere to go. The selector now offers alternatives
largest-first and places the first that actually fits, so the room gets the 1.6 m
two-seater instead of nothing.

**The layout solver treats placement as constraints, not preferences.** Candidates
are generated by sliding each item along each bounding wall; a candidate is
*rejected* if it violates declared clearance, overlaps another item, or blocks a
door swing (a quarter-disc of floor). Surviving candidates are scored on design
judgement — longest wall for a bed, view of a window for seating, circulation kept
clear. Fixtures read from the CAD file (sanitaryware, kitchen counters) are
**locked**: they encode plumbing stacks that a heuristic must not move.

---

## Verified results

**CAD path** — `demo-cad-1.dwg`, 10 MB, AutoCAD 2018 (AC1032):

- LibreDWG → 69 MB DXF in **1.4 s**
- It emits DXF that neither `ezdxf.readfile` nor `ezdxf.recover` can read: an
  unescaped newline inside an MTEXT note breaks group-code/value line parity.
  `dxf_repair.py` restores parity → loads with 1,463 fixes
- Flattened **4,148 entities** from 581 INSERTs, depth 5, **0 errors**, 0.3 s
- Segmented **16 dwellings** from the drawing's own RERA carpet-area polygons
  (rejecting the 636,350 sqft plot boundary and 5 sqft niches)
- Unit 1 (4 BHK): **15 spaces** (14 labelled + 1 unlabelled passage), **74 walls**,
  **24 openings**; wall thicknesses recovered as 150 mm ×25, 200 mm ×2, 100 mm ×2
- **The space map covers 100% of the dwelling footprint** and leaves **zero unclassified
  rooms**. Before the space-first restructure the same flat left 24% of its footprint
  (39.6 m²) belonging to no room and four fragments unlabelled
- **14 of 15 rooms are 100% enclosed.** The fifteenth is DINING at 87%, which is correct —
  the shortfall is exactly its open threshold into LIVING. Measured by taking each room's
  perimeter and asking what fraction of it has a wall solid against it
- **Every one of the 25 openings is genuinely cut through its wall**, verified by
  projecting every wall triangle into wall-local 2D and confirming none covers an opening
  rectangle. This is the check behind "you can see in and out of the door"
- **0 walls run through a room**, **0 non-axis-aligned walls**, **0 overlapping rooms**,
  **0 kind/exterior mismatches**
- **All 31 furniture items sit inside the room they claim**, none badly overlapping
- GLB: 58 meshes, 142k triangles, **0 with missing faces, zero area or absent UVs**;
  bounds 15.5 × 12.3 m, height 2.95 m
- Room floor areas total **1,364 sqft**. The unit's declared polygon is 1,557 sqft, which
  includes the wall footprints the floors exclude — roughly 300 sqft of 150 mm partitions
  across this flat, so the two reconcile

Getting there took fixing seven distinct defects, the largest being that walls were built
on the matched CAD *centreline* rather than on the space boundary. The full list, with what
each one did to the output, is in the git history and in `spaces.py`'s comments
- **33 scene items**: 26 solved by the layout constraint solver, **7 backed by real CC0
  meshes**, plus a styling pass adding curtains at every habitable window and framed art
  on accent walls — every one verified to sit inside the room it claims
- **Ceiling closed over the wall zones**, so interiors no longer show sky through the
  strips above partitions
- **34 CC0 material sets** + **5 CC0 models**; GLB **~22 MB** per scheme with maps and
  meshes embedded, 58,194 triangles (0.1 MB / 2,928 triangles with
  `--no-textures --no-models`)

**Image path** — `3d-2.png` (3 BHK marketing plan):

- Read **15/15 rooms** with correct labels, printed dimensions and adjacency
- Room areas total **1,298 sqft** against the sheet's printed **1,358 sqft** — a
  **96% independent cross-check**
- That cross-check caught a real bug: `CARPET AREA:1358 SQFT` parsed as **358**
  (regex matched a suffix of the number). Fixed, with a regression test

**All three sample inputs, end to end:**

| Input | Rooms read | Measured | Printed | Notes |
|---|---|---|---|---|
| `demo-cad-1.dwg` 4 BHK (unit 1 of 16) | 15 | 1,364 sqft | 1,558 | 14/15 rooms 100% enclosed, 0 unclassified |
| `demo-cad-1.dwg` 3 BHK (unit 5 of 16) | 16 | 1,232 sqft | 1,358 | per-room agreement **0.951** |
| `3d-1.png` 2.5 BHK | 14 | 702 sqft labelled | 950 | 14/14 printed dimensions read exactly |
| `3d-2.png` 3 BHK | 15 | 1,298 sqft | 1,358 | 96% area cross-check |

The three samples are **independent projects**, not three views of one. A unit-similarity
check across all 16 CAD dwellings puts `3d-2.png`'s best match at 8% (2 shared rooms of
15) — the 1358 sqft coincidence with CAD unit 5 is just a coincidence.

**Rendering** — Blender 5.1.2 Cycles. One ceiling-off doll's-house overview plus
interior views at 24 mm and 1.55 m eye height, camera kept level so verticals stay
vertical. Interior views are **gated on room shape**: a room must be at least 7 m²,
at least 2.2 m in its narrow direction, and fill ≥60% of its bounding box. Rooms
failing the gate are skipped rather than photographed badly — the wedge-shaped zones
produced by the open-plan split cannot be framed with a 24 mm lens. Output in
`out/renders/`.

**Configurator** — verified in-browser: all four schemes, per-slot material
override, room selection with camera jump, furniture swap/hide. Scheme change
re-renders instantly with no geometry rebuild. 35 of 38 meshes carry colour, normal
and roughness maps, 66 GPU textures, 174k triangles — including the five real Poly
Haven meshes, so the browser preview and the Cycles render agree.

One robustness fix worth recording: the viewer would not mount at all in a page whose
`visibilityState` is `hidden` (a background tab, a prerendered page, a restored
session). R3F waits for ResizeObserver before mounting the Canvas subtree, and a
browser delivers ResizeObserver callbacks only during a rendering step, which it skips
entirely for a hidden page — so it waited forever, with no error anywhere. A polling
observer polyfill removes the dependency.

Tests: `cd pipeline && ../.venv/bin/python -m pytest tests/ -q` → **37 passed**.

---

## Honest limitations

1. **A 330 sqft space in the demo flat carries no label, and is called PASSAGE.** The
   space map is complete by construction, so every square metre inside the dwelling
   boundary has to be assigned to something. This region is inside the drawing's own
   declared carpet area (our total cross-checks to 97% of it) but the drawing never names
   it. Calling it circulation is the honest reading — it is certainly not more bathroom —
   but 330 sqft is a lot of circulation, and part of it may belong to the living room or
   to a foyer the draughtsman left unlabelled. **This is the one place an LLM would
   genuinely help**: naming an unlabelled space from the drawing image is a semantic
   judgement, unlike the geometry, which has ground truth and stays deterministic.

2. **Where the boundary between two open-plan zones falls is a judgement, not a
   measurement.** Living and dining are one continuous space here; the split between them
   is placed to match each label's printed area, which is the best available evidence but
   is not something the drawing records. LIVING lands at 190 sqft against a printed 204,
   LOBBY at 23 against 39.

3. **Nine of twenty-three openings are inferred, not measured.** This drawing carries
   only three door blocks for the whole flat and draws most partitions continuous through
   their doorways. Those doors are now placed by walking the space adjacency graph out
   from the front door, weighted so rooms are entered from circulation — so each one sits
   on a boundary two named spaces actually share, and every room ends up reachable.
   Presence is near-certain and the pairing is reasoned; the exact position along the
   boundary is still a placement rather than a reading.

4. **The image path approximates arrangement.** Room *sizes* come from printed
   dimensions and are trustworthy — 14/14 read exactly on `3d-1.png`. The *layout* is
   packed row by row, edge to edge, from the model's reported adjacency, which gets
   rooms sharing walls (openings rose 7 -> 11 once packing was fixed) but fills only
   ~64% of its envelope and will not reproduce a non-rectangular plan outline. Use
   DWG/DXF when exact geometry matters — the PlanIR says so itself in its warnings.

5. **Interior camera framing is scored, and still selective.** Views are ranked on what
   falls in frame (window, furniture, depth, off-centre subjects) and must clear a
   minimum composition score, be enclosed, and terminate in a wall. Two or three of
   eighteen rooms qualify; the rest are covered by the doll's-house overview. The scorer
   picks a *defensible* frame, not an inspired one — a hero shot of a specific awkward
   room still needs a hand-placed camera.

6. **Single storey.** `Level` exists in the schema and multi-storey was designed for,
   but stairs are not connected between levels.

7. **No IFC/BIM export yet.** IfcOpenShell is the route; PlanIR has everything needed
   (walls with thickness, openings hosted on walls, room boundaries).

8. **Beds, wardrobes, counters and sanitaryware are parametric.** They are built in
   layers — plinth, mattress, turned-down duvet, stacked pillows, folded throw; sofa
   with per-seat cushions, rolled arms and offset scatter cushions — and textured with
   real fabric maps, which is a long way from a single box. They remain generated
   geometry, not photographed product. No CC0 source has modern beds or sofas; a
   licensed library wired through `external_model` closes it, and that path is proven
   working on both the GLB/Cycles and browser renderers.

9. **`furnish_plan` cannot fit ~26 items** in the tightest rooms, mostly bedside tables
   and plants. Each rejection is reported with its reason rather than silently dropped.

## Suggested next steps, in value order

1. A licensed modern furniture library through `external_model` (fixes 7)
2. IFC export via IfcOpenShell (fixes 6)
3. Read door schedules where drawings carry them, replacing inference with measurement
   (fixes 2)
4. Multi-storey with stair connection (fixes 5)
5. Let the LLM propose *design intent* per room (mood, palette, furniture set) feeding
   the existing solver — the plumbing is already there

## Layout

```
pipeline/atlas/
  units.py            unit + label parsing (feet-inches, areas, scale inference)
  planir.py           canonical 2D plan IR
  sceneir.py          3D scene IR (material slots, placements, cameras)
  cli.py              build / render / export-data
  ingest/
    dxf_repair.py     restore DXF line parity that LibreDWG breaks
    dxf_flatten.py    recursive block flattening + xref layer normalisation
    layers.py         layer name -> semantic role (ABA_*, AIA A-*, non-Latin)
    wall_solver.py    face pairs -> centrelines + thickness
    dxf_plan.py       orchestration: scale, unit segmentation, rooms, openings
    raster_vlm.py     vision-model ingest for images
  build/
    solids.py         analytic wall/floor/ceiling/skirting geometry
    schemes.py        four curated design schemes (all slots validated)
    textures.py       ambientCG CC0 map acquisition, caching, neutralised variants
    models.py         Poly Haven CC0 furniture mesh acquisition
    assemble.py       PlanIR -> SceneIR: accent walls, lighting, camera framing
    gltf.py           GLB export
  furnish/
    catalog.py        25 parametric items with real dimensions + placement rules
    solver.py         constraint-based layout
  render/
    plan_svg.py       presentation + debug plan drawings
    blender_render.py headless Cycles hero renders
  tests/              37 tests, focused on unit/label parsing
apps/web/             React + three.js configurator
```
