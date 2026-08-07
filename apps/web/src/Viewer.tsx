/**
 * Real-time viewer.
 *
 * Geometry is built from SceneIR *once* and memoised; materials are resolved from
 * the active scheme on every render. That split is the whole reason a scheme change
 * or a floor swap is instantaneous instead of a rebuild — it is the data design of
 * SceneIR showing up as a product property.
 *
 * The scene loads from `scene.json`, not from the GLB. The GLB has materials baked
 * in and is the right artefact for handing to someone else; for a configurator we
 * want the slots still unresolved so they remain swappable.
 */

import { OrbitControls } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { EffectComposer, N8AO } from "@react-three/postprocessing";
import { Component, Suspense, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { kelvinToRgb, resolveMaterial, useStore } from "./store";
import type { CatalogItem, FurniturePlacement, Material, OpeningSpec, SceneIR, Surface } from "./types";

/**
 * A ResizeObserver that also works when the browser is not painting.
 *
 * react-three-fiber will not mount the Canvas subtree until its measurement hook
 * reports a non-zero size, and that hook is driven by ResizeObserver. A browser
 * only delivers ResizeObserver callbacks during a rendering step, and it skips
 * rendering steps entirely for a page whose `visibilityState` is `hidden` — a
 * background tab, a prerendered page, a restored session. In that state the
 * observer never fires, R3F waits forever, and the viewer is a blank rectangle with
 * nothing in the console and no error thrown. It is a genuinely nasty failure mode
 * because the app is completely correct.
 *
 * Passing this as the observer polyfill removes the dependency: it measures with
 * `getBoundingClientRect` immediately and then on a slow timer, both of which run
 * regardless of visibility. The timer costs nothing (it stops once the size is
 * stable and there are no observed elements) and the real ResizeObserver is not
 * needed for correctness — only for promptness.
 */
class PollingResizeObserver {
  private cb: (entries: { target: Element; contentRect: DOMRectReadOnly }[]) => void;
  private targets = new Set<Element>();
  private last = new Map<Element, string>();
  private timer: number | null = null;

  constructor(cb: (entries: any[]) => void) {
    this.cb = cb;
  }

  observe(target: Element) {
    this.targets.add(target);
    this.tick();
    if (this.timer === null) {
      this.timer = setInterval(() => this.tick(), 250) as unknown as number;
    }
  }

  unobserve(target: Element) {
    this.targets.delete(target);
    this.last.delete(target);
    if (!this.targets.size) this.disconnect();
  }

  disconnect() {
    if (this.timer !== null) clearInterval(this.timer);
    this.timer = null;
    this.targets.clear();
    this.last.clear();
  }

  private tick() {
    const changed: { target: Element; contentRect: DOMRectReadOnly }[] = [];
    for (const target of this.targets) {
      const rect = target.getBoundingClientRect();
      const key = `${Math.round(rect.width)}x${Math.round(rect.height)}`;
      if (this.last.get(target) !== key) {
        this.last.set(target, key);
        changed.push({ target, contentRect: rect as DOMRectReadOnly });
      }
    }
    if (changed.length) this.cb(changed);
  }
}

/**
 * One shared texture cache for the page.
 *
 * Materials are rebuilt on every scheme change and per-slot override. Loading the
 * images again each time would re-download and re-decode megabytes on every click,
 * so textures are cached by URL and outlive the materials that reference them —
 * which also means `material.dispose()` must never dispose these.
 */
const _texCache = new Map<string, THREE.Texture>();

function loadTexture(url: string, srgb: boolean): THREE.Texture {
  const key = `${url}|${srgb}`;
  const hit = _texCache.get(key);
  if (hit) return hit;
  const tex = new THREE.TextureLoader().load(url);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  // Colour maps are sRGB-encoded; normal/roughness/AO are linear data. Tagging a
  // data map as sRGB washes out the roughness and flattens the normals.
  tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  tex.anisotropy = 16;
  _texCache.set(key, tex);
  return tex;
}

/**
 * Slots whose colour is a paint choice rather than a material photograph.
 * Mirrors `Material.texture_neutralize` on the Python side.
 */
const NEUTRAL_SLOTS = new Set([
  "wall.interior",
  "wall.accent",
  "wall.exterior",
  "ceiling",
  "skirting",
  "parapet",
  "floor.circulation",
]);

/** Build a THREE material from a scheme material. Memoised by the caller. */
function makeMaterial(
  m: Material | null,
  slot: string,
  manifest: Record<string, Record<string, string>>,
  assetBase = "",
): THREE.Material {
  if (!m) {
    // Loud magenta rather than plausible grey: a missing material should be
    // impossible to ship unnoticed.
    return new THREE.MeshStandardMaterial({ color: 0xff00cc, roughness: 0.5 });
  }
  const colour = new THREE.Color(...m.base_color).convertSRGBToLinear();
  const transparent = m.opacity < 0.999;
  if (slot === "glazing" || transparent) {
    // Physical glass: real IOR and transmission, so windows read as glass rather
    // than as tinted plastic sheets.
    return new THREE.MeshPhysicalMaterial({
      color: colour,
      roughness: m.roughness,
      metalness: 0,
      transparent: true,
      opacity: Math.max(0.08, m.opacity),
      transmission: 0.9,
      thickness: 0.01,
      ior: 1.52,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
  }
  const mat = new THREE.MeshPhysicalMaterial({
    color: colour,
    roughness: m.roughness,
    metalness: m.metallic,
    side: THREE.FrontSide,
  });
  if (m.sheen > 0) {
    // Fabric without sheen reads as painted plastic.
    mat.sheen = m.sheen;
    mat.sheenRoughness = 0.7;
    mat.sheenColor = new THREE.Color(1, 1, 1);
  }

  const maps = m.acg_asset ? manifest[m.acg_asset] : undefined;
  if (maps) {
    // UVs are in world metres, so one texture tile per `texture_scale_m` metres is
    // exactly repeat = 1/scale. This keeps a 600 mm tile 600 mm instead of
    // stretching one tile across a whole floor.
    const rep = 1 / Math.max(m.texture_scale_m, 1e-6);
    const url = (f: string) => `${assetBase}/data/textures/${f}`;
    const apply = (t: THREE.Texture) => {
      t.repeat.set(rep, rep);
      return t;
    };
    const chosen = NEUTRAL_SLOTS.has(slot)
      ? maps["base_color_neutral"] ?? maps["base_color"]
      : maps["base_color"] ?? maps["base_color_neutral"];
    if (chosen) mat.map = apply(loadTexture(url(chosen), true));
    if (maps["normal"]) {
      mat.normalMap = apply(loadTexture(url(maps["normal"]), false));
      mat.normalScale = new THREE.Vector2(0.8, 0.8);
    }
    if (maps["roughness"]) {
      mat.roughnessMap = apply(loadTexture(url(maps["roughness"]), false));
      mat.roughness = 1.0; // the scalar multiplies the map
    }
    if (maps["ao"]) {
      mat.aoMap = apply(loadTexture(url(maps["ao"]), false));
      mat.aoMapIntensity = 0.7;
    }
    mat.needsUpdate = true;
  }
  return mat;
}

/** Merge SceneIR surfaces into one BufferGeometry per material slot. */
function useArchGeometry(scene: SceneIR, showCeilings: boolean) {
  return useMemo(() => {
    const bySlot = new Map<string, { pos: number[]; nor: number[]; uv: number[]; idx: number[] }>();
    for (const s of scene.surfaces as Surface[]) {
      if (!showCeilings && s.kind === "ceiling") continue;
      let g = bySlot.get(s.slot);
      if (!g) {
        g = { pos: [], nor: [], uv: [], idx: [] };
        bySlot.set(s.slot, g);
      }
      const offset = g.pos.length / 3;
      g.pos.push(...s.vertices);
      if (s.normals?.length === s.vertices.length) g.nor.push(...s.normals);
      else for (let i = 0; i < s.vertices.length / 3; i++) g.nor.push(0, 1, 0);
      if (s.uvs?.length === (s.vertices.length / 3) * 2) g.uv.push(...s.uvs);
      else for (let i = 0; i < s.vertices.length / 3; i++) g.uv.push(0, 0);
      for (const i of s.indices) g.idx.push(i + offset);
    }

    const out: { slot: string; geometry: THREE.BufferGeometry }[] = [];
    for (const [slot, g] of bySlot) {
      if (!g.idx.length) continue;
      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.Float32BufferAttribute(g.pos, 3));
      geom.setAttribute("normal", new THREE.Float32BufferAttribute(g.nor, 3));
      const uvAttr = new THREE.Float32BufferAttribute(g.uv, 2);
      geom.setAttribute("uv", uvAttr);
      // three samples aoMap from uv1, not uv; without this the AO map does nothing.
      geom.setAttribute("uv1", uvAttr.clone());
      geom.setIndex(g.idx);
      geom.computeBoundingSphere();
      out.push({ slot, geometry: geom });
    }
    return out;
  }, [scene, showCeilings]);
}

/** Boxes for furniture and openings, grouped by slot, as one geometry each. */
function useBoxGeometry(
  placements: FurniturePlacement[],
  catalog: Record<string, CatalogItem>,
  openings: OpeningSpec[],
  swaps: Record<string, string>,
  hidden: Set<string>,
  showFurniture: boolean,
  skipIds: Set<string> = new Set(),
) {
  return useMemo(() => {
    type Box = { c: [number, number, number]; s: [number, number, number]; yaw: number };
    const bySlot = new Map<string, Box[]>();
    const push = (slot: string, b: Box) => {
      const arr = bySlot.get(slot);
      if (arr) arr.push(b);
      else bySlot.set(slot, [b]);
    };

    if (showFurniture) {
      for (const pl of placements) {
        if (hidden.has(pl.id) || skipIds.has(pl.id)) continue;
        const item = catalog[swaps[pl.id] ?? pl.catalog_id];
        if (!item) continue;
        const [px, py, pz] = pl.position;
        const yaw = pl.rotation_y;
        const cos = Math.cos(yaw);
        const sin = Math.sin(yaw);
        // Plan dimensions honour the placement's scale, height does not — matching the
        // GLB exporter and the solver. The solver may place an item smaller than its
        // catalogue size in a tight room and reserves floor for the size it chose; drawing
        // it full size would overlap its neighbours while the collision model believed it
        // fitted.
        const [sx, , sz] = pl.scale ?? [1, 1, 1];
        for (const part of item.parts) {
          const lx = (part.x / 1000) * sx;
          const ly = part.y / 1000;
          const lz = (part.z / 1000) * sz;
          push(pl.material_overrides[part.slot] ?? part.slot, {
            c: [px + lx * cos + lz * sin, py + ly, pz - lx * sin + lz * cos],
            s: [(part.w / 1000) * sx, part.h / 1000, (part.d / 1000) * sz],
            yaw,
          });
        }
      }
    }

    for (const op of openings) {
      const frame = 0.06;
      const depth = Math.max(0.05, op.thickness * 0.9);
      const [cx, cy, cz] = op.centre;
      const cos = Math.cos(op.rotation_y);
      const sin = Math.sin(op.rotation_y);
      const add = (slot: string, lx: number, ly: number, sx: number, sy: number, sz: number) =>
        push(slot, { c: [cx + lx * cos, cy + ly, cz - lx * sin], s: [sx, sy, sz], yaw: op.rotation_y });

      add(op.frame_slot, -(op.width / 2 - frame / 2), 0, frame, op.height, depth);
      add(op.frame_slot, op.width / 2 - frame / 2, 0, frame, op.height, depth);
      add(op.frame_slot, 0, op.height / 2 - frame / 2, op.width, frame, depth);
      add(op.frame_slot, 0, -(op.height / 2 - frame / 2), op.width, frame, depth);
      if (op.glazed) {
        add("glazing", 0, 0, op.width - 2 * frame, op.height - 2 * frame, 0.012);
        if (op.width > 1.4) add(op.frame_slot, 0, 0, 0.05, op.height - 2 * frame, depth * 0.8);
      } else if (op.kind === "sliding_door" || op.swing == null) {
        // A sliding door runs in the wall plane; it does not swing. One panel parked over
        // half the opening leaves the other half clear, as an open patio door does.
        // Matches the GLB exporter.
        const panel = (op.width - 2 * frame) / 2;
        add("door.leaf", -(op.width / 2 - frame) + panel / 2, 0, panel, op.height - frame, 0.042);
      } else {
        // Leaf swung open about its hinge jamb. Flat in the wall plane it fills the
        // opening, so the hole cut in the wall is invisible and you cannot see through
        // the doorway at all.
        const leafW = op.width - 2 * frame;
        // 24 degrees, matching the GLB exporter. Nearly perpendicular (78) stood every
        // leaf out into the middle of its room and the model read as a field of screens.
        const ang = (24 * Math.PI) / 180;
        const hingeX = -leafW / 2;
        const lx = hingeX + (leafW / 2) * Math.cos(ang);
        const lz = (leafW / 2) * Math.sin(ang) * (op.swing_inward === false ? -1 : 1);
        push("door.leaf", {
          c: [cx + lx * cos + lz * sin, cy, cz - lx * sin + lz * cos],
          s: [leafW, op.height - frame, 0.042],
          yaw: op.rotation_y + ang * (op.swing_inward === false ? -1 : 1),
        });
      }
    }

    // Boxes are built explicitly rather than by cloning BoxGeometry, whose UVs are
    // 0..1 per face. Textures scaled in real-world metres need UVs in metres, so
    // furniture and door leaves would otherwise get a single texture pixel stretched
    // over the whole panel while the walls beside them were correctly grained.
    const CORNERS: [number, number, number][] = [
      [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
      [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
    ];
    const TRIS: [number, number, number][] = [
      [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
      [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
      [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3],
    ];

    const out: { slot: string; geometry: THREE.BufferGeometry }[] = [];
    for (const [slot, boxes] of bySlot) {
      const pos: number[] = [];
      const uv: number[] = [];
      const idx: number[] = [];
      for (const b of boxes) {
        const base = pos.length / 3;
        const [sx, sy, sz] = b.s;
        const cos = Math.cos(b.yaw);
        const sin = Math.sin(b.yaw);
        for (const [lx, ly, lz] of CORNERS) {
          const px = lx * sx;
          const py = ly * sy;
          const pz = lz * sz;
          pos.push(b.c[0] + px * cos + pz * sin, b.c[1] + py, b.c[2] - px * sin + pz * cos);
          let u: number, v: number;
          if (sy <= sx && sy <= sz) { u = (lx + 0.5) * sx; v = (lz + 0.5) * sz; }
          else if (sx <= sy && sx <= sz) { u = (lz + 0.5) * sz; v = (ly + 0.5) * sy; }
          else { u = (lx + 0.5) * sx; v = (ly + 0.5) * sy; }
          uv.push(u, v);
        }
        for (const [a, c, d] of TRIS) idx.push(base + a, base + c, base + d);
      }
      if (!idx.length) continue;
      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
      const uvAttr = new THREE.Float32BufferAttribute(uv, 2);
      geom.setAttribute("uv", uvAttr);
      geom.setAttribute("uv1", uvAttr.clone());
      geom.setIndex(idx);
      geom.computeVertexNormals();
      geom.computeBoundingSphere();
      out.push({ slot, geometry: geom });
    }
    return out;
  }, [placements, catalog, openings, swaps, hidden, showFurniture, skipIds]);
}

/** Minimal geometry merge — avoids pulling in three's addons for one function. */
function mergeGeometries(geoms: THREE.BufferGeometry[]): THREE.BufferGeometry | null {
  if (!geoms.length) return null;
  let vCount = 0;
  let iCount = 0;
  for (const g of geoms) {
    vCount += g.attributes.position.count;
    iCount += g.index ? g.index.count : g.attributes.position.count;
  }
  const pos = new Float32Array(vCount * 3);
  const nor = new Float32Array(vCount * 3);
  const uv = new Float32Array(vCount * 2);
  const idx = new Uint32Array(iCount);
  let vo = 0;
  let io = 0;
  for (const g of geoms) {
    const p = g.attributes.position as THREE.BufferAttribute;
    const n = g.attributes.normal as THREE.BufferAttribute | undefined;
    const t = g.attributes.uv as THREE.BufferAttribute | undefined;
    pos.set(p.array as Float32Array, vo * 3);
    if (n) nor.set(n.array as Float32Array, vo * 3);
    if (t) uv.set(t.array as Float32Array, vo * 2);
    const gi = g.index;
    if (gi) {
      for (let i = 0; i < gi.count; i++) idx[io + i] = gi.getX(i) + vo;
      io += gi.count;
    } else {
      for (let i = 0; i < p.count; i++) idx[io + i] = i + vo;
      io += p.count;
    }
    vo += p.count;
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  out.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
  out.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  out.setIndex(new THREE.BufferAttribute(idx, 1));
  out.computeBoundingSphere();
  return out;
}

/**
 * Real downloaded meshes for the catalogue items that have one.
 *
 * The GLB export and the Cycles render already place these; without the same here
 * the browser and the render would disagree on five items, which is exactly the kind
 * of divergence that makes a configurator untrustworthy as a preview.
 *
 * Fitting repeats the exporter's rule deliberately: uniform scale so proportions
 * survive, longer horizontal axis aligned to the item's longer declared axis (glTF
 * authors disagree on facing), and dropped so the lowest point rests on the floor.
 * A model that fails to load leaves the parametric boxes in place rather than a hole.
 */
const _modelCache = new Map<string, Promise<THREE.Group>>();

function loadModel(url: string): Promise<THREE.Group> {
  const hit = _modelCache.get(url);
  if (hit) return hit;
  const p = new Promise<THREE.Group>((resolve, reject) => {
    new GLTFLoader().load(url, (g) => resolve(g.scene), undefined, reject);
  });
  _modelCache.set(url, p);
  return p;
}

function ExternalModels({
  placements,
  catalog,
  manifest,
  swaps,
  hidden,
  assetBase,
}: {
  placements: FurniturePlacement[];
  catalog: Record<string, CatalogItem>;
  manifest: Record<string, string>;
  swaps: Record<string, string>;
  hidden: Set<string>;
  assetBase: string;
}) {
  const [nodes, setNodes] = useState<THREE.Object3D[]>([]);

  useEffect(() => {
    let cancelled = false;
    const wanted = placements.filter((pl) => {
      if (hidden.has(pl.id)) return false;
      const item = catalog[swaps[pl.id] ?? pl.catalog_id];
      return !!(item?.external_model && manifest[item.external_model]);
    });

    Promise.all(
      wanted.map(async (pl) => {
        const item = catalog[swaps[pl.id] ?? pl.catalog_id]!;
        const src = await loadModel(`${assetBase}/data/${manifest[item.external_model!]}`);
        const obj = src.clone(true);

        const box = new THREE.Box3().setFromObject(obj);
        const size = new THREE.Vector3();
        box.getSize(size);
        if (size.x < 1e-6 || size.y < 1e-6 || size.z < 1e-6) return null;

        const wantW = item.width_mm / 1000;
        const wantD = item.depth_mm / 1000;
        const wantH = item.height_mm / 1000;
        const swap = size.x >= size.z !== wantW >= wantD;
        const modelW = swap ? size.z : size.x;
        const modelD = swap ? size.x : size.z;
        const scale = Math.min(wantW / modelW, wantD / modelD, wantH / size.y);

        const group = new THREE.Group();
        obj.scale.setScalar(scale);
        if (swap) obj.rotation.y = Math.PI / 2;
        obj.updateMatrixWorld(true);
        const scaled = new THREE.Box3().setFromObject(obj);
        const c = new THREE.Vector3();
        scaled.getCenter(c);
        obj.position.set(-c.x, -scaled.min.y, -c.z);
        group.add(obj);
        group.rotation.y = pl.rotation_y;
        group.position.set(...pl.position);
        group.traverse((o) => {
          if ((o as THREE.Mesh).isMesh) {
            o.castShadow = true;
            o.receiveShadow = true;
          }
        });
        return group;
      }),
    )
      .then((list) => {
        if (!cancelled) setNodes(list.filter(Boolean) as THREE.Object3D[]);
      })
      .catch(() => {
        if (!cancelled) setNodes([]);
      });

    return () => {
      cancelled = true;
    };
  }, [placements, catalog, manifest, swaps, hidden, assetBase]);

  return (
    <>
      {nodes.map((n, i) => (
        <primitive key={i} object={n} />
      ))}
    </>
  );
}

function SceneContents() {
  const scene = useStore((s) => s.scene)!;
  const schemes = useStore((s) => s.schemes);
  const schemeId = useStore((s) => s.schemeId);
  const overrides = useStore((s) => s.slotOverrides);
  const catalog = useStore((s) => s.catalog);
  const swaps = useStore((s) => s.furnitureSwaps);
  const hidden = useStore((s) => s.hiddenPlacements);
  const manifest = useStore((s) => s.textures);
  const assetBase = useStore((s) => s.assetBase);
  const showCeilings = useStore((s) => s.showCeilings);
  const showFurniture = useStore((s) => s.showFurniture);
  const selectedRoom = useStore((s) => s.selectedRoom);

  const arch = useArchGeometry(scene, showCeilings);
  const modelManifest = useStore((s) => s.models);
  // Placements drawn as real meshes must not also be drawn as boxes, or the two
  // overlap and z-fight.
  const externalIds = useMemo(() => {
    const out = new Set<string>();
    for (const pl of scene.furniture) {
      const item = catalog[swaps[pl.id] ?? pl.catalog_id];
      if (item?.external_model && modelManifest[item.external_model]) out.add(pl.id);
    }
    return out;
  }, [scene.furniture, catalog, swaps, modelManifest]);
  const boxes = useBoxGeometry(
    scene.furniture, catalog, scene.openings, swaps, hidden, showFurniture, externalIds,
  );

  // Materials are rebuilt only when the palette actually changes — cheap, and it
  // keeps geometry untouched across a scheme swap.
  const materials = useMemo(() => {
    const map = new Map<string, THREE.Material>();
    const slots = new Set<string>([...arch.map((a) => a.slot), ...boxes.map((b) => b.slot)]);
    for (const slot of slots) {
      map.set(
        slot,
        makeMaterial(resolveMaterial(slot, schemes, schemeId, overrides), slot, manifest, assetBase),
      );
    }
    return map;
  }, [arch, boxes, schemes, schemeId, overrides, manifest, assetBase]);

  useEffect(() => () => materials.forEach((m) => m.dispose()), [materials]);

  const scheme = schemes[schemeId];
  const sunColour = scheme ? kelvinToRgb(scheme.sun_temperature_k) : [1, 1, 1];
  const fillLights = scene.lights.filter((l) => l.kind === "area" && l.room_id);

  return (
    <group>
      {/* Architecture */}
      {arch.map(({ slot, geometry }) => (
        <mesh
          key={`arch-${slot}`}
          geometry={geometry}
          material={materials.get(slot)}
          castShadow
          receiveShadow
        />
      ))}

      {/* Furniture, joinery, doors, glazing */}
      {boxes.map(({ slot, geometry }) => (
        <mesh
          key={`box-${slot}`}
          geometry={geometry}
          material={materials.get(slot)}
          castShadow
          receiveShadow
        />
      ))}

      {showFurniture && (
        <ExternalModels
          placements={scene.furniture}
          catalog={catalog}
          manifest={modelManifest}
          swaps={swaps}
          hidden={hidden}
          assetBase={assetBase}
        />
      )}

      {/* Room highlight: a translucent slab over the selected room's floor. */}
      {selectedRoom &&
        (() => {
          const rz = scene.rooms.find((r) => r.id === selectedRoom);
          if (!rz || rz.polygon.length < 3) return null;
          const shape = new THREE.Shape(rz.polygon.map(([x, z]) => new THREE.Vector2(x, z)));
          const geom = new THREE.ShapeGeometry(shape);
          geom.rotateX(Math.PI / 2);
          geom.translate(0, rz.floor_level + 0.012, 0);
          return (
            <mesh geometry={geom}>
              <meshBasicMaterial
                color={new THREE.Color(0.25, 0.62, 0.95)}
                transparent
                opacity={0.22}
                side={THREE.DoubleSide}
                depthWrite={false}
              />
            </mesh>
          );
        })()}

      {/* Sun: the only shadow caster, so shadows stay crisp and cheap. */}
      <directionalLight
        position={[14, 22, 10]}
        intensity={(scheme?.sun_intensity ?? 3) * 0.55}
        color={new THREE.Color(...(sunColour as [number, number, number]))}
        castShadow
        shadow-mapSize={[4096, 4096]}
        shadow-radius={4}
        shadow-camera-left={-22}
        shadow-camera-right={22}
        shadow-camera-top={22}
        shadow-camera-bottom={-22}
        shadow-bias={-0.0008}
      />
      {/* Per-room fill. A rasteriser cannot compute the indirect light a path
          tracer gets for free, so these stand in for the bounce. */}
      {fillLights.map((l) => (
        <pointLight
          key={l.id}
          position={l.position}
          intensity={l.intensity * 0.5}
          distance={7}
          decay={2}
          color={new THREE.Color(...kelvinToRgb(l.colour_k))}
        />
      ))}
      <ambientLight intensity={(scheme?.ambient_intensity ?? 0.35) * 0.8} />
    </group>
  );
}

function CameraRig() {
  const scene = useStore((s) => s.scene);
  const activeCamera = useStore((s) => s.activeCamera);
  const { camera } = useThree();
  const controls = useRef<any>(null);

  useEffect(() => {
    if (!scene) return;
    const spec = scene.cameras.find((c) => c.id === activeCamera);
    if (spec) {
      camera.position.set(...spec.position);
      if (controls.current) {
        controls.current.target.set(...spec.target);
        controls.current.update();
      }
      if ((camera as THREE.PerspectiveCamera).isPerspectiveCamera) {
        // 36 mm full-frame sensor, matching the Blender camera, so a view framed
        // in the browser matches the hero render.
        const cam = camera as THREE.PerspectiveCamera;
        cam.fov = 2 * Math.atan(18 / spec.focal_mm) * (180 / Math.PI);
        cam.updateProjectionMatrix();
      }
      return;
    }
    const [minx, miny, minz, maxx, maxy, maxz] = scene.bounds;
    const cx = (minx + maxx) / 2;
    const cz = (minz + maxz) / 2;
    const span = Math.max(maxx - minx, maxz - minz, 4);
    camera.position.set(cx + span * 0.75, maxy + span * 0.7, cz + span * 0.75);
    if (controls.current) {
      controls.current.target.set(cx, 0.8, cz);
      controls.current.update();
    }
  }, [scene, activeCamera, camera]);

  return <OrbitControls ref={controls} makeDefault enableDamping dampingFactor={0.08} maxPolarAngle={Math.PI / 2.02} />;
}

/**
 * Error boundary around the 3D subtree.
 *
 * Without this, a throw anywhere inside <Canvas> unmounts the whole subtree and
 * leaves a blank viewport with *nothing* in the console — which is exactly how
 * this went wrong during development. A visible message beats silent nothing.
 */
class GlBoundary extends Component<{ children: ReactNode }, { err: Error | null }> {
  state = { err: null as Error | null };
  static getDerivedStateFromError(err: Error) {
    return { err };
  }
  componentDidCatch(err: Error, info: unknown) {
    console.error("[atlas] 3D subtree failed:", err, info);
  }
  render() {
    if (this.state.err) {
      return (
        <mesh>
          <boxGeometry args={[1, 1, 1]} />
          <meshBasicMaterial color="#ff00cc" />
        </mesh>
      );
    }
    return this.props.children;
  }
}

export function Viewer() {
  const scene = useStore((s) => s.scene);
  const exposure = useStore((s) => s.exposure);
  if (!scene) return null;
  return (
    <Canvas
      shadows
      dpr={[1, 2]}
      gl={{ antialias: true, preserveDrawingBuffer: true }}
      camera={{ fov: 50, near: 0.02, far: 400 }}
      resize={{ polyfill: PollingResizeObserver as unknown as typeof ResizeObserver, debounce: 0 }}
      onCreated={(state) => {
        // AgX, not ACESFilmic: it is the transform the Cycles art direction uses,
        // so the viewer and offline renders agree about colour — and its highlight
        // rolloff keeps sunlit floors from clipping to white.
        state.gl.toneMapping = THREE.AgXToneMapping;
        state.gl.outputColorSpace = THREE.SRGBColorSpace;
        state.gl.shadowMap.type = THREE.PCFSoftShadowMap;
      }}
    >
      <ResizeGuard />
      <ToneMapping exposure={exposure} />
      <GlBoundary>
      <color attach="background" args={["#e9edf2"]} />
      <fog attach="fog" args={["#e9edf2", 40, 130]} />
      {/* Image-based lighting, generated locally.
          drei's <Environment preset> was the obvious choice and is the wrong one:
          it fetches an HDRI from a CDN, and while that request is in flight the
          component suspends. With no boundary above it that suspended the entire
          canvas, so the app rendered a blank grey rectangle with no error in the
          console — the hardest kind of failure to diagnose. three ships
          RoomEnvironment, which is generated in-process, needs no network, and
          gives metal and glass something to reflect. */}
      <LocalEnvironment intensity={0.95} />
      <Suspense fallback={null}>
        <SceneContents />
      </Suspense>
      {/* Ambient occlusion is the single biggest realism lever a rasteriser has:
          the darkening where a sofa meets the floor or two walls meet is what a
          path tracer computes as indirect shadowing, and its absence is what makes
          real-time interiors read as "floaty". Half-resolution N8AO keeps it cheap
          enough for an integrated GPU. */}
      <EffectComposer enableNormalPass={false}>
        <N8AO
          halfRes
          aoRadius={0.6}
          intensity={2.2}
          distanceFalloff={0.75}
          quality="performance"
        />
      </EffectComposer>
      </GlBoundary>
      <CameraRig />
    </Canvas>
  );
}

/**
 * Keep the drawing buffer matched to the canvas's real CSS size.
 *
 * R3F sizes the canvas from a ResizeObserver, and here that observer reported zero
 * on first paint and never fired again, so `gl.setSize` was never called and the
 * viewport stayed at the default 300x150 while the container was 1140x861. The
 * result was a blank grey panel with correct geometry, correct materials and no
 * error anywhere — so this observes the canvas directly and sets the size itself.
 */
function ResizeGuard() {
  const { gl, camera, size } = useThree();
  useEffect(() => {
    const canvas = gl.domElement;
    const parent = canvas.parentElement;
    if (!parent) return;
    const apply = () => {
      const w = parent.clientWidth || canvas.clientWidth;
      const h = parent.clientHeight || canvas.clientHeight;
      if (w < 2 || h < 2) return;
      // updateStyle=false: the CSS rule already stretches the element, and letting
      // three write inline pixel sizes would fight it on the next resize.
      gl.setSize(w, h, false);
      gl.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      const cam = camera as THREE.PerspectiveCamera;
      if (cam.isPerspectiveCamera) {
        cam.aspect = w / h;
        cam.updateProjectionMatrix();
      }
    };
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(parent);
    window.addEventListener("resize", apply);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", apply);
    };
  }, [gl, camera, size]);
  return null;
}

/** Generate an environment map in-process, with no external asset fetch. */
function LocalEnvironment({ intensity = 1 }: { intensity?: number }) {
  const { gl, scene } = useThree();
  useEffect(() => {
    const pmrem = new THREE.PMREMGenerator(gl);
    pmrem.compileEquirectangularShader();
    const env = new RoomEnvironment();
    const target = pmrem.fromScene(env, 0.04);
    scene.environment = target.texture;
    scene.environmentIntensity = intensity;
    return () => {
      target.dispose();
      pmrem.dispose();
      scene.environment = null;
    };
  }, [gl, scene, intensity]);
  return null;
}

function ToneMapping({ exposure }: { exposure: number }) {
  const { gl, scene, camera } = useThree();
  useEffect(() => {
    gl.toneMappingExposure = exposure;
  }, [gl, exposure]);
  useEffect(() => {
    // Dev handle. Debugging a renderer from the outside is otherwise guesswork:
    // geometry, materials and camera all look fine in isolation and the symptom
    // is simply "nothing is drawn".
    (window as any).__atlas = { gl, scene, camera, THREE };
  }, [gl, scene, camera]);
  return null;
}
