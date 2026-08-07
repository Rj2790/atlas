/**
 * TypeScript mirrors of the Python SceneIR types.
 *
 * These are *only* type declarations — the actual scheme and catalogue data is
 * fetched from JSON that the Python pipeline writes (`atlas export-data`). That
 * split matters: types can safely be duplicated because a mismatch is a compile
 * error, whereas duplicated *data* silently drifts, and a configurator offering a
 * material the renderer does not have is worse than one offering fewer.
 */

export type Vec3 = [number, number, number];
export type Vec2 = [number, number];

export interface Material {
  id: string;
  name: string;
  base_color: Vec3;
  roughness: number;
  metallic: number;
  opacity: number;
  sheen: number;
  texture_scale_m: number;
  acg_asset: string | null;
}

/** asset id -> role -> filename under /data/textures/. Written by `atlas build`. */
export type TextureManifest = Record<string, Record<string, string>>;

export interface Scheme {
  id: string;
  name: string;
  description: string;
  /** Material slot name -> material id. */
  slots: Record<string, string>;
  materials: Record<string, Material>;
  accent_colour: Vec3;
  ambient_intensity: number;
  sun_intensity: number;
  sun_temperature_k: number;
  interior_temperature_k: number;
}

export interface Surface {
  id: string;
  kind: string;
  slot: string;
  vertices: number[];
  indices: number[];
  normals: number[];
  uvs: number[];
  room_id: string | null;
  wall_id: string | null;
}

export interface OpeningSpec {
  id: string;
  kind: string;
  wall_id: string;
  centre: Vec3;
  width: number;
  height: number;
  sill: number;
  rotation_y: number;
  thickness: number;
  glazed: boolean;
  leaf_slot: string;
  frame_slot: string;
  /** null on a sliding door — it runs in the wall plane rather than swinging. */
  swing: string | null;
  swing_inward: boolean;
}

export interface FurniturePlacement {
  id: string;
  catalog_id: string;
  room_id: string;
  position: Vec3;
  rotation_y: number;
  /** Plan dimensions only; the Y component is always 1. The solver may place an item
   *  smaller than its catalogue size when the room is tight, and a work surface's height
   *  is set by the human body rather than by the spare floor. */
  scale: Vec3;
  footprint: Vec2;
  height: number;
  material_overrides: Record<string, string>;
  anchored_to_wall: string | null;
  locked: boolean;
  source: string;
  reason: string;
}

export interface LightSpec {
  id: string;
  kind: "sun" | "area" | "point" | "spot" | "cove";
  position: Vec3;
  direction: Vec3;
  intensity: number;
  colour_k: number;
  size: Vec2;
  room_id: string | null;
  cast_shadow: boolean;
  realtime_only: boolean;
}

export interface CameraSpec {
  id: string;
  room_id: string;
  name: string;
  position: Vec3;
  target: Vec3;
  focal_mm: number;
}

export interface RoomZone {
  id: string;
  name: string;
  kind: string;
  centroid: Vec3;
  area_m2: number;
  floor_slot: string;
  wall_slot: string;
  ceiling_slot: string;
  accent_wall_id: string | null;
  polygon: Vec2[];
  ceiling_height: number;
  floor_level: number;
  stated_dimension_m: Vec2 | null;
}

export interface SceneIR {
  schema_version: string;
  name: string;
  source_plan: string;
  origin_offset: Vec2;
  surfaces: Surface[];
  openings: OpeningSpec[];
  furniture: FurniturePlacement[];
  lights: LightSpec[];
  cameras: CameraSpec[];
  rooms: RoomZone[];
  scheme_id: string;
  available_schemes: string[];
  /** Art-director slot overrides. Keys may be `slot` or `slot@wallId`. */
  slot_overrides: Record<string, string>;
  art_direction: {
    scheme_id?: string;
    rationale?: string;
    mood?: string;
    hero_room_id?: string;
    source?: string;
    exposure?: number;
  };
  bounds: [number, number, number, number, number, number];
  imperial_display: boolean;
  notes: string[];
}

export interface CatalogPart {
  x: number;
  y: number;
  z: number;
  w: number;
  h: number;
  d: number;
  slot: string;
}

export interface CatalogItem {
  id: string;
  name: string;
  category: string;
  width_mm: number;
  depth_mm: number;
  height_mm: number;
  anchor: string;
  rooms: string[];
  priority: number;
  min_room_area_m2: number;
  tags: string[];
  notes: string;
  parts: CatalogPart[];
  external_model: string | null;
}

/** Poly Haven slug -> glTF path under /data/. Written by `atlas build`. */
export type ModelManifest = Record<string, string>;

export type Catalog = Record<string, CatalogItem>;
export type Schemes = Record<string, Scheme>;
