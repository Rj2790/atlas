/**
 * Configurator state.
 *
 * The whole point of the SceneIR design is that a visual change is a *data* change,
 * never a geometry rebuild. This store is where that pays off:
 *
 *  - changing the scheme swaps one id;
 *  - overriding a floor or wall finish writes one entry into `slotOverrides`;
 *  - swapping a sofa rewrites one `catalog_id`.
 *
 * Nothing here touches vertices. The renderer reads geometry once and resolves
 * materials per frame from `resolveMaterial`, so every swap is instant.
 */

import { create } from "zustand";
import type { Catalog, Material, ModelManifest, SceneIR, Schemes, TextureManifest } from "./types";

export type ViewMode = "orbit" | "walk" | "plan";

interface State {
  scene: SceneIR | null;
  schemes: Schemes;
  catalog: Catalog;
  textures: TextureManifest;
  models: ModelManifest;
  /** Prefix for texture/model URLs — the API host when a job is loaded, else same-origin. */
  assetBase: string;
  loading: boolean;
  error: string | null;

  schemeId: string;
  /** Per-slot material overrides, applied on top of the active scheme. */
  slotOverrides: Record<string, string>;
  /** Placement id -> catalogue id, for individual furniture swaps. */
  furnitureSwaps: Record<string, string>;
  hiddenPlacements: Set<string>;

  selectedRoom: string | null;
  hoveredRoom: string | null;
  viewMode: ViewMode;
  showCeilings: boolean;
  showFurniture: boolean;
  activeCamera: string | null;
  exposure: number;

  jobId: string | null;
  /** Which dwelling of the job is on screen. Null before a job, or for a raster plan
   *  where there is only ever one. */
  unit: number | null;
  showUnit: (unit: number) => Promise<void>;
  /** The build this workspace is showing.
   *
   * Held in the store rather than in React `useState` deliberately: the job is
   * application state that the scene, the compare panes and the warnings panel all
   * read, so one source of truth is simpler. It also subscribes through
   * `useSyncExternalStore`, which updates reliably even when the page is not being
   * painted — a React `useState` set from a promise can sit unflushed in a
   * backgrounded or prerendered tab, leaving the upload screen on screen after the
   * build has finished. */
  job: Job | null;
  openJob: (job: Job | null) => void;
  load: (jobId?: string | null, unit?: number | null) => Promise<void>;
  setScheme: (id: string) => void;
  overrideSlot: (slot: string, materialId: string | null) => void;
  swapFurniture: (placementId: string, catalogId: string) => void;
  togglePlacement: (placementId: string) => void;
  selectRoom: (id: string | null) => void;
  hoverRoom: (id: string | null) => void;
  setViewMode: (m: ViewMode) => void;
  setShowCeilings: (v: boolean) => void;
  setShowFurniture: (v: boolean) => void;
  setActiveCamera: (id: string | null) => void;
  setExposure: (v: number) => void;
  resetOverrides: () => void;
}

import { API, jobUrl, unitUrl, type Job } from "./api";

/**
 * Where scene data comes from.
 *
 * Two sources on purpose. A `jobId` loads the result of an upload from the API — the
 * platform path. Without one it falls back to the static `/data` directory the CLI
 * writes, so the configurator still runs standalone with no server. The material and
 * model caches are always served by the API when a job is loaded, because they are
 * ~250 MB and shared across every build rather than copied per job.
 */
const DATA = "/data";

export const useStore = create<State>((set, get) => ({
  scene: null,
  schemes: {},
  catalog: {},
  textures: {},
  models: {},
  assetBase: "",
  loading: true,
  error: null,

  schemeId: "warm-minimal",
  slotOverrides: {},
  furnitureSwaps: {},
  hiddenPlacements: new Set<string>(),

  selectedRoom: null,
  hoveredRoom: null,
  viewMode: "orbit",
  showCeilings: false,
  showFurniture: true,
  activeCamera: null,
  exposure: 1.0,
  jobId: null,
  unit: null,
  job: null,

  showUnit: async (unit: number) => {
    const { jobId } = get();
    if (!jobId) return;
    set({ unit });
    await get().load(jobId, unit);
  },

  load: async (jobId: string | null = null, unit: number | null = null) => {
    set({ loading: true, error: null, jobId, unit });
    // A job with several dwellings serves each one's scene separately; the job-level
    // scene.json is the first-built dwelling, kept so a link without a unit still works.
    const sceneUrl = jobId
      ? unit != null
        ? unitUrl(jobId, unit, "scene.json")
        : jobUrl(jobId, "scene.json")
      : `${DATA}/scene.json`;
    const assetBase = jobId ? API : "";
    try {
      const [scene, schemes, catalog, textures, models] = await Promise.all([
        fetch(sceneUrl).then((r) => {
          if (!r.ok) throw new Error(`scene.json ${r.status}`);
          return r.json() as Promise<SceneIR>;
        }),
        fetch(`${DATA}/schemes.json`).then((r) => {
          if (!r.ok) throw new Error(`schemes.json ${r.status}`);
          return r.json() as Promise<Schemes>;
        }),
        fetch(`${DATA}/catalog.json`).then((r) => {
          if (!r.ok) throw new Error(`catalog.json ${r.status}`);
          return r.json() as Promise<Catalog>;
        }),
        // Optional: a build run with --no-textures has no manifest, and the scene
        // must still load and render from scalar material values alone.
        // Texture and model manifests are derived from the shared cache listing when a
        // job is loaded; the CLI writes them as files for the standalone path.
        fetch(`${DATA}/textures.json`)
          .then((r) => (r.ok ? (r.json() as Promise<TextureManifest>) : {}))
          .catch(() => ({}) as TextureManifest),
        // Also optional: --no-models produces no manifest and the viewer falls back
        // to the catalogue's parametric parts.
        fetch(`${DATA}/models.json`)
          .then((r) => (r.ok ? (r.json() as Promise<ModelManifest>) : {}))
          .catch(() => ({}) as ModelManifest),
      ]);
      set({
        scene,
        schemes,
        catalog,
        textures,
        models,
        assetBase,
        schemeId: schemes[scene.scheme_id] ? scene.scheme_id : Object.keys(schemes)[0],
        // Start from the art director's decisions so the browser preview matches the
        // GLB and the Cycles render. Per-wall keys (`slot@wallId`) are dropped: the
        // viewer resolves materials by slot, and a key it cannot apply would show in
        // the UI as a change that never happened.
        slotOverrides: Object.fromEntries(
          Object.entries(scene.slot_overrides ?? {}).filter(([k]) => !k.includes("@")),
        ),
        loading: false,
        error: null,
      });
    } catch (e) {
      set({
        loading: false,
        error:
          `Could not load scene data: ${(e as Error).message}. ` +
          `Run: python -m atlas build <plan> --export-web apps/web/public/data`,
      });
    }
  },

  openJob: (job) => {
    const url = new URL(location.href);
    if (job) url.searchParams.set("job", job.id);
    else url.searchParams.delete("job");
    history.replaceState(null, "", url);
    set({ job, scene: job ? null : get().scene, loading: !!job });
    if (job) void get().load(job.id);
  },

  setScheme: (id) =>
    // Overrides are cleared on a scheme change on purpose. A material id from the
    // old scheme usually does not exist in the new one, and keeping a dangling
    // override would silently fall back to a default while the UI still showed
    // the user's choice.
    set({ schemeId: id, slotOverrides: {} }),

  overrideSlot: (slot, materialId) =>
    set((s) => {
      const next = { ...s.slotOverrides };
      if (materialId === null) delete next[slot];
      else next[slot] = materialId;
      return { slotOverrides: next };
    }),

  swapFurniture: (placementId, catalogId) =>
    set((s) => ({ furnitureSwaps: { ...s.furnitureSwaps, [placementId]: catalogId } })),

  togglePlacement: (placementId: string) =>
    set((s) => {
      const next = new Set(s.hiddenPlacements);
      if (next.has(placementId)) next.delete(placementId);
      else next.add(placementId);
      return { hiddenPlacements: next };
    }),

  selectRoom: (id) => set({ selectedRoom: id }),
  hoverRoom: (id) => set({ hoveredRoom: id }),
  setViewMode: (m) => set({ viewMode: m }),
  setShowCeilings: (v) => set({ showCeilings: v }),
  setShowFurniture: (v) => set({ showFurniture: v }),
  setActiveCamera: (id) => set({ activeCamera: id }),
  setExposure: (v) => set({ exposure: v }),
  resetOverrides: () => set({ slotOverrides: {}, furnitureSwaps: {}, hiddenPlacements: new Set() }),
}));

/** Resolve a material slot through the active scheme and any override. */
export function resolveMaterial(
  slot: string,
  schemes: Schemes,
  schemeId: string,
  overrides: Record<string, string>,
): Material | null {
  const scheme = schemes[schemeId];
  if (!scheme) return null;
  const id = overrides[slot] ?? scheme.slots[slot];
  return (id && scheme.materials[id]) || null;
}

/** Kelvin -> linear RGB, matched to the Blender-side conversion. */
export function kelvinToRgb(k: number): [number, number, number] {
  const t = k / 100;
  let r: number, g: number, b: number;
  if (t <= 66) {
    r = 1;
    g = 0.3900815787690196 * Math.log(Math.max(t, 1e-3)) - 0.6318414437886275;
  } else {
    r = 1.292936186062745 * Math.pow(t - 60, -0.1332047592);
    g = 1.1298908608952941 * Math.pow(t - 60, -0.0755148492);
  }
  if (t >= 66) b = 1;
  else if (t <= 19) b = 0;
  else b = 0.5432067891101961 * Math.log(t - 10) - 1.19625408914;
  const c = (v: number) => Math.min(1, Math.max(0, v));
  return [c(r), c(g), c(b)];
}
