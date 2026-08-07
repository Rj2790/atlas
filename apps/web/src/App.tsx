/**
 * Configurator shell.
 *
 * The panel is organised the way a designer works, not the way the data is
 * structured: pick a look, then adjust the finishes, then the furniture. The
 * material list is generated from the active scheme rather than hard-coded, so
 * adding a scheme in Python adds it here with no front-end change.
 */

import { useEffect, useMemo, useState } from "react";
import Compare from "./Compare";
import Upload from "./Upload";
import { resolveMaterial, useStore } from "./store";
import { getJob, unitUrl, type Job } from "./api";
import type { Material } from "./types";
import "./styles.css";

/** Slots the UI exposes, grouped. Not every slot is worth a control — nobody
 *  wants to restyle a window reveal independently — so this is a curated subset. */
const SLOT_GROUPS: { label: string; slots: { slot: string; label: string }[] }[] = [
  {
    label: "Flooring",
    slots: [
      { slot: "floor.living", label: "Living / dining" },
      { slot: "floor.bedroom", label: "Bedrooms" },
      { slot: "floor.kitchen", label: "Kitchen" },
      { slot: "floor.wet", label: "Bathrooms" },
      { slot: "floor.circulation", label: "Circulation" },
      { slot: "floor.outdoor", label: "Balconies" },
    ],
  },
  {
    label: "Walls & ceiling",
    slots: [
      { slot: "wall.interior", label: "Walls" },
      { slot: "wall.accent", label: "Accent wall" },
      { slot: "wall.wet", label: "Bathroom walls" },
      { slot: "ceiling", label: "Ceiling" },
      { slot: "skirting", label: "Skirting" },
    ],
  },
  {
    label: "Joinery & fittings",
    slots: [
      { slot: "joinery", label: "Wardrobes / carcasses" },
      { slot: "counter", label: "Counters" },
      { slot: "door.leaf", label: "Doors" },
      { slot: "window.frame", label: "Window frames" },
      { slot: "metal", label: "Metalwork" },
      { slot: "fabric", label: "Upholstery" },
    ],
  },
];

function Swatch({ m, active, onClick }: { m: Material; active: boolean; onClick: () => void }) {
  const [r, g, b] = m.base_color;
  const css = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
  return (
    <button
      className={`swatch${active ? " active" : ""}`}
      style={{ background: css }}
      onClick={onClick}
      title={`${m.name} · roughness ${m.roughness.toFixed(2)}${
        m.metallic > 0.5 ? " · metal" : ""
      } · tiles at ${m.texture_scale_m} m`}
      aria-label={m.name}
    />
  );
}

function SlotControl({ slot, label }: { slot: string; label: string }) {
  const schemes = useStore((s) => s.schemes);
  const schemeId = useStore((s) => s.schemeId);
  const overrides = useStore((s) => s.slotOverrides);
  const overrideSlot = useStore((s) => s.overrideSlot);

  const scheme = schemes[schemeId];
  if (!scheme) return null;
  const current = resolveMaterial(slot, schemes, schemeId, overrides);
  const options = useMemo(() => Object.values(scheme.materials), [scheme]);

  return (
    <div className="slot">
      <div className="slot-head">
        <span className="slot-label">{label}</span>
        <span className="slot-value">{current?.name ?? "—"}</span>
      </div>
      <div className="swatches">
        {options.map((m) => (
          <Swatch
            key={m.id}
            m={m}
            active={current?.id === m.id}
            onClick={() => overrideSlot(slot, m.id === scheme.slots[slot] ? null : m.id)}
          />
        ))}
      </div>
    </div>
  );
}

function RoomList() {
  const scene = useStore((s) => s.scene);
  const selected = useStore((s) => s.selectedRoom);
  const selectRoom = useStore((s) => s.selectRoom);
  const setActiveCamera = useStore((s) => s.setActiveCamera);
  const imperial = scene?.imperial_display ?? false;
  if (!scene) return null;

  const toSqft = (m2: number) => m2 * 10.7639;
  const rooms = [...scene.rooms].sort((a, b) => b.area_m2 - a.area_m2);

  return (
    <div className="rooms">
      {rooms.map((r) => {
        const cam = scene.cameras.find((c) => c.room_id === r.id);
        const stated = r.stated_dimension_m;
        return (
          <button
            key={r.id}
            className={`room${selected === r.id ? " active" : ""}`}
            onClick={() => {
              selectRoom(selected === r.id ? null : r.id);
              if (cam) setActiveCamera(cam.id);
            }}
          >
            <span className="room-name">{r.name || r.kind}</span>
            <span className="room-meta">
              {imperial ? `${toSqft(r.area_m2).toFixed(0)} sq ft` : `${r.area_m2.toFixed(1)} m²`}
              {stated ? (
                <em title="printed on the drawing">
                  {" · drawn "}
                  {imperial
                    ? `${ftin(stated[0])}×${ftin(stated[1])}`
                    : `${stated[0].toFixed(2)}×${stated[1].toFixed(2)} m`}
                </em>
              ) : null}
              {cam ? <span className="cam-dot" title="has a framed view">◉</span> : null}
            </span>
          </button>
        );
      })}
    </div>
  );
}

function ftin(metres: number): string {
  const inches = metres / 0.0254;
  let ft = Math.floor(inches / 12);
  let inc = Math.round(inches - ft * 12);
  if (inc === 12) {
    ft += 1;
    inc = 0;
  }
  return `${ft}'${inc}"`;
}

function FurniturePanel() {
  const scene = useStore((s) => s.scene);
  const catalog = useStore((s) => s.catalog);
  const swaps = useStore((s) => s.furnitureSwaps);
  const hidden = useStore((s) => s.hiddenPlacements);
  const swapFurniture = useStore((s) => s.swapFurniture);
  const togglePlacement = useStore((s) => s.togglePlacement);
  const selectedRoom = useStore((s) => s.selectedRoom);
  if (!scene) return null;

  const items = scene.furniture.filter((f) => !selectedRoom || f.room_id === selectedRoom);
  if (!items.length) {
    return (
      <p className="hint">
        {selectedRoom
          ? "No furniture was placed in this room."
          : "No furniture in this scene."}
      </p>
    );
  }

  return (
    <div className="furniture">
      {items.map((f) => {
        const activeId = swaps[f.id] ?? f.catalog_id;
        const item = catalog[activeId];
        // Alternatives are same-category items only: offering a bed where a basin
        // belongs is not a swap, it is a mistake.
        const alts = Object.values(catalog).filter(
          (c) => item && c.category === item.category,
        );
        const isHidden = hidden.has(f.id);
        return (
          <div key={f.id} className={`fitem${isHidden ? " hidden" : ""}`}>
            <div className="fitem-head">
              <strong>{item?.name ?? activeId}</strong>
              <button className="mini" onClick={() => togglePlacement(f.id)}>
                {isHidden ? "show" : "hide"}
              </button>
            </div>
            <div className="fitem-meta">
              {item
                ? `${(item.width_mm / 1000).toFixed(2)} × ${(item.depth_mm / 1000).toFixed(
                    2,
                  )} × ${(item.height_mm / 1000).toFixed(2)} m`
                : ""}
              {f.locked ? <span className="tag locked">from drawing</span> : null}
            </div>
            <div className="fitem-reason">{f.reason}</div>
            {alts.length > 1 && (
              <select value={activeId} onChange={(e) => swapFurniture(f.id, e.target.value)}>
                {alts.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name}
                  </option>
                ))}
              </select>
            )}
          </div>
        );
      })}
    </div>
  );
}

/** Switch between the dwellings a drawing contains.
 *
 * A production CAD file is a whole floor — the demo file holds sixteen apartments — and
 * asking which one you want *before* showing any of them is the wrong way round: you
 * cannot know until you have looked. Every dwelling is built up front, so this only swaps
 * which scene is loaded.
 *
 * Each option carries its measured floor area, how much of the declared carpet polygon is
 * accounted for (floor + wall structure — the like-for-like figure, since carpet area
 * includes internal walls), and the number of geometric invariants it violates. That is
 * deliberate: some dwellings extract far better than others, and a unit accounting for
 * 35% of its declared polygon should look wrong in the menu rather than look like the
 * product. Bare floor over declared is *not* shown as a percentage — it reads 77-90% by
 * definition and every reader takes it for a defect.
 */
function UnitPicker({ job }: { job: Job }) {
  const unit = useStore((s) => s.unit);
  const showUnit = useStore((s) => s.showUnit);
  const current = unit ?? job.built[0]?.unit ?? null;

  return (
    <label className="units">
      <span>dwelling</span>
      <select
        value={current ?? ""}
        onChange={(e) => showUnit(Number(e.target.value))}
      >
        {job.built.map((b) => {
          const accounted =
            b.accounted != null ? Math.round(b.accounted * 100) : null;
          return (
            <option key={b.unit} value={b.unit}>
              {b.unit}. {b.name}
              {b.measured_sqft != null ? ` — ${Math.round(b.measured_sqft)} sf floor` : ""}
              {b.declared_sqft != null ? ` / ${Math.round(b.declared_sqft)} sf declared` : ""}
              {accounted != null ? ` · ${accounted}% accounted` : ""}
              {b.rooms != null ? ` · ${b.rooms} rooms` : ""}
              {b.issues ? ` · ${b.issues} issue${b.issues > 1 ? "s" : ""}` : ""}
            </option>
          );
        })}
      </select>
    </label>
  );
}


export default function App() {
  const job = useStore((s) => s.job);
  const openJob = useStore((s) => s.openJob);

  // `?job=<id>` opens an existing build directly — a build is a URL you can send
  // someone, and it avoids re-uploading to look at a result again.
  useEffect(() => {
    const id = new URLSearchParams(location.search).get("job");
    if (id && !job) {
      void getJob(id)
        .then(openJob)
        .catch(() => undefined);
    }
  }, [job, openJob]);

  if (!job) return <Upload onReady={openJob} />;
  return <Workspace job={job} onNew={() => openJob(null)} />;
}

/** Downloads for the current dwelling: GLB for 3D tools, IFC for BIM tools.
 *
 * Both are exported lazily by the API on first request, so the first click can
 * take a few seconds; after that they are served from disk.
 */
function DownloadMenu({ job }: { job: Job }) {
  const unit = useStore((s) => s.unit);
  const current = unit ?? job.built[0]?.unit ?? 1;
  return (
    <>
      <a className="dl" href={unitUrl(job.id, current, "model.glb")} download>
        GLB
      </a>
      <a className="dl" href={unitUrl(job.id, current, "model.ifc")} download>
        IFC
      </a>
    </>
  );
}

function Workspace({ job, onNew }: { job: Job; onNew: () => void }) {
  const loading = useStore((s) => s.loading);
  const error = useStore((s) => s.error);
  const scene = useStore((s) => s.scene);
  const schemes = useStore((s) => s.schemes);
  const schemeId = useStore((s) => s.schemeId);
  const setScheme = useStore((s) => s.setScheme);
  const showCeilings = useStore((s) => s.showCeilings);
  const setShowCeilings = useStore((s) => s.setShowCeilings);
  const showFurniture = useStore((s) => s.showFurniture);
  const setShowFurniture = useStore((s) => s.setShowFurniture);
  const setActiveCamera = useStore((s) => s.setActiveCamera);
  const activeCamera = useStore((s) => s.activeCamera);
  const exposure = useStore((s) => s.exposure);
  const setExposure = useStore((s) => s.setExposure);
  const resetOverrides = useStore((s) => s.resetOverrides);
  const overrides = useStore((s) => s.slotOverrides);
  const [panel, setPanel] = useState(true);

  if (loading) return <div className="boot">Loading scene…</div>;
  if (error) return <div className="boot error">{error}</div>;
  if (!scene) return null;

  const totalSqft = scene.rooms.reduce((a, r) => a + r.area_m2, 0) * 10.7639;

  return (
    <div className={`app${panel ? "" : " nopanel"}`}>
      <div className="stage">
        {/* Header first, then the compare split, then the footer controls. The stage is
            a flex column, so DOM order is layout order — with Compare first the header
            landed on top of the per-pane toolbars. */}
        <div className="overlay-top">
          <div className="brand">
            <strong>Atlas</strong>
            <span>{job.source_name}</span>
          </div>
          {job.built.length > 1 && <UnitPicker job={job} />}
          <div className="views">
            <button className={!activeCamera ? "active" : ""} onClick={() => setActiveCamera(null)}>
              Overview
            </button>
            {scene.cameras
              .filter((c) => c.id !== "cam-overview")
              .map((c) => (
                <button
                  key={c.id}
                  className={activeCamera === c.id ? "active" : ""}
                  onClick={() => setActiveCamera(c.id)}
                >
                  {c.name}
                </button>
              ))}
            <button onClick={() => setPanel((p) => !p)}>{panel ? "hide panel" : "show panel"}</button>
            <DownloadMenu job={job} />
            <button onClick={onNew}>new upload</button>
          </div>
        </div>
        <Compare job={job} />
        <div className="overlay-bottom">
          <label>
            <input type="checkbox" checked={showCeilings} onChange={(e) => setShowCeilings(e.target.checked)} />
            Ceilings
          </label>
          <label>
            <input type="checkbox" checked={showFurniture} onChange={(e) => setShowFurniture(e.target.checked)} />
            Furniture
          </label>
          <label className="slider">
            Exposure
            <input type="range" min={0.4} max={2.2} step={0.05} value={exposure}
              onChange={(e) => setExposure(parseFloat(e.target.value))} />
          </label>
        </div>
      </div>

      {panel && (
        <aside className="panel">
          {job.warnings.length > 0 && (
            <section>
              <h2>Check before trusting <span className="count">{job.warnings.length}</span></h2>
              <ul className="notes">
                {job.warnings.slice(0, 8).map((w, i) => (<li key={i}>{w}</li>))}
              </ul>
            </section>
          )}

          {scene.art_direction?.rationale && (
            <section>
              <h2>Art direction</h2>
              <p className="hint" style={{ marginBottom: 6 }}><strong>{scene.art_direction.mood}</strong></p>
              <p className="hint">{scene.art_direction.rationale}</p>
              <p className="hint" style={{ opacity: 0.7 }}>decided by {scene.art_direction.source}</p>
            </section>
          )}

          <section>
            <h2>Design scheme</h2>
            <div className="schemes">
              {Object.values(schemes).map((s) => (
                <button key={s.id} className={`scheme${schemeId === s.id ? " active" : ""}`}
                  onClick={() => setScheme(s.id)}>
                  <span className="scheme-name">{s.name}</span>
                  <span className="scheme-desc">{s.description}</span>
                </button>
              ))}
            </div>
          </section>

          <section>
            <div className="h2row">
              <h2>Finishes</h2>
              {Object.keys(overrides).length > 0 && (
                <button className="mini" onClick={resetOverrides}>reset {Object.keys(overrides).length}</button>
              )}
            </div>
            {SLOT_GROUPS.map((g) => (
              <div key={g.label} className="group">
                <h3>{g.label}</h3>
                {g.slots.map((s) => (<SlotControl key={s.slot} slot={s.slot} label={s.label} />))}
              </div>
            ))}
          </section>

          <section>
            <h2>Rooms <span className="count">{scene.rooms.length}</span></h2>
            <p className="hint">{totalSqft.toFixed(0)} sq ft total · click a room to jump a camera</p>
            <RoomList />
          </section>

          <section>
            <h2>Furniture</h2>
            <FurniturePanel />
          </section>

          <section>
            <h2>Provenance</h2>
            <ul className="notes">
              {scene.notes.map((n, i) => (<li key={i}>{n}</li>))}
            </ul>
          </section>
        </aside>
      )}
    </div>
  );
}
