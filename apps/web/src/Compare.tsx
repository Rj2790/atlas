/**
 * Side-by-side comparison of the input drawing and the 3D result.
 *
 * The point of this view is verification, not presentation. The single most useful
 * question about any of this is "does the model actually match the drawing?", and the
 * only honest way to answer it is to put them next to each other and let the eye check.
 *
 * Three left-hand views, because they answer three different questions:
 *
 * * **Source** — the drawing as uploaded. Ground truth.
 * * **Extracted** — what the pipeline *read*: recovered walls with their thicknesses,
 *   room polygons, openings, and each room's printed dimension beside its measured
 *   area. Comparing this against the source finds extraction errors.
 * * **Layout** — the same drawing with furniture footprints and facing directions.
 *   A sofa in a doorway or a bed through a wall is invisible in a coordinate list and
 *   unmissable here; this view is how several real placement bugs were found.
 *
 * The divider is draggable rather than fixed at 50%, because comparing a tall
 * residential plan against a 3D view needs different splits at different moments.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { Viewer } from "./Viewer";
import { jobUrl, unitUrl, type Job } from "./api";
import { useStore } from "./store";

type LeftView = "source" | "extracted" | "layout";

export default function Compare({ job }: { job: Job }) {
  const [split, setSplit] = useState(0.46);
  const [left, setLeft] = useState<LeftView>(job.kind === "raster" ? "source" : "extracted");
  const [zoom, setZoom] = useState(1);
  const wrap = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const scene = useStore((s) => s.scene);
  const unit = useStore((s) => s.unit);

  const onMove = useCallback((e: MouseEvent) => {
    if (!dragging.current || !wrap.current) return;
    const r = wrap.current.getBoundingClientRect();
    // Clamped so neither pane can be dragged to nothing.
    setSplit(Math.min(0.8, Math.max(0.2, (e.clientX - r.left) / r.width)));
  }, []);

  useEffect(() => {
    const up = () => (dragging.current = false);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", up);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", up);
    };
  }, [onMove]);

  // The drawing itself is the whole floor, so it is never per-unit. The extraction and
  // layout drawings are: without this the left pane kept showing the first dwelling's plan
  // while the right showed whichever one you had picked, which is worse than no comparison
  // at all.
  const asset = (name: string) =>
    unit != null ? unitUrl(job.id, unit, name) : jobUrl(job.id, name);
  const src =
    left === "source"
      ? jobUrl(job.id, "source")
      : left === "extracted"
        ? asset("plan.svg")
        : asset("layout.svg");

  const q = job.quality ?? {};
  const agree = q["mean_dimension_agreement"];

  return (
    <div className="compare" ref={wrap}>
      <div className="pane left" style={{ width: `${split * 100}%` }}>
        <div className="pane-bar">
          <div className="seg">
            {(["source", "extracted", "layout"] as LeftView[]).map((v) => (
              <button
                key={v}
                className={left === v ? "active" : ""}
                onClick={() => setLeft(v)}
                disabled={v === "source" && job.kind === "cad"}
                title={
                  v === "source" && job.kind === "cad"
                    ? "a DWG has no image to show — use Extracted"
                    : v === "source"
                      ? "the drawing you uploaded"
                      : v === "extracted"
                        ? "what the pipeline read from it"
                        : "furniture footprints and facing"
                }
              >
                {v}
              </button>
            ))}
          </div>
          <div className="zoom">
            <button className="mini" onClick={() => setZoom((z) => Math.max(0.4, z - 0.2))}>
              −
            </button>
            <span className="hint">{Math.round(zoom * 100)}%</span>
            <button className="mini" onClick={() => setZoom((z) => Math.min(4, z + 0.2))}>
              +
            </button>
            <button className="mini" onClick={() => setZoom(1)}>
              fit
            </button>
          </div>
        </div>
        <div className="pane-body scroll">
          <img src={src} alt={left} style={{ width: `${zoom * 100}%` }} />
        </div>
      </div>

      <div
        className="divider"
        onMouseDown={() => (dragging.current = true)}
        role="separator"
        aria-orientation="vertical"
      />

      <div className="pane right" style={{ width: `${(1 - split) * 100}%` }}>
        <div className="pane-bar">
          <span className="pane-title">3D result</span>
          <span className="hint">
            {scene ? `${scene.rooms.length} rooms · ${scene.furniture.length} items` : "loading…"}
            {typeof agree === "number" ? ` · ${Math.round(agree * 100)}% dimension match` : ""}
          </span>
        </div>
        <div className="pane-body">
          <Viewer />
        </div>
      </div>
    </div>
  );
}
