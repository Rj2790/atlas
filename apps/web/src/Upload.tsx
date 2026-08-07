/**
 * Upload, dwelling selection, and build progress.
 *
 * CAD and raster deliberately take different routes. A production drawing is a whole
 * floor — the sample holds 16 apartments — so handing back "the plan" would be
 * meaningless; a CAD upload lists the dwellings and the user picks one. A raster sheet
 * is a single dwelling and goes straight to building.
 *
 * The build reports its stage rather than showing a spinner, because it takes tens of
 * seconds and "converting DWG" versus "preparing materials" is the difference between
 * waiting and wondering whether it has hung.
 */

import { useCallback, useRef, useState } from "react";
import { pollJob, uploadPlan, type Job } from "./api";

const ACCEPT = ".dwg,.dxf,.png,.jpg,.jpeg,.webp";

export default function Upload({ onReady }: { onReady: (job: Job) => void }) {
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const run = useCallback(
    async (file: File) => {
      setErr(null);
      setBusy(true);
      try {
        const started = await uploadPlan(file);
        setJob(started);
        // Straight through. A CAD drawing is a whole floor and every dwelling in it is
        // built in one pass, so there is nothing to ask: the viewer opens on the first and
        // switches between the rest.
        const finished = await pollJob(started.id, setJob);
        if (finished.status === "error") setErr(finished.error);
        else if (finished.status === "done") onReady(finished);
      } catch (e) {
        setErr((e as Error).message);
      } finally {
        setBusy(false);
      }
    },
    [onReady],
  );

  const working = busy;

  return (
    <div className="upload">
      <div className="upload-inner">
        <h1>Atlas</h1>
        <p className="lede">
          A 2D floor plan becomes a dimensioned, furnished, modular 3D model. Upload a
          DWG or DXF for geometry measured from the drawing, or a plan image to have its
          printed dimensions read.
        </p>

        {!job && (
          <>
            <div
              className={`drop${drag ? " over" : ""}`}
              onDragOver={(e) => {
                e.preventDefault();
                setDrag(true);
              }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDrag(false);
                const f = e.dataTransfer.files?.[0];
                if (f) void run(f);
              }}
              onClick={() => input.current?.click()}
              role="button"
              tabIndex={0}
            >
              <strong>Drop a floor plan here</strong>
              <span>or click to choose · DWG, DXF, PNG, JPG</span>
              <input
                ref={input}
                type="file"
                accept={ACCEPT}
                hidden
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) void run(f);
                }}
              />
            </div>
            <p className="hint centred">
              DWG is read with LibreDWG and measured exactly. An image is read by a
              vision model, which recovers the printed room dimensions but approximates
              the arrangement.
            </p>
          </>
        )}

        {job && (
          <div className="job">
            <div className="job-head">
              <strong>{job.source_name}</strong>
              <span className="hint">{job.kind === "cad" ? "CAD · measured" : "image · vision-read"}</span>
            </div>

            {working && (
              <>
                <div className="bar">
                  <div className="bar-fill" style={{ width: `${Math.max(4, job.progress * 100)}%` }} />
                </div>
                <p className="hint">{job.stage}</p>
              </>
            )}

            {err && <p className="err">{err}</p>}

            {!working && (
              <button
                className="mini"
                onClick={() => {
                  setJob(null);
                  setErr(null);
                }}
              >
                start over
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
