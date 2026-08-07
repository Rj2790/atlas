/**
 * Client for the build API.
 *
 * Builds are polled rather than awaited. A DWG takes 15-60 seconds — LibreDWG
 * conversion, block flattening, wall solving, then texture preparation — and a
 * request held open that long dies to proxy timeouts and leaves the user staring at
 * nothing. Polling also lets the UI name the stage it is in, which is the difference
 * between "working" and "hung".
 */

export const API = import.meta.env.VITE_API ?? "http://127.0.0.1:8077";

export type JobStatus = "queued" | "building" | "done" | "error";

export interface UnitRow {
  index: number;
  name: string;
  declared_sqft: number | null;
  measured_sqft: number;
  rooms: number;
  mix: Record<string, number>;
}

/** A dwelling the backend has actually built, with enough to tell it apart in a menu. */
export interface BuiltUnit {
  unit: number;
  name: string;
  declared_sqft: number | null;
  measured_sqft: number | null;
  rooms: number | null;
  openings: number | null;
  mix: string;
  scheme: string;
  /** Geometric invariants this dwelling violates; 0 is clean. */
  issues: number;
  /** Share of the declared carpet polygon assigned to floor or wall structure, 0-1.
   *  The like-for-like figure: carpet area includes internal walls, so bare floor
   *  over declared reads 77-90% by definition and is not shown as a percentage. */
  accounted: number | null;
  /** Mean printed-dimension agreement across labelled rooms, 0-1. */
  dims_agreement: number | null;
}

export interface Job {
  id: string;
  source_name: string;
  kind: "cad" | "raster";
  status: JobStatus;
  progress: number;
  stage: string;
  units: UnitRow[];
  /** Every dwelling built for this job. A CAD drawing is a whole floor, so the viewer
   *  switches between them rather than making the user choose one before seeing any. */
  built: BuiltUnit[];
  error: string;
  notes: string[];
  warnings: string[];
  quality: Record<string, number | null>;
  art_direction: {
    scheme_id?: string;
    rationale?: string;
    mood?: string;
    source?: string;
    hero_room_id?: string;
  };
}

async function asJson<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try {
      const body = await r.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* keep the status line */
    }
    throw new Error(detail);
  }
  return (await r.json()) as T;
}

export async function uploadPlan(file: File, opts: { scheme?: string; direct?: boolean } = {}) {
  const fd = new FormData();
  fd.append("file", file);
  if (opts.scheme) fd.append("scheme", opts.scheme);
  fd.append("direct", String(opts.direct ?? true));
  return asJson<Job>(await fetch(`${API}/api/upload`, { method: "POST", body: fd }));
}

export async function buildUnit(jobId: string, unit: number, opts: { scheme?: string; direct?: boolean } = {}) {
  const fd = new FormData();
  fd.append("unit", String(unit));
  if (opts.scheme) fd.append("scheme", opts.scheme);
  fd.append("direct", String(opts.direct ?? true));
  return asJson<Job>(await fetch(`${API}/api/jobs/${jobId}/build`, { method: "POST", body: fd }));
}

export async function getJob(jobId: string) {
  return asJson<Job>(await fetch(`${API}/api/jobs/${jobId}`));
}

/** Poll until the job leaves a working state. Reports each status to `onTick`. */
export async function pollJob(
  jobId: string,
  onTick: (j: Job) => void,
  { intervalMs = 1500, timeoutMs = 15 * 60_000 } = {},
): Promise<Job> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const job = await getJob(jobId);
    onTick(job);
    if (job.status === "done" || job.status === "error") return job;
    if (Date.now() > deadline) throw new Error("build timed out");
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}

export const jobUrl = (jobId: string, name: string) => `${API}/api/jobs/${jobId}/${name}`;

/** Asset URL for one dwelling of a job. */
export const unitUrl = (jobId: string, unit: number, name: string) =>
  `${API}/api/jobs/${jobId}/units/${unit}/${name}`;
