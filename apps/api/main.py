"""Upload API: a plan goes in, a comparable 3D scene comes out.

The CLI was enough to develop the pipeline but it is not a platform. This is the
smallest honest service around it: upload a drawing, watch it build, then look at the
source and the result side by side.

Two design points worth stating.

**Builds are background jobs with polled status, not a long request.** A DWG takes 15
to 60 seconds — LibreDWG conversion, block flattening, wall solving — and the first
build also downloads texture sets. A synchronous request that long dies to proxy
timeouts and gives the user a blank page with no idea whether anything is happening.
Each stage reports progress instead.

**CAD is a two-step flow.** A production drawing is a whole floor: the sample holds 16
apartments. Uploading it and being handed "the plan" would be meaningless, so a CAD
upload first returns the dwelling list and the user picks. A raster upload is a single
dwelling and skips straight to the build.

Texture and model caches are shared across jobs and served from one place — they are
~250 MB and identical for every build.
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))

JOBS_DIR = ROOT / "out-jobs"
SHARED = ROOT / "data"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

CAD_SUFFIXES = {".dwg", ".dxf"}
IMG_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

app = FastAPI(title="Atlas", version="1.0")
app.add_middleware(
    CORSMiddleware,
    # Any localhost origin, on any port.
    #
    # A hardcoded port list is a trap: Vite moves to the next free port when one is
    # taken, and the failure is silent and misleading. `multipart/form-data` is a
    # CORS-safelisted content type, so an upload POST from a disallowed origin still
    # *reaches* the server and builds the job — only the response is blocked. The
    # server logs a completed build while the browser reports "Failed to fetch" and
    # the UI never advances, which looks like a front-end state bug and is not one.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# job state
# ---------------------------------------------------------------------------

@dataclass
class Job:
    id: str
    source_name: str
    kind: Literal["cad", "raster"]
    status: Literal["queued", "building", "done", "error"] = "queued"
    """No "units" state any more: a CAD upload no longer pauses to ask which dwelling to
    build. It builds all of them and the viewer switches between the results."""

    progress: float = 0.0
    stage: str = "queued"
    units: list[dict[str, Any]] = field(default_factory=list)
    built: list[dict[str, Any]] = field(default_factory=list)
    """Dwellings that have actually been built, with per-unit metadata for the picker."""

    error: str = ""
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)
    art_direction: dict[str, Any] = field(default_factory=dict)
    created: float = field(default_factory=time.time)

    @property
    def dir(self) -> Path:
        return JOBS_DIR / self.id

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_name": self.source_name,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 3),
            "stage": self.stage,
            "units": self.units,
            "built": self.built,
            "error": self.error,
            "notes": self.notes,
            "warnings": self.warnings,
            "quality": self.quality,
            "art_direction": self.art_direction,
        }


_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()


def _job(job_id: str) -> Job:
    with _LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, f"no such job {job_id}")
    return job


# ---------------------------------------------------------------------------
# pipeline work
# ---------------------------------------------------------------------------

def _build(job: Job, src: Path, unit: int | None, scheme: str | None, direct: bool) -> None:
    """Run the full pipeline into the job directory.

    A CAD drawing is a whole floor, so **every** dwelling in it is built and the viewer
    switches between them. Making the user pick one up front was the wrong shape: they
    cannot know which unit they want before seeing any of them, and re-running the
    pipeline per unit repeats the expensive part for nothing.

    The cost is much lower than it looks, because the browser renders from `scene.json`
    rather than from the GLB:

    * **Extraction runs once** for the whole drawing — the slow step, ~60 s on the demo
      file — and every unit is a `subset()` of that one result.
    * **Textures and furniture models are prepared once** and shared by all units.
    * **GLBs are not built here.** They are exported on demand by the download route,
      because sixteen of them would be ~320 MB and several minutes for files most users
      never ask for.
    """
    from atlas.build.assemble import AssembleConfig, assemble_scene
    from atlas.build.models import ModelCache
    from atlas.build.schemes import get_scheme
    from atlas.build.textures import TextureCache
    from atlas.render.plan_svg import SvgStyle, render_plan

    try:
        job.status, job.stage, job.progress = "building", "reading the plan", 0.15

        evidence = None
        if job.kind == "cad":
            from atlas.ingest.dxf_plan import extract_plan
            from atlas.audit import _drawing_evidence

            full = extract_plan(src, work_dir=job.dir / "work", max_units=40)
            # The drawing's own testimony (walls, door symbols, window line work), once
            # per job, so every dwelling's issue count runs the same audit the test
            # suite runs — a weaker audit here is how the picker once showed 4 issues
            # on a drawing the real audit passes.
            evidence = _drawing_evidence(src)
            plans = (
                [(n, full.subset(u), u) for n, u in enumerate(full.units, 1)]
                if full.units
                else [(1, full, None)]
            )
            # One unit explicitly requested: keep the others already on disk.
            if unit is not None:
                plans = [t for t in plans if t[0] == unit] or plans[:1]
            job.notes = list(full.ingest_notes)
            job.warnings = list(full.warnings)
        else:
            from atlas.ingest.raster_vlm import VlmConfig, extract_plan_from_image

            job.stage = "reading the drawing with a vision model"
            full = extract_plan_from_image(src, cfg=VlmConfig())
            plans = [(1, full, None)]
            job.notes = list(full.ingest_notes)
            job.warnings = list(full.warnings)

        if not any(p.rooms for _, p, _ in plans):
            job.status, job.error = "error", "no rooms could be recovered from this drawing"
            return

        (job.dir / "plan.json").write_text(full.to_json())

        built: list[dict[str, Any]] = []
        for i, (n, plan, meta) in enumerate(plans):
            if not plan.rooms:
                continue
            job.stage = f"building dwelling {n} of {len(plans)}"
            job.progress = 0.15 + 0.55 * (i / max(len(plans), 1))
            udir = job.dir / "units" / str(n)
            udir.mkdir(parents=True, exist_ok=True)

            scene = assemble_scene(
                plan,
                config=AssembleConfig(
                    scheme_id=scheme or "warm-minimal", direct_art=direct
                ),
            )
            (udir / "scene.json").write_text(scene.to_json())
            (udir / "plan.json").write_text(plan.to_json())
            (udir / "plan.svg").write_text(render_plan(plan, SvgStyle(width_px=1500)))
            (udir / "layout.svg").write_text(
                render_plan(plan, SvgStyle(width_px=1500, show_furniture=True), scene)
            )
            q = plan.quality_report()
            built.append(
                {
                    "unit": n,
                    "name": (meta.name if meta else "Plan"),
                    "declared_sqft": (meta.stated_area_sqft if meta else None),
                    "measured_sqft": q.get("total_area_sqft"),
                    "rooms": q.get("rooms"),
                    "openings": q.get("openings"),
                    "mix": _room_mix(plan),
                    "scheme": scene.scheme_id,
                    # Invariant failures for *this* dwelling, not the drawing's warning
                    # count.
                    #
                    # `subset()` copies the whole plan's warnings, so every unit reported
                    # the same number — 83 for all sixteen — which is worse than showing
                    # nothing: it makes a good dwelling and a poor one look identical in the
                    # menu, defeating the reason the figure is there. Unit names repeat
                    # ("4 BHK" four times) so the warnings cannot be filtered by name
                    # either. The audit measures per dwelling by construction.
                    **_unit_quality_figures(plan, evidence),
                }
            )
            # The first built unit is also the job-level default, so existing callers
            # of /scene.json keep working unchanged.
            if i == 0:
                for name in ("scene.json", "plan.svg", "layout.svg"):
                    (job.dir / name).write_text((udir / name).read_text())
                job.quality = q
                job.art_direction = dict(scene.art_direction)
                job.notes.extend(scene.notes)

        job.built = built
        if not built:
            job.status, job.error = "error", "no dwelling could be built"
            return

        job.stage, job.progress = "preparing materials", 0.78
        schemes = {b["scheme"] for b in built}
        wanted = sorted(
            {
                m.acg_asset
                for sid in schemes
                for m in get_scheme(sid).materials.values()
                if m.acg_asset
            }
        )
        TextureCache(SHARED / "materials").ensure(wanted)

        job.stage, job.progress = "preparing furniture models", 0.92
        from atlas.furnish.catalog import CATALOG

        ModelCache(cache_dir=SHARED / "models").ensure(
            sorted({i.external_model for i in CATALOG.values() if i.external_model})
        )

        job.status, job.stage, job.progress = "done", f"{len(built)} dwelling(s) ready", 1.0
    except Exception as exc:  # noqa: BLE001
        job.status, job.error = "error", f"{type(exc).__name__}: {exc}"


def _unit_quality_figures(plan: "Any", evidence: "Any" = None) -> dict:
    """Issue count plus the honest area figure, from the audit itself.

    ``evidence`` is the drawing's own testimony bundle. It must be supplied: without it
    the audit runs on built walls alone, so a floor edge sealed by drawn line work reads
    as a leak — the picker showed 4 issues on a drawing the real audit passes cleanly,
    and a quality figure that disagrees with the quality gate is worse than none.

    ``accounted`` is the share of the declared carpet polygon assigned to floor or wall
    structure. It is the like-for-like comparison: carpet area includes internal walls,
    so bare floor divided by declared always reads 77-90% and looks like a defect when
    it is a definition. Users read the raw ratio as "off by 15-20%"; this figure is the
    one that actually measures extraction quality (92-100% across the demo drawing).
    """
    try:
        from atlas.audit import audit_plan

        res = audit_plan(
            plan,
            drawn=(evidence.walls if evidence else None),
            evidence=evidence,
        )
        return {
            "issues": sum(len(u.failures) for u in res.units),
            "accounted": next(
                (u.accounted for u in res.units if u.accounted is not None), None
            ),
            "dims_agreement": next(
                (u.dims for u in res.units if u.dims is not None), None
            ),
        }
    except Exception:  # noqa: BLE001 - a quality figure must never fail a build
        return {"issues": 0, "accounted": None, "dims_agreement": None}


def _room_mix(plan: "Any") -> str:
    """e.g. "3x bedroom, 2x bathroom" — enough to tell dwellings apart in a menu."""
    from collections import Counter

    mix = Counter(r.kind.value for r in plan.rooms if r.kind.is_habitable or r.kind.is_wet)
    return ", ".join(f"{n}x {k}" for k, n in mix.most_common(4))


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    scheme: str | None = Form(None),
    direct: bool = Form(True),
) -> JSONResponse:
    """Accept a drawing. CAD returns a dwelling list; a raster builds immediately."""
    name = Path(file.filename or "upload").name
    suffix = Path(name).suffix.lower()
    if suffix not in CAD_SUFFIXES | IMG_SUFFIXES:
        raise HTTPException(
            400,
            f"unsupported file type {suffix!r}. Accepts .dwg/.dxf for measured geometry "
            "and .png/.jpg for vision-read geometry.",
        )

    job = Job(
        id=uuid.uuid4().hex[:12],
        source_name=name,
        kind="cad" if suffix in CAD_SUFFIXES else "raster",
    )
    job.dir.mkdir(parents=True, exist_ok=True)
    src = job.dir / f"source{suffix}"
    with src.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    with _LOCK:
        _JOBS[job.id] = job

    # Build straight away, every dwelling, whatever the file holds.
    #
    # A CAD upload used to stop and ask which dwelling to build. That is the wrong way
    # round — you cannot know which flat you want before seeing any of them — and it made
    # extraction run twice, once to list the dwellings and again to build the chosen one.
    # Extraction is the expensive step, so scanning first cost roughly a minute to gain
    # nothing. `_build` lists and builds in one pass; the viewer switches between the
    # results.
    background.add_task(_build, job, src, None, scheme, direct)
    return JSONResponse(job.public())


@app.post("/api/jobs/{job_id}/build")
async def build_unit(
    job_id: str,
    background: BackgroundTasks,
    unit: int = Form(...),
    scheme: str | None = Form(None),
    direct: bool = Form(True),
) -> JSONResponse:
    """Build a chosen dwelling from an already-uploaded CAD file."""
    job = _job(job_id)
    src = next(iter(job.dir.glob("source.*")), None)
    if src is None:
        raise HTTPException(404, "source file missing")
    job.status, job.progress, job.error = "queued", 0.0, ""
    background.add_task(_build, job, src, unit, scheme, direct)
    return JSONResponse(job.public())


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str) -> JSONResponse:
    return JSONResponse(_job(job_id).public())


@app.get("/api/jobs/{job_id}/scene.json")
async def job_scene(job_id: str) -> FileResponse:
    return _file(_job(job_id), "scene.json", "application/json")


@app.get("/api/jobs/{job_id}/plan.svg")
async def job_plan(job_id: str) -> FileResponse:
    return _file(_job(job_id), "plan.svg", "image/svg+xml")


@app.get("/api/jobs/{job_id}/layout.svg")
async def job_layout(job_id: str) -> FileResponse:
    return _file(_job(job_id), "layout.svg", "image/svg+xml")


@app.get("/api/jobs/{job_id}/model.glb")
async def job_glb(job_id: str) -> FileResponse:
    return _file(_job(job_id), "model.glb", "model/gltf-binary")


@app.get("/api/jobs/{job_id}/units/{unit}/scene.json")
async def unit_scene(job_id: str, unit: int) -> FileResponse:
    return _unit_file(_job(job_id), unit, "scene.json", "application/json")


@app.get("/api/jobs/{job_id}/units/{unit}/plan.svg")
async def unit_plan(job_id: str, unit: int) -> FileResponse:
    return _unit_file(_job(job_id), unit, "plan.svg", "image/svg+xml")


@app.get("/api/jobs/{job_id}/units/{unit}/layout.svg")
async def unit_layout(job_id: str, unit: int) -> FileResponse:
    return _unit_file(_job(job_id), unit, "layout.svg", "image/svg+xml")


@app.get("/api/jobs/{job_id}/units/{unit}/model.glb")
async def unit_glb(job_id: str, unit: int, scheme: str | None = None) -> FileResponse:
    """Export this dwelling's GLB on first request, then serve it from disk.

    Building every unit's GLB up front would be ~320 MB and several minutes on a
    sixteen-dwelling drawing, for files most users never download. The browser renders
    from scene.json, so nothing on screen waits for this.
    """
    import json as _json

    from atlas.build.assemble import AssembleConfig, assemble_scene
    from atlas.build.models import ModelCache
    from atlas.build.textures import TextureCache
    from atlas.planir import PlanIR

    job = _job(job_id)
    udir = job.dir / "units" / str(unit)
    plan_path = udir / "plan.json"
    if not plan_path.exists():
        raise HTTPException(409, f"dwelling {unit} is not built (job is {job.status})")

    recorded = next((b for b in job.built if b.get("unit") == unit), None)
    sid = scheme or (recorded or {}).get("scheme") or "warm-minimal"
    out = udir / f"model-{sid}.glb"
    if not out.exists():
        from atlas.build.gltf import export_glb

        # Re-assemble rather than deserialise the scene: SceneIR is write-only today, and
        # assembly is deterministic, so with the art director off and the scheme the
        # viewer is already showing, this reproduces exactly what is on screen.
        plan = PlanIR.from_dict(_json.loads(plan_path.read_text()))
        scene = assemble_scene(
            plan, config=AssembleConfig(scheme_id=sid, direct_art=False)
        )
        export_glb(
            scene,
            out,
            scheme_id=sid,
            textures=TextureCache(SHARED / "materials"),
            models=ModelCache(cache_dir=SHARED / "models"),
        )
    return FileResponse(out, media_type="model/gltf-binary")


def _unit_file(job: Job, unit: int, name: str, mime: str) -> FileResponse:
    path = job.dir / "units" / str(unit) / name
    if not path.exists():
        raise HTTPException(409, f"dwelling {unit}/{name} not ready (job is {job.status})")
    return FileResponse(path, media_type=mime)


@app.get("/api/jobs/{job_id}/source")
async def job_source(job_id: str) -> FileResponse:
    """The uploaded file itself — the left-hand side of the comparison."""
    job = _job(job_id)
    src = next(iter(job.dir.glob("source.*")), None)
    if src is None:
        raise HTTPException(404, "source missing")
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(src.suffix.lower(), "application/octet-stream")
    return FileResponse(src, media_type=mime)


def _file(job: Job, name: str, mime: str) -> FileResponse:
    path = job.dir / name
    if not path.exists():
        raise HTTPException(409, f"{name} not ready (job is {job.status})")
    return FileResponse(path, media_type=mime)


@app.get("/api/schemes")
async def schemes() -> JSONResponse:
    """Scheme and catalogue metadata, identical to what the CLI exports."""
    from atlas.build.schemes import scheme_summaries

    return JSONResponse({"schemes": scheme_summaries()})


@app.get("/api/health")
async def health() -> dict[str, Any]:
    import shutil as sh

    return {
        "ok": True,
        "dwg_reader": sh.which("dwg2dxf") is not None,
        "blender": Path("/Applications/Blender.app/Contents/MacOS/Blender").exists(),
        "jobs": len(_JOBS),
    }


# Shared asset caches. Identical for every job and ~250 MB, so served once.
for route, folder in (("/data/textures", SHARED / "materials" / "web"), ("/data/models", SHARED / "models")):
    folder.mkdir(parents=True, exist_ok=True)
    app.mount(route, StaticFiles(directory=str(folder)), name=route.strip("/").replace("/", "-"))
