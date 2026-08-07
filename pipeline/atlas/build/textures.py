"""Acquire and prepare CC0 PBR texture maps.

Flat base colours are what make a correct model still read as a massing study
rather than a photograph. The eye identifies a material from its *microstructure* —
oak grain, tile grout, the weave of linen — long before it reads the colour, so no
amount of tuning the base colour or roughness scalar substitutes for a real map.

Textures come from **ambientCG**, which is CC0: usable commercially with no
attribution obligation and no per-asset licence audit. (Objaverse and 3D-FUTURE are
per-object licensed, which is why the furniture catalogue is parametric instead.)

Three practical decisions:

**NormalGL, not NormalDX.** ambientCG ships both. glTF, three.js and Blender all
expect the OpenGL convention with +Y up. Picking the DirectX map inverts the green
channel, and the result is subtly, maddeningly wrong: bevels light from the wrong
side and nothing looks obviously broken.

**Two output sizes.** 1K maps go to the web app as separate files so the browser can
cache and parallel-load them; 512 px versions are embedded in the GLB, because a GLB
carrying twenty 1K materials at four maps each is tens of megabytes and this one has
to stay openable.

**Displacement and metalness are discarded.** Displacement needs tessellation or
parallax to mean anything and silently does nothing in a plain glTF viewer;
metalness for architectural surfaces is either 0 or 1 and is already a scheme
scalar. Shipping maps that have no effect is just weight.
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "TextureSet",
    "TextureCache",
    "MAP_ROLES",
    "acg_download_url",
]

ACG_GET = "https://ambientcg.com/get?file={file}"
_UA = "atlas-floorplan/1.0 (+https://github.com/atlas-plan)"

# Suffix in the ambientCG zip -> our role name. Order matters only for logging.
MAP_ROLES: dict[str, str] = {
    "_Color": "base_color",
    "_NormalGL": "normal",
    "_Roughness": "roughness",
    "_AmbientOcclusion": "ao",
}


def acg_download_url(asset_id: str, resolution: str = "1K", fmt: str = "JPG") -> str:
    return ACG_GET.format(file=f"{asset_id}_{resolution}-{fmt}.zip")


@dataclass(slots=True)
class TextureSet:
    """The prepared maps for one material."""

    asset_id: str
    web: dict[str, str] = field(default_factory=dict)
    """role -> path relative to the web data directory."""

    embed: dict[str, Path] = field(default_factory=dict)
    """role -> absolute path to the small version for GLB embedding."""

    credit: str = ""

    @property
    def ok(self) -> bool:
        return "base_color" in self.web or "base_color" in self.embed


class TextureCache:
    """Downloads, extracts and resizes ambientCG materials, with a local cache.

    The cache is keyed on asset id and resolution, so a rebuild costs nothing and
    a scheme change only fetches what it newly needs.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        resolution: str = "1K",
        web_px: int = 1024,
        embed_px: int = 512,
        timeout: int = 300,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.raw_dir = self.cache_dir / "raw"
        self.web_dir = self.cache_dir / "web"
        self.embed_dir = self.cache_dir / "embed"
        for d in (self.raw_dir, self.web_dir, self.embed_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.resolution = resolution
        self.web_px = web_px
        self.embed_px = embed_px
        self.timeout = timeout
        self.notes: list[str] = []
        self._sets: dict[str, TextureSet] = {}

    # -- download -----------------------------------------------------------

    def _zip_path(self, asset_id: str) -> Path:
        return self.raw_dir / f"{asset_id}_{self.resolution}-JPG.zip"

    def _fetch(self, asset_id: str) -> Path | None:
        dest = self._zip_path(asset_id)
        if dest.exists() and dest.stat().st_size > 10_000:
            return dest
        url = acg_download_url(asset_id, self.resolution)
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read()
            if len(data) < 10_000 or not data[:2] == b"PK":
                self.notes.append(f"{asset_id}: download did not look like a zip")
                return None
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001 - a missing texture must not fail a build
            self.notes.append(f"{asset_id}: download failed ({type(exc).__name__}: {exc})")
            return None

    # -- prepare ------------------------------------------------------------

    def _prepare(self, asset_id: str) -> TextureSet:
        ts = TextureSet(asset_id=asset_id, credit=f"{asset_id} — ambientCG, CC0")
        zip_path = self._fetch(asset_id)
        if zip_path is None:
            return ts

        try:
            from PIL import Image
        except ImportError:
            self.notes.append("Pillow not installed; textures skipped")
            return ts

        try:
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
                for suffix, role in MAP_ROLES.items():
                    match = next(
                        (n for n in names if n.endswith(f"{suffix}.jpg") or n.endswith(f"{suffix}.png")),
                        None,
                    )
                    if match is None:
                        continue
                    with zf.open(match) as fh:
                        img = Image.open(io.BytesIO(fh.read()))
                        img.load()

                    # Normal maps must stay RGB; the rest are single-channel data
                    # and storing them as greyscale cuts the payload by two thirds.
                    if role == "normal":
                        img = img.convert("RGB")
                    elif role in ("roughness", "ao"):
                        img = img.convert("L")
                    else:
                        img = img.convert("RGB")

                    # A neutralised colour variant for painted/plastered surfaces:
                    # luminance only, re-levelled to a mid tone so the scheme's
                    # base_color decides the hue outright (see
                    # Material.texture_neutralize).
                    variants: list[tuple[str, "Image.Image"]] = [(role, img)]
                    if role == "base_color":
                        grey = img.convert("L")
                        stat = grey.resize((64, 64)).getdata()
                        mean = (sum(stat) / len(stat)) or 128.0
                        gain = 210.0 / mean
                        levelled = grey.point(lambda v, g=gain: min(255, int(v * g)))
                        variants.append(("base_color_neutral", levelled.convert("RGB")))

                    for vrole, vimg in variants:
                      for px, out_dir, target in (
                        (self.web_px, self.web_dir, "web"),
                        (self.embed_px, self.embed_dir, "embed"),
                    ):
                        resized = vimg if max(vimg.size) <= px else vimg.resize(
                            (px, px), Image.LANCZOS
                        )
                        out = out_dir / f"{asset_id}_{vrole}.jpg"
                        if not out.exists():
                            mode = "RGB" if vrole in ("base_color", "base_color_neutral", "normal") else "L"
                            resized.convert(mode).save(out, "JPEG", quality=88, optimize=True)
                        if target == "web":
                            ts.web[vrole] = out.name
                        else:
                            ts.embed[vrole] = out
        except Exception as exc:  # noqa: BLE001
            self.notes.append(f"{asset_id}: extract failed ({type(exc).__name__}: {exc})")
        return ts

    # -- public -------------------------------------------------------------

    def ensure(self, asset_ids: list[str], *, workers: int = 6) -> dict[str, TextureSet]:
        """Download and prepare every asset, in parallel. Never raises.

        A material whose textures could not be fetched simply keeps its scalar
        values and still renders — degraded, not broken — and the reason lands in
        ``notes``.
        """
        todo = [a for a in dict.fromkeys(asset_ids) if a and a not in self._sets]
        if not todo:
            return self._sets

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for ts in pool.map(self._prepare, todo):
                self._sets[ts.asset_id] = ts

        ok = sum(1 for a in todo if self._sets[a].ok)
        self.notes.append(
            f"textures: {ok}/{len(todo)} ambientCG materials ready "
            f"({self.resolution}, web {self.web_px}px, embed {self.embed_px}px)"
        )
        return self._sets

    def get(self, asset_id: str | None) -> TextureSet | None:
        if not asset_id:
            return None
        ts = self._sets.get(asset_id)
        return ts if ts and ts.ok else None

    def publish_web(self, dest: Path | str) -> int:
        """Copy the web-sized maps into a served directory. Returns file count."""
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for ts in self._sets.values():
            for name in ts.web.values():
                src = self.web_dir / name
                if src.exists():
                    out = dest / name
                    if not out.exists() or out.stat().st_size != src.stat().st_size:
                        shutil.copy2(src, out)
                    n += 1
        return n

    def credits(self) -> list[str]:
        """Attribution lines. CC0 needs none, but recording provenance is right."""
        return sorted({ts.credit for ts in self._sets.values() if ts.ok})

    def manifest(self) -> dict[str, dict[str, str]]:
        """asset id -> role -> web filename, for the front end."""
        return {a: dict(ts.web) for a, ts in self._sets.items() if ts.ok}

    def write_manifest(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.manifest(), indent=1))
