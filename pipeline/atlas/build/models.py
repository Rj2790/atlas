"""Fetch CC0 furniture models from Poly Haven.

Parametric furniture guarantees dimensional truth, which is the point of this
project, but it reads as well-proportioned massing rather than as photographed
product. The fix is not to replace it wholesale — it is to be selective about
*where* a real mesh earns its download.

**What gets a real model.** Small, detailed, organically-shaped items: a coffee
table, a side table, an armchair, a plant, a cabinet. These are the objects whose
silhouette carries the detail a viewer reads as "real", and none of them has to hit
an exact dimension to be correct.

**What stays parametric.** Beds, wardrobes, kitchen counters and sanitaryware. Two
reasons, both decisive. Their *size is the requirement* — a queen bed must be
1524 mm and a counter 900 mm high, and a downloaded mesh scaled to fit a slot is no
longer the object it claims to be. And they are box-like anyway, so a real mesh buys
almost nothing visually.

**Why not more from Poly Haven.** Their furniture library is 85 models but it is
overwhelmingly vintage, gothic, farmhouse and industrial: `GothicBed_01`,
`painted_wooden_cabinet`, `chinese_commode`. Dropping a gothic bed into the Japandi
scheme would look *worse* than clean massing, because the failure would be one of
taste rather than of fidelity. Only the genuinely contemporary pieces are mapped.

Poly Haven assets are CC0, so no attribution is required; provenance is recorded
anyway. Files arrive as a small `.gltf` plus a separate `.bin` and texture files,
each with its own URL, so a download means reconstructing the directory layout the
glTF's relative paths expect.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["POLYHAVEN_MODELS", "ModelCache"]

_API_FILES = "https://api.polyhaven.com/files/{slug}"
_UA = "atlas-floorplan/1.0 (+https://github.com/atlas-plan)"

# Catalogue item id -> Poly Haven slug. Deliberately short; see the module docstring
# for why the big dimensional pieces are absent.
POLYHAVEN_MODELS: dict[str, str] = {
    "coffee-table": "modern_coffee_table_01",
    "bedside": "side_table_01",
    "armchair": "modern_arm_chair_01",
    "tv-unit": "modern_wooden_cabinet",
    "plant-tall": "calathea_orbifolia_01",
}


@dataclass
class ModelAsset:
    slug: str
    gltf_path: Path | None = None
    credit: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.gltf_path is not None and self.gltf_path.exists()


@dataclass
class ModelCache:
    """Downloads Poly Haven glTF models into a local cache."""

    cache_dir: Path
    resolution: str = "1k"
    """1k textures. These are accent objects a metre or two from the camera, not the
    subject of the shot, and 4k maps per side table would dwarf the whole scene."""

    timeout: int = 240
    notes: list[str] = field(default_factory=list)
    _assets: dict[str, ModelAsset] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_json(self, url: str) -> dict:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _download(self, url: str, dest: Path) -> None:
        if dest.exists() and dest.stat().st_size > 0:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = resp.read()
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(dest)

    def fetch(self, slug: str) -> ModelAsset:
        if slug in self._assets:
            return self._assets[slug]
        asset = ModelAsset(slug=slug, credit=f"{slug} — Poly Haven, CC0")
        root = self.cache_dir / slug

        # A cached copy short-circuits the API entirely.
        cached = sorted(root.glob("*.gltf"))
        if cached:
            asset.gltf_path = cached[0]
            self._assets[slug] = asset
            return asset

        try:
            files = self._get_json(_API_FILES.format(slug=slug))
            gltf = files.get("gltf", {})
            res = self.resolution if self.resolution in gltf else next(iter(gltf), None)
            if res is None:
                raise RuntimeError("no glTF variant published")
            entry = gltf[res]["gltf"]

            main = root / Path(entry["url"]).name
            self._download(entry["url"], main)
            # The .bin and textures are separate downloads whose keys are the paths
            # the glTF references, so they must land at exactly those relative paths.
            for rel, info in (entry.get("include") or {}).items():
                self._download(info["url"], root / rel)
            asset.gltf_path = main
        except Exception as exc:  # noqa: BLE001 - a missing model falls back to parametric
            asset.error = f"{type(exc).__name__}: {exc}"
            self.notes.append(f"{slug}: {asset.error}")

        self._assets[slug] = asset
        return asset

    def ensure(self, slugs: list[str], *, workers: int = 5) -> dict[str, ModelAsset]:
        import concurrent.futures

        todo = [s for s in dict.fromkeys(slugs) if s and s not in self._assets]
        if todo:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                for asset in pool.map(self.fetch, todo):
                    self._assets[asset.slug] = asset
            ok = sum(1 for s in todo if self._assets[s].ok)
            self.notes.append(f"models: {ok}/{len(todo)} Poly Haven CC0 models ready ({self.resolution})")
        return self._assets

    def get(self, slug: str | None) -> ModelAsset | None:
        if not slug:
            return None
        a = self._assets.get(slug)
        return a if a and a.ok else None

    def credits(self) -> list[str]:
        return sorted({a.credit for a in self._assets.values() if a.ok})
