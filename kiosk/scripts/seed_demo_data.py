"""
scripts/seed_demo_data.py
--------------------------
Seeds the local SQLite database with:
  1. Demo Asset Registry (10 municipal assets with geo-polygons)
  2. Demo Imagery Store (synthetic before/after PNG tile pairs per asset)

Run once before the first kiosk demo:
  poetry run python scripts/seed_demo_data.py

Safe to re-run — existing rows are upserted (INSERT OR REPLACE).

Demo geofence: small area near Pune, Maharashtra (73.85–73.86°E, 18.51–18.52°N)
  — matches the default KIOSK_GEOFENCE_GEOJSON in .env.example.

Asset coverage for acceptance criteria (implementation.md §14):
  - At least one asset WITH before/after imagery → evidence_status "verified"
  - At least one asset WITHOUT imagery → evidence_status "unavailable"
  - At least one complaint with no asset match possible → evidence_status "no_asset_match"
    (achieved by a complaint whose location_hint resolves outside all asset polygons)

Imagery format:
  Synthetic PNGs are generated using Pillow — simple coloured rectangles
  with text labels. "Before" images are light grey; "After" images show a
  coloured patch simulating a visible change (pothole, flooded area, etc.).
  The ONNX change-detection model will see a genuine pixel delta between them.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Allow running directly: `python scripts/seed_demo_data.py`
_KIOSK_ROOT = Path(__file__).parent.parent.resolve()
if str(_KIOSK_ROOT) not in sys.path:
    sys.path.insert(0, str(_KIOSK_ROOT))

from kiosk_agent.db import (
    AssetRegistryRow,
    Base,
    ImageryRow,
    db_session,
    get_engine,
    init_db,
)
from kiosk_agent.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGERY_DIR = _KIOSK_ROOT / "models" / "imagery" / "demo"
REGISTRY_VERSION = 1
SEEDED_AT = datetime.now(UTC).isoformat()

# Image tile size (pixels) — small enough to keep the demo DB lightweight.
TILE_W, TILE_H = 256, 256

# ---------------------------------------------------------------------------
# Demo asset definitions
# Each asset sits inside the default geofence (73.85–73.86°E, 18.51–18.52°N).
# Polygons are small squares (~100m × 100m) centred on each asset.
# ---------------------------------------------------------------------------

def _box(lon_c: float, lat_c: float, d: float = 0.0005) -> str:
    """
    Return a GeoJSON Polygon string for a square of half-side `d` degrees
    centred at (lon_c, lat_c).  d=0.0005° ≈ 55m at 18°N latitude.
    """
    coords = [
        [lon_c - d, lat_c - d],
        [lon_c + d, lat_c - d],
        [lon_c + d, lat_c + d],
        [lon_c - d, lat_c + d],
        [lon_c - d, lat_c - d],  # close ring
    ]
    return json.dumps({"type": "Polygon", "coordinates": [coords]})


DEMO_ASSETS: list[dict] = [
    # ── With imagery (change detectable) ─────────────────────────────────
    {
        "asset_id":        "ROAD-SEG-DEMO-001",
        "asset_type":      "road_segment",
        "department_code": "ROAD",
        "district_code":   "PUNE",
        "asset_label":     "Main Road near Community Centre",
        "centroid_lat":    18.515,
        "centroid_lon":    73.855,
        "geo_polygon":     _box(73.855, 18.515),
        "has_imagery":     True,
        "change_type":     "pothole",     # colour patch in 'after' image
    },
    {
        "asset_id":        "WATER-PUMP-DEMO-001",
        "asset_type":      "handpump",
        "department_code": "WATER",
        "district_code":   "PUNE",
        "asset_label":     "Handpump at Village Square",
        "centroid_lat":    18.513,
        "centroid_lon":    73.852,
        "geo_polygon":     _box(73.852, 18.513),
        "has_imagery":     True,
        "change_type":     "water_leak",
    },
    {
        "asset_id":        "LIGHT-POLE-DEMO-001",
        "asset_type":      "streetlight_pole",
        "department_code": "LIGHT",
        "district_code":   "PUNE",
        "asset_label":     "Street Light Pole near Bus Stop",
        "centroid_lat":    18.516,
        "centroid_lon":    73.858,
        "geo_polygon":     _box(73.858, 18.516),
        "has_imagery":     True,
        "change_type":     "dark_area",
    },
    {
        "asset_id":        "SANI-BIN-DEMO-001",
        "asset_type":      "garbage_bin",
        "department_code": "SANI",
        "district_code":   "PUNE",
        "asset_label":     "Garbage Bin at Market",
        "centroid_lat":    18.514,
        "centroid_lon":    73.856,
        "geo_polygon":     _box(73.856, 18.514),
        "has_imagery":     True,
        "change_type":     "garbage_pile",
    },
    {
        "asset_id":        "ELEC-BOX-DEMO-001",
        "asset_type":      "transformer_box",
        "department_code": "ELEC",
        "district_code":   "PUNE",
        "asset_label":     "Electricity Transformer near School",
        "centroid_lat":    18.518,
        "centroid_lon":    73.853,
        "geo_polygon":     _box(73.853, 18.518),
        "has_imagery":     True,
        "change_type":     "burn_mark",
    },
    # ── Without imagery (evidence_status = "unavailable") ────────────────
    {
        "asset_id":        "ROAD-SEG-DEMO-002",
        "asset_type":      "road_segment",
        "department_code": "ROAD",
        "district_code":   "PUNE",
        "asset_label":     "Side Lane near Temple",
        "centroid_lat":    18.511,
        "centroid_lon":    73.854,
        "geo_polygon":     _box(73.854, 18.511),
        "has_imagery":     False,
        "change_type":     None,
    },
    {
        "asset_id":        "WATER-PIPE-DEMO-001",
        "asset_type":      "pipeline_junction",
        "department_code": "WATER",
        "district_code":   "PUNE",
        "asset_label":     "Water Pipeline Junction near Fields",
        "centroid_lat":    18.519,
        "centroid_lon":    73.857,
        "geo_polygon":     _box(73.857, 18.519),
        "has_imagery":     False,
        "change_type":     None,
    },
    {
        "asset_id":        "SANI-DRAIN-DEMO-001",
        "asset_type":      "drain",
        "department_code": "SANI",
        "district_code":   "PUNE",
        "asset_label":     "Open Drain near Residential Area",
        "centroid_lat":    18.512,
        "centroid_lon":    73.859,
        "geo_polygon":     _box(73.859, 18.512),
        "has_imagery":     False,
        "change_type":     None,
    },
    {
        "asset_id":        "LIGHT-POLE-DEMO-002",
        "asset_type":      "streetlight_pole",
        "department_code": "LIGHT",
        "district_code":   "PUNE",
        "asset_label":     "Street Light Pole at Park Entrance",
        "centroid_lat":    18.517,
        "centroid_lon":    73.851,
        "geo_polygon":     _box(73.851, 18.517),
        "has_imagery":     False,
        "change_type":     None,
    },
    {
        "asset_id":        "ELEC-BOX-DEMO-002",
        "asset_type":      "distribution_panel",
        "department_code": "ELEC",
        "district_code":   "PUNE",
        "asset_label":     "Distribution Panel near Panchayat Office",
        "centroid_lat":    18.515,
        "centroid_lon":    73.860,
        "geo_polygon":     _box(73.860, 18.515),
        "has_imagery":     False,
        "change_type":     None,
    },
]


# ---------------------------------------------------------------------------
# Synthetic image generation
# ---------------------------------------------------------------------------

# Change type → RGB colour of the visible change patch in the 'after' image.
CHANGE_COLOURS: dict[str, tuple[int, int, int]] = {
    "pothole":      (60,  60,  60),   # dark grey (broken asphalt)
    "water_leak":   (30, 100, 200),   # blue (standing water)
    "dark_area":    (10,  10,  10),   # near-black (unlit area)
    "garbage_pile": (120, 80,  40),   # brown (refuse heap)
    "burn_mark":    (40,  20,   0),   # very dark brown (scorching)
}


def _make_before_image(asset_label: str) -> "Image":  # type: ignore[return]
    """
    Generate a synthetic 'before' tile: uniform light-grey background
    with an asset label overlaid in dark text.
    """
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

    img = Image.new("RGB", (TILE_W, TILE_H), color=(210, 210, 210))
    draw = ImageDraw.Draw(img)

    # Draw a simple asset outline (white rectangle).
    margin = 30
    draw.rectangle(
        [margin, margin, TILE_W - margin, TILE_H - margin],
        outline=(180, 180, 180),
        width=2,
    )
    # Label — use default font (no external font required).
    draw.text((10, 10), f"BEFORE\n{asset_label[:20]}", fill=(80, 80, 80))
    return img


def _make_after_image(asset_label: str, change_type: str) -> "Image":  # type: ignore[return]
    """
    Generate a synthetic 'after' tile: same background as 'before' but
    with a coloured change patch (30×30 px) in the centre — simulating
    a visible defect that the change-detection model will score > threshold.
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415

    img = Image.new("RGB", (TILE_W, TILE_H), color=(210, 210, 210))
    draw = ImageDraw.Draw(img)

    margin = 30
    draw.rectangle(
        [margin, margin, TILE_W - margin, TILE_H - margin],
        outline=(180, 180, 180),
        width=2,
    )

    # Change patch — centred, 60×60 px.
    change_colour = CHANGE_COLOURS.get(change_type, (100, 100, 100))
    cx, cy = TILE_W // 2, TILE_H // 2
    patch = 30
    draw.rectangle(
        [cx - patch, cy - patch, cx + patch, cy + patch],
        fill=change_colour,
    )

    draw.text((10, 10), f"AFTER\n{asset_label[:20]}", fill=(80, 80, 80))
    return img


def generate_imagery(assets: list[dict]) -> None:
    """
    Generate and save synthetic before/after PNG tiles for assets that
    have has_imagery=True. Tiles are saved under:
      models/imagery/demo/{asset_id}/before.png
      models/imagery/demo/{asset_id}/after.png
    """
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        logger.error(
            "Pillow is not installed — cannot generate imagery. "
            "Run `poetry install` and retry."
        )
        return

    IMAGERY_DIR.mkdir(parents=True, exist_ok=True)

    for asset in assets:
        if not asset.get("has_imagery"):
            continue

        asset_dir = IMAGERY_DIR / asset["asset_id"]
        asset_dir.mkdir(exist_ok=True)

        before_path = asset_dir / "before.png"
        after_path  = asset_dir / "after.png"

        _make_before_image(asset["asset_label"]).save(before_path, format="PNG")
        _make_after_image(asset["asset_label"], asset["change_type"]).save(
            after_path, format="PNG"
        )
        logger.info(
            "  Imagery → %s  (before=%s, after=%s)",
            asset["asset_id"],
            before_path.name,
            after_path.name,
        )


# ---------------------------------------------------------------------------
# Database seeding
# ---------------------------------------------------------------------------

def seed_asset_registry(assets: list[dict]) -> None:
    """
    Upsert demo assets into the asset_registry table.
    Uses raw SQL INSERT OR REPLACE for compatibility with plain SQLite
    (no SQLAlchemy merge needed for this seed script).
    """
    from sqlalchemy import text  # noqa: PLC0415

    with db_session() as db:
        for asset in assets:
            db.execute(
                text("""
                    INSERT OR REPLACE INTO asset_registry
                      (asset_id, asset_type, department_code, district_code,
                       geo_polygon, asset_label, centroid_lat, centroid_lon,
                       registry_version, last_synced_at)
                    VALUES
                      (:asset_id, :asset_type, :department_code, :district_code,
                       :geo_polygon, :asset_label, :centroid_lat, :centroid_lon,
                       :registry_version, :last_synced_at)
                """),
                {
                    "asset_id":        asset["asset_id"],
                    "asset_type":      asset["asset_type"],
                    "department_code": asset["department_code"],
                    "district_code":   asset["district_code"],
                    "geo_polygon":     asset["geo_polygon"],
                    "asset_label":     asset["asset_label"],
                    "centroid_lat":    asset["centroid_lat"],
                    "centroid_lon":    asset["centroid_lon"],
                    "registry_version":REGISTRY_VERSION,
                    "last_synced_at":  SEEDED_AT,
                },
            )
        logger.info("  Asset registry: %d assets upserted.", len(assets))


def seed_imagery_table(assets: list[dict]) -> None:
    """
    Upsert imagery rows into the imagery table for assets that have
    has_imagery=True and whose PNG files have been generated.
    """
    from sqlalchemy import text  # noqa: PLC0415

    with db_session() as db:
        count = 0
        for asset in assets:
            if not asset.get("has_imagery"):
                continue

            asset_dir = IMAGERY_DIR / asset["asset_id"]
            before_path = asset_dir / "before.png"
            after_path  = asset_dir / "after.png"

            if not before_path.exists() or not after_path.exists():
                logger.warning(
                    "  Imagery files missing for %s — skipping imagery row.",
                    asset["asset_id"],
                )
                continue

            db.execute(
                text("""
                    INSERT OR REPLACE INTO imagery
                      (asset_id, before_path, after_path, tile_version, captured_at)
                    VALUES
                      (:asset_id, :before_path, :after_path, :tile_version, :captured_at)
                """),
                {
                    "asset_id":    asset["asset_id"],
                    "before_path": str(before_path),
                    "after_path":  str(after_path),
                    "tile_version":REGISTRY_VERSION,
                    "captured_at": SEEDED_AT,
                },
            )
            count += 1
        logger.info("  Imagery table: %d rows upserted.", count)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 60)
    logger.info("GroundTruth — Demo Data Seeder")
    logger.info("DB path : %s", settings.db_path.resolve())
    logger.info("Imagery : %s", IMAGERY_DIR)
    logger.info("=" * 60)

    # 1. Ensure schema exists.
    init_db()
    logger.info("Schema initialised.")

    # 2. Generate synthetic imagery PNG files.
    logger.info("Generating synthetic imagery tiles…")
    generate_imagery(DEMO_ASSETS)

    # 3. Seed asset registry.
    logger.info("Seeding asset registry…")
    seed_asset_registry(DEMO_ASSETS)

    # 4. Seed imagery table.
    logger.info("Seeding imagery table…")
    seed_imagery_table(DEMO_ASSETS)

    # 5. Summary.
    total   = len(DEMO_ASSETS)
    with_img= sum(1 for a in DEMO_ASSETS if a["has_imagery"])
    without = total - with_img

    logger.info("=" * 60)
    logger.info("Seeding complete.")
    logger.info("  Total assets       : %d", total)
    logger.info("  With imagery       : %d  (evidence_status='verified' achievable)", with_img)
    logger.info("  Without imagery    : %d  (evidence_status='unavailable')", without)
    logger.info(
        "  No-match demo      : complaints whose location_hint resolves outside "
        "all polygons will get evidence_status='no_asset_match'."
    )
    logger.info("=" * 60)
    logger.info("Run `poetry run kiosk-agent` to start the kiosk.")


if __name__ == "__main__":
    main()
