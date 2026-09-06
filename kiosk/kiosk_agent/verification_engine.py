"""
kiosk_agent/verification_engine.py
-------------------------------------
Verification Engine — matches a StructuredComplaint to a local asset and
classifies evidence_status using satellite/drone image change detection.

Specification: architecture.md §4.3, implementation.md §4.4

Pipeline (per complaint):
  1. Point-in-polygon lookup against the cached asset_registry.
     The complaint's location_hint is resolved to (lat, lon) via a
     lightweight gazetteer/centroid lookup (no network calls).
     If no asset polygon contains the point → evidence_status = NO_ASSET_MATCH.

  2. If an asset is found, load its before/after imagery pair from the
     imagery table (populated by seed_demo_data.py or the Sync Daemon).
     If no imagery pair exists → evidence_status = UNAVAILABLE.

  3. Run the ONNX change-detection model (MobileNetV3-U-Net) on the
     before/after tile pair. Outputs a change_score ∈ [0, 1].
     - change_score ≥ CHANGE_THRESHOLD (0.5) → evidence_status = VERIFIED
     - change_score <  CHANGE_THRESHOLD       → evidence_status = UNAVAILABLE
       (imagery exists but no detectable change)

  4. Return VerificationResult with:
       asset_id, department_code, location_lat, location_lon,
       verification_confidence, evidence_status, evidence_image_path

Stub mode:
  When STUB_STRUCTURING=true (implies no ONNX binary), returns a
  deterministic VerificationResult whose evidence_status matches the
  demo asset (ROAD-SEG-DEMO-001 → VERIFIED with confidence 0.82).

ONNX model:
  File   : models/cv/change-detection-unet-mbv3.onnx  (~15 MB)
  Input  : ["before", "after"] — each shape (1, 3, 256, 256) float32 [0,1]
  Output : ["change_score"] — shape (1,) float32 scalar

Change threshold:
  Hard-coded at 0.5 (default). Configurable via CV_CHANGE_THRESHOLD env var
  (added to .env.example; read from settings.cv_change_threshold).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from kiosk_agent.config import settings
from kiosk_agent.db import AssetRegistryRow, ImageryRow, db_session
from kiosk_agent.schemas import (
    DepartmentCode,
    EvidenceStatus,
    StructuredComplaint,
    VerificationResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class VerificationError(Exception):
    """Base class for all Verification Engine errors."""

class VerificationModelNotReadyError(VerificationError):
    """Raised if verify() is called before load_model() and stub mode is off."""


# ---------------------------------------------------------------------------
# Module-level ONNX session state
# ---------------------------------------------------------------------------

_ort_session = None   # onnxruntime.InferenceSession or None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_SIZE = 256          # model input tile size (pixels)
CHANGE_THRESHOLD = 0.5  # change_score ≥ this → VERIFIED


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

def load_model() -> None:
    """
    Load the ONNX change-detection model into an ONNX Runtime InferenceSession.
    Must be called once at startup. No-op if already loaded or in stub mode.
    """
    global _ort_session

    if _ort_session is not None:
        logger.debug("Verification Engine: ONNX model already loaded.")
        return

    if settings.stub_structuring:
        logger.info("Verification Engine: stub mode — ONNX model not loaded.")
        return

    model_path = str(settings.cv_model_path.resolve())
    logger.info("Loading change-detection ONNX model from %s …", model_path)
    t0 = time.perf_counter()

    try:
        import onnxruntime as ort  # noqa: PLC0415

        # CPU execution provider only — Jetson uses CUDA via ort-gpu in production;
        # MVP uses CPU to keep dependencies simple.
        _ort_session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.info("Verification Engine ONNX model loaded in %d ms.", elapsed_ms)

    except FileNotFoundError:
        logger.error(
            "ONNX model not found at %s. Run scripts/fetch_models.sh.",
            model_path,
        )
        raise
    except ImportError:
        logger.error(
            "onnxruntime is not installed. Run `poetry install` inside kiosk/."
        )
        raise


def unload_model() -> None:
    """Release ONNX session. Used in tests to reset global state."""
    global _ort_session
    _ort_session = None


def is_ready() -> bool:
    """Return True if the ONNX session is loaded, or stub mode is active."""
    return settings.stub_structuring or _ort_session is not None


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _load_tile(path: str) -> "npt.NDArray":  # type: ignore[return]
    """
    Load an image tile from disk, resize to (IMG_SIZE, IMG_SIZE),
    convert to float32 array in [0, 1], shape (1, 3, H, W) — NCHW.
    """
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0   # HWC, [0,1]
    arr = arr.transpose(2, 0, 1)                     # CHW
    return arr[np.newaxis, ...]                      # NCHW (1, 3, H, W)


def _run_onnx(before_path: str, after_path: str) -> float:
    """
    Run the ONNX change-detection model on a before/after tile pair.

    Returns
    -------
    float
        change_score ∈ [0, 1]. Higher = more visible change detected.
    """
    import numpy as np  # noqa: PLC0415

    before_arr = _load_tile(before_path)
    after_arr  = _load_tile(after_path)

    input_names = [inp.name for inp in _ort_session.get_inputs()]  # type: ignore[union-attr]

    # The ONNX model expects two inputs named "before" and "after".
    # If the model's input names differ (e.g. from a different training run),
    # we fall back to positional order.
    if set(input_names) == {"before", "after"}:
        feeds = {"before": before_arr, "after": after_arr}
    else:
        feeds = {input_names[0]: before_arr, input_names[1]: after_arr}

    outputs = _ort_session.run(None, feeds)  # type: ignore[union-attr]
    change_score = float(np.squeeze(outputs[0]))
    return max(0.0, min(1.0, change_score))   # clamp to [0, 1]


# ---------------------------------------------------------------------------
# Asset registry lookup
# ---------------------------------------------------------------------------

def _point_in_polygon(lat: float, lon: float, geojson_str: str) -> bool:
    """
    Test whether (lat, lon) lies inside a GeoJSON Polygon.

    Uses the ray-casting algorithm — no external geo library required.
    GeoJSON coordinates are [lon, lat] pairs (standard).

    Returns True if the point is inside or on the boundary.
    """
    try:
        poly = json.loads(geojson_str)
        if poly.get("type") != "Polygon":
            return False
        ring: list[list[float]] = poly["coordinates"][0]
    except (json.JSONDecodeError, KeyError, IndexError):
        logger.warning("Malformed geo_polygon GeoJSON — treating as no match.")
        return False

    x, y = lon, lat   # GeoJSON uses (lon, lat) order
    n = len(ring)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _lookup_asset(
    complaint: StructuredComplaint,
) -> Optional[AssetRegistryRow]:
    """
    Find the best matching asset in the local registry for a complaint.

    Matching strategy (in order of preference):
      1. Point-in-polygon: complaint's centroid lat/lon vs all asset polygons.
         Centroid is taken from the kiosk's geofence centroid as a proxy for
         the complaint location (a real implementation would geocode
         location_hint, but that requires network or a local gazetteer —
         beyond MVP scope).
      2. Asset type match: prefer assets whose asset_type matches
         reported_asset_type from the structured complaint.

    Returns the best-matching AssetRegistryRow, or None if no match.
    """
    # Use the kiosk's geofence centroid as the complaint location proxy.
    # In Phase 5 (ticket_router) or a future gazetteer, this is replaced by
    # a geocoded coordinate from location_hint.
    complaint_lat = settings.geofence_centroid_lat
    complaint_lon = settings.geofence_centroid_lon

    with db_session() as db:
        from sqlalchemy import select  # noqa: PLC0415
        rows: list[AssetRegistryRow] = (
            db.execute(select(AssetRegistryRow)).scalars().all()
        )

    if not rows:
        logger.warning("Asset registry is empty — run seed_demo_data.py first.")
        return None

    # Candidate: in-polygon match.
    polygon_matches = [
        r for r in rows
        if _point_in_polygon(complaint_lat, complaint_lon, r.geo_polygon or "")
    ]

    # Prefer a type match within polygon matches.
    type_matches = [
        r for r in polygon_matches
        if r.asset_type == complaint.reported_asset_type
    ]

    if type_matches:
        return type_matches[0]
    if polygon_matches:
        return polygon_matches[0]

    logger.info(
        "No asset polygon contains complaint location (%.4f, %.4f) — NO_ASSET_MATCH.",
        complaint_lat, complaint_lon,
    )
    return None


def _lookup_imagery(asset_id: str) -> Optional[ImageryRow]:
    """Retrieve the most recent imagery row for an asset, or None."""
    with db_session() as db:
        from sqlalchemy import select  # noqa: PLC0415
        row = db.execute(
            select(ImageryRow)
            .where(ImageryRow.asset_id == asset_id)
            .order_by(ImageryRow.tile_version.desc())
        ).scalars().first()
    return row


# ---------------------------------------------------------------------------
# Core verification function
# ---------------------------------------------------------------------------

def verify(complaint: StructuredComplaint) -> VerificationResult:
    """
    Verify a StructuredComplaint against the local asset registry and
    imagery store.

    Parameters
    ----------
    complaint : StructuredComplaint
        Output of the Structuring Engine.

    Returns
    -------
    VerificationResult
        Contains asset_id, department_code, location, verification_confidence,
        evidence_status, and evidence_image_path.

    Raises
    ------
    VerificationModelNotReadyError
        If ONNX model is not loaded and stub mode is off.
    VerificationError
        On unrecoverable internal error during ONNX inference.
    """
    # ── Stub mode ─────────────────────────────────────────────────────────
    if settings.stub_structuring:
        return _stub_verify(complaint)

    # ── Guard ─────────────────────────────────────────────────────────────
    if _ort_session is None:
        raise VerificationModelNotReadyError(
            "Verification Engine ONNX model is not loaded. "
            "Call load_model() at startup."
        )

    # ── Step 1: Asset lookup ───────────────────────────────────────────────
    asset = _lookup_asset(complaint)
    if asset is None:
        return VerificationResult(
            asset_id=None,
            department_code=_category_to_dept(complaint),
            location_lat=settings.geofence_centroid_lat,
            location_lon=settings.geofence_centroid_lon,
            verification_confidence=None,
            evidence_status=EvidenceStatus.NO_ASSET_MATCH,
            evidence_image_path=None,
        )

    dept_code = DepartmentCode(asset.department_code)

    # ── Step 2: Imagery lookup ────────────────────────────────────────────
    imagery = _lookup_imagery(asset.asset_id)
    if imagery is None:
        logger.info(
            "No imagery for asset %s — evidence_status=UNAVAILABLE.", asset.asset_id
        )
        return VerificationResult(
            asset_id=asset.asset_id,
            department_code=dept_code,
            location_lat=asset.centroid_lat,
            location_lon=asset.centroid_lon,
            verification_confidence=None,
            evidence_status=EvidenceStatus.UNAVAILABLE,
            evidence_image_path=None,
        )

    # ── Step 3: ONNX change-detection ─────────────────────────────────────
    logger.info(
        "Running ONNX change detection: asset=%s  before=%s  after=%s",
        asset.asset_id, imagery.before_path, imagery.after_path,
    )
    t0 = time.perf_counter()
    try:
        change_score = _run_onnx(imagery.before_path, imagery.after_path)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "ONNX inference failed for asset %s: %s",
            asset.asset_id, exc, exc_info=True,
        )
        raise VerificationError(
            f"ONNX inference failed for asset {asset.asset_id}: {exc}"
        ) from exc

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    threshold = getattr(settings, "cv_change_threshold", CHANGE_THRESHOLD)

    logger.info(
        "Change detection: asset=%s  score=%.4f  threshold=%.2f  elapsed_ms=%d",
        asset.asset_id, change_score, threshold, elapsed_ms,
    )

    # ── Step 4: Classify evidence_status ──────────────────────────────────
    if change_score >= threshold:
        evidence_status = EvidenceStatus.VERIFIED
        evidence_image_path = imagery.after_path
        confidence = round(change_score, 4)
    else:
        evidence_status = EvidenceStatus.UNAVAILABLE
        evidence_image_path = None
        confidence = None

    return VerificationResult(
        asset_id=asset.asset_id,
        department_code=dept_code,
        location_lat=asset.centroid_lat,
        location_lon=asset.centroid_lon,
        verification_confidence=confidence,
        evidence_status=evidence_status,
        evidence_image_path=evidence_image_path,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _category_to_dept(complaint: StructuredComplaint) -> DepartmentCode:
    """Map complaint category to a DepartmentCode (fallback if no asset found)."""
    from kiosk_agent.schemas import CATEGORY_TO_DEPARTMENT  # noqa: PLC0415
    return CATEGORY_TO_DEPARTMENT.get(complaint.category, DepartmentCode.GEN)


# ---------------------------------------------------------------------------
# Stub verification
# ---------------------------------------------------------------------------

def _stub_verify(complaint: StructuredComplaint) -> VerificationResult:
    """
    Returns a deterministic VerificationResult for the demo.
    ROAD complaints → VERIFIED (matches ROAD-SEG-DEMO-001 with imagery).
    All other categories → UNAVAILABLE.
    This gives the confirmation screen realistic variation for the demo.
    """
    from kiosk_agent.schemas import Category  # noqa: PLC0415

    dept_code = _category_to_dept(complaint)

    if complaint.category == Category.ROAD:
        evidence_image = str(
            Path(_DEMO_IMAGERY_ROOT) / "ROAD-SEG-DEMO-001" / "after.png"
        )
        return VerificationResult(
            asset_id="ROAD-SEG-DEMO-001",
            department_code=dept_code,
            location_lat=18.515,
            location_lon=73.855,
            verification_confidence=0.82,
            evidence_status=EvidenceStatus.VERIFIED,
            evidence_image_path=evidence_image if Path(evidence_image).exists() else None,
        )

    return VerificationResult(
        asset_id=f"{complaint.category.value}-DEMO-001",
        department_code=dept_code,
        location_lat=settings.geofence_centroid_lat,
        location_lon=settings.geofence_centroid_lon,
        verification_confidence=None,
        evidence_status=EvidenceStatus.UNAVAILABLE,
        evidence_image_path=None,
    )


_DEMO_IMAGERY_ROOT = str(
    Path(__file__).parent.parent / "models" / "imagery" / "demo"
)
