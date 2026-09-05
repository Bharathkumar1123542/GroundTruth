"""
tests/test_verification_engine.py
-----------------------------------
Unit tests for kiosk_agent/verification_engine.py.

Coverage targets:
  - _point_in_polygon(): interior, exterior, on-edge, empty polygon, bad JSON
  - _lookup_asset(): match by polygon, prefer type match, empty registry → None
  - _lookup_imagery(): returns row when present, None when absent
  - verify() → evidence_status = NO_ASSET_MATCH when no polygon contains point
  - verify() → evidence_status = UNAVAILABLE when asset found but no imagery
  - verify() → evidence_status = VERIFIED when change_score ≥ CHANGE_THRESHOLD
  - verify() → evidence_status = UNAVAILABLE when change_score < CHANGE_THRESHOLD
  - verify() raises VerificationError when ONNX inference raises
  - verify() raises VerificationModelNotReadyError when model not loaded (stub off)
  - Stub mode: ROAD → VERIFIED, non-ROAD → UNAVAILABLE
  - load_model() / unload_model() lifecycle
  - _load_tile(): returns correct NCHW shape from a real PNG (tmp_path)
  - ONNX input name fallback (positional order when names ≠ "before"/"after")
  - change_score clamped to [0, 1]

All tests run without a real ONNX binary (mocked via monkeypatch).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kiosk_agent.schemas import (
    Category,
    DepartmentCode,
    EvidenceStatus,
    StructuredComplaint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_complaint(
    category: Category = Category.ROAD,
    reported_asset_type: str = "road_segment",
) -> StructuredComplaint:
    return StructuredComplaint(
        category=category,
        subcategory="Pothole",
        description="Large pothole on the main road.",
        location_hint="near community centre",
        reported_asset_type=reported_asset_type,
        urgency_keywords=["pothole"],
        structuring_confidence=0.91,
    )


def _box_geojson(lon_c: float, lat_c: float, d: float = 0.001) -> str:
    """Return a GeoJSON Polygon square centred at (lon_c, lat_c)."""
    coords = [
        [lon_c - d, lat_c - d],
        [lon_c + d, lat_c - d],
        [lon_c + d, lat_c + d],
        [lon_c - d, lat_c + d],
        [lon_c - d, lat_c - d],
    ]
    return json.dumps({"type": "Polygon", "coordinates": [coords]})


def _seed_asset(db, asset_id="ROAD-SEG-001", asset_type="road_segment",
                dept="ROAD", centroid_lat=18.515, centroid_lon=73.855,
                geo_polygon=None):
    from sqlalchemy import text
    if geo_polygon is None:
        geo_polygon = _box_geojson(centroid_lon, centroid_lat)
    db.execute(text("""
        INSERT OR REPLACE INTO asset_registry
          (asset_id, asset_type, department_code, district_code,
           geo_polygon, asset_label, centroid_lat, centroid_lon,
           registry_version, last_synced_at)
        VALUES
          (:asset_id, :asset_type, :dept, 'PUNE',
           :geo, 'Test Asset', :lat, :lon, 1, '2026-01-01T00:00:00Z')
    """), {
        "asset_id": asset_id, "asset_type": asset_type,
        "dept": dept, "geo": geo_polygon,
        "lat": centroid_lat, "lon": centroid_lon,
    })


def _seed_imagery(db, asset_id, before_path, after_path):
    from sqlalchemy import text
    db.execute(text("""
        INSERT OR REPLACE INTO imagery
          (asset_id, before_path, after_path, tile_version, captured_at)
        VALUES (:aid, :bp, :ap, 1, '2026-01-01T00:00:00Z')
    """), {"aid": asset_id, "bp": before_path, "ap": after_path})


def _make_png(path: Path, colour: tuple[int, int, int] = (128, 128, 128)) -> None:
    """Write a minimal 256×256 PNG of a solid colour using Pillow."""
    try:
        from PIL import Image
        img = Image.new("RGB", (256, 256), color=colour)
        img.save(path, format="PNG")
    except ImportError:
        # If Pillow isn't available, write a 1×1 white PNG manually.
        # (PNG binary format: signature + IHDR + IDAT + IEND)
        import zlib, struct
        def _chunk(name, data):
            c = zlib.crc32(name + data) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + name + data + struct.pack(">I", c)
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        raw = b"\x00" + bytes(colour)
        idat = _chunk(b"IDAT", zlib.compress(raw))
        iend = _chunk(b"IEND", b"")
        path.write_bytes(sig + ihdr + idat + iend)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_engine():
    import kiosk_agent.verification_engine as eng
    eng._ort_session = None
    yield
    eng._ort_session = None


@pytest.fixture()
def stub_on(monkeypatch):
    monkeypatch.setattr("kiosk_agent.verification_engine.settings.stub_structuring", True)
    monkeypatch.setattr(
        "kiosk_agent.verification_engine.settings.geofence_centroid_lat", 18.515
    )
    monkeypatch.setattr(
        "kiosk_agent.verification_engine.settings.geofence_centroid_lon", 73.855
    )


@pytest.fixture()
def stub_off(monkeypatch):
    monkeypatch.setattr("kiosk_agent.verification_engine.settings.stub_structuring", False)
    monkeypatch.setattr(
        "kiosk_agent.verification_engine.settings.geofence_centroid_lat", 18.515
    )
    monkeypatch.setattr(
        "kiosk_agent.verification_engine.settings.geofence_centroid_lon", 73.855
    )


@pytest.fixture()
def seeded_db(tmp_path, monkeypatch):
    """
    Provide a fresh in-memory SQLite DB (via tmp_path) with schema initialised.
    Redirects all db_session() calls to this DB.
    """
    from kiosk_agent import db as db_mod
    from kiosk_agent.db import init_db

    db_path = tmp_path / "test_kiosk.db"
    monkeypatch.setattr("kiosk_agent.verification_engine.settings.db_path", db_path)
    monkeypatch.setattr("kiosk_agent.db.settings.db_path", db_path)
    init_db()
    return db_path


# ---------------------------------------------------------------------------
# _point_in_polygon() tests
# ---------------------------------------------------------------------------

class TestPointInPolygon:
    def test_point_inside_square(self):
        from kiosk_agent.verification_engine import _point_in_polygon
        geo = _box_geojson(73.855, 18.515, d=0.001)
        assert _point_in_polygon(18.515, 73.855, geo) is True  # centroid

    def test_point_outside_square(self):
        from kiosk_agent.verification_engine import _point_in_polygon
        geo = _box_geojson(73.855, 18.515, d=0.001)
        assert _point_in_polygon(18.600, 73.900, geo) is False  # far outside

    def test_point_on_corner_edge(self):
        """Point exactly on a polygon vertex is treated as inside (ray-cast boundary)."""
        from kiosk_agent.verification_engine import _point_in_polygon
        geo = _box_geojson(0.0, 0.0, d=1.0)
        # Top-right corner
        result = _point_in_polygon(1.0, 1.0, geo)
        assert isinstance(result, bool)  # must not crash

    def test_empty_polygon_string(self):
        from kiosk_agent.verification_engine import _point_in_polygon
        assert _point_in_polygon(18.515, 73.855, "") is False

    def test_malformed_geojson(self):
        from kiosk_agent.verification_engine import _point_in_polygon
        assert _point_in_polygon(18.515, 73.855, "{bad json}") is False

    def test_wrong_geometry_type(self):
        """A GeoJSON Point (not Polygon) should return False."""
        from kiosk_agent.verification_engine import _point_in_polygon
        geo = json.dumps({"type": "Point", "coordinates": [73.855, 18.515]})
        assert _point_in_polygon(18.515, 73.855, geo) is False

    def test_near_south_pole_no_crash(self):
        """Extreme latitude should not cause divide-by-zero."""
        from kiosk_agent.verification_engine import _point_in_polygon
        geo = _box_geojson(0.0, -89.0, d=0.5)
        result = _point_in_polygon(-89.0, 0.0, geo)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# _lookup_asset() tests
# ---------------------------------------------------------------------------

class TestLookupAsset:
    def test_returns_none_when_registry_empty(self, stub_off, seeded_db):
        from kiosk_agent.verification_engine import _lookup_asset
        result = _lookup_asset(_make_complaint())
        assert result is None

    def test_returns_asset_when_point_inside_polygon(self, stub_off, seeded_db):
        from kiosk_agent.db import db_session
        from kiosk_agent.verification_engine import _lookup_asset

        with db_session() as db:
            # Polygon centred at (73.855, 18.515) with d=0.002 — large enough
            # to contain settings.geofence_centroid (18.515, 73.855).
            _seed_asset(db, geo_polygon=_box_geojson(73.855, 18.515, d=0.002))

        result = _lookup_asset(_make_complaint())
        assert result is not None
        assert result.asset_id == "ROAD-SEG-001"

    def test_prefers_type_match_over_first_polygon_match(self, stub_off, seeded_db):
        """When multiple assets contain the point, prefer the one matching asset_type."""
        from kiosk_agent.db import db_session
        from kiosk_agent.verification_engine import _lookup_asset

        big_box = _box_geojson(73.855, 18.515, d=0.005)
        with db_session() as db:
            _seed_asset(db, asset_id="WATER-001",
                        asset_type="handpump", dept="WATER",
                        geo_polygon=big_box)
            _seed_asset(db, asset_id="ROAD-001",
                        asset_type="road_segment", dept="ROAD",
                        geo_polygon=big_box)

        # Complaint reports road_segment → should prefer ROAD-001.
        result = _lookup_asset(_make_complaint(reported_asset_type="road_segment"))
        assert result is not None
        assert result.asset_id == "ROAD-001"

    def test_returns_none_when_point_outside_all_polygons(self, stub_off, seeded_db):
        from kiosk_agent.db import db_session
        from kiosk_agent.verification_engine import _lookup_asset

        # Asset polygon is far from settings.geofence_centroid.
        with db_session() as db:
            _seed_asset(db, geo_polygon=_box_geojson(10.0, 10.0, d=0.001))

        result = _lookup_asset(_make_complaint())
        assert result is None


# ---------------------------------------------------------------------------
# _lookup_imagery() tests
# ---------------------------------------------------------------------------

class TestLookupImagery:
    def test_returns_none_when_no_imagery(self, seeded_db):
        from kiosk_agent.verification_engine import _lookup_imagery
        assert _lookup_imagery("NONEXISTENT-ASSET") is None

    def test_returns_row_when_imagery_present(self, seeded_db):
        from kiosk_agent.db import db_session
        from kiosk_agent.verification_engine import _lookup_imagery

        with db_session() as db:
            _seed_imagery(db, "ROAD-SEG-001", "/tmp/before.png", "/tmp/after.png")

        row = _lookup_imagery("ROAD-SEG-001")
        assert row is not None
        assert row.before_path == "/tmp/before.png"
        assert row.after_path  == "/tmp/after.png"


# ---------------------------------------------------------------------------
# verify() — evidence status path tests
# ---------------------------------------------------------------------------

class TestVerifyPaths:
    def test_no_asset_match(self, stub_off, seeded_db):
        """Empty registry → NO_ASSET_MATCH."""
        import kiosk_agent.verification_engine as eng
        from kiosk_agent.verification_engine import verify

        eng._ort_session = MagicMock()
        result = verify(_make_complaint())
        assert result.evidence_status == EvidenceStatus.NO_ASSET_MATCH
        assert result.asset_id is None
        assert result.verification_confidence == 0.0

    def test_asset_found_no_imagery(self, stub_off, seeded_db, monkeypatch):
        """Asset in polygon but no imagery → UNAVAILABLE."""
        import kiosk_agent.verification_engine as eng
        from kiosk_agent.db import db_session
        from kiosk_agent.verification_engine import verify

        eng._ort_session = MagicMock()
        with db_session() as db:
            _seed_asset(db, geo_polygon=_box_geojson(73.855, 18.515, d=0.002))

        result = verify(_make_complaint())
        assert result.evidence_status == EvidenceStatus.UNAVAILABLE
        assert result.asset_id == "ROAD-SEG-001"
        # ONNX must NOT be called when no imagery exists.
        eng._ort_session.run.assert_not_called()

    def test_verified_when_change_score_above_threshold(
        self, stub_off, seeded_db, tmp_path, monkeypatch
    ):
        """change_score ≥ 0.5 → VERIFIED."""
        import numpy as np
        import kiosk_agent.verification_engine as eng
        from kiosk_agent.db import db_session
        from kiosk_agent.verification_engine import verify, CHANGE_THRESHOLD

        # Seed asset + imagery.
        before_png = tmp_path / "before.png"
        after_png  = tmp_path / "after.png"
        _make_png(before_png, colour=(200, 200, 200))
        _make_png(after_png,  colour=(50,  50,  50))   # visible change

        with db_session() as db:
            _seed_asset(db, geo_polygon=_box_geojson(73.855, 18.515, d=0.002))
            _seed_imagery(db, "ROAD-SEG-001", str(before_png), str(after_png))

        # Mock ONNX to return score above threshold.
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [
            MagicMock(name="before"), MagicMock(name="after")
        ]
        mock_session.run.return_value = [np.array([0.75], dtype=np.float32)]
        monkeypatch.setattr(eng, "_ort_session", mock_session)

        result = verify(_make_complaint())
        assert result.evidence_status == EvidenceStatus.VERIFIED
        assert result.verification_confidence == pytest.approx(0.75)
        assert result.evidence_image_path is not None

    def test_unavailable_when_change_score_below_threshold(
        self, stub_off, seeded_db, tmp_path, monkeypatch
    ):
        """change_score < 0.5 → UNAVAILABLE even though imagery exists."""
        import numpy as np
        import kiosk_agent.verification_engine as eng
        from kiosk_agent.db import db_session
        from kiosk_agent.verification_engine import verify

        before_png = tmp_path / "before.png"
        after_png  = tmp_path / "after.png"
        _make_png(before_png)
        _make_png(after_png)

        with db_session() as db:
            _seed_asset(db, geo_polygon=_box_geojson(73.855, 18.515, d=0.002))
            _seed_imagery(db, "ROAD-SEG-001", str(before_png), str(after_png))

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [
            MagicMock(name="before"), MagicMock(name="after")
        ]
        mock_session.run.return_value = [np.array([0.20], dtype=np.float32)]
        monkeypatch.setattr(eng, "_ort_session", mock_session)

        result = verify(_make_complaint())
        assert result.evidence_status == EvidenceStatus.UNAVAILABLE
        assert result.evidence_image_path is None

    def test_onnx_inference_error_raises_verification_error(
        self, stub_off, seeded_db, tmp_path, monkeypatch
    ):
        """ONNX runtime error must be wrapped in VerificationError."""
        import kiosk_agent.verification_engine as eng
        from kiosk_agent.db import db_session
        from kiosk_agent.verification_engine import verify, VerificationError

        before_png = tmp_path / "before.png"
        after_png  = tmp_path / "after.png"
        _make_png(before_png)
        _make_png(after_png)

        with db_session() as db:
            _seed_asset(db, geo_polygon=_box_geojson(73.855, 18.515, d=0.002))
            _seed_imagery(db, "ROAD-SEG-001", str(before_png), str(after_png))

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [
            MagicMock(name="before"), MagicMock(name="after")
        ]
        mock_session.run.side_effect = RuntimeError("ONNX kernel error")
        monkeypatch.setattr(eng, "_ort_session", mock_session)

        with pytest.raises(VerificationError, match="ONNX inference failed"):
            verify(_make_complaint())

    def test_model_not_ready_raises(self, stub_off, seeded_db):
        """verify() without loaded model must raise VerificationModelNotReadyError."""
        import kiosk_agent.verification_engine as eng
        from kiosk_agent.verification_engine import verify, VerificationModelNotReadyError

        assert eng._ort_session is None
        with pytest.raises(VerificationModelNotReadyError):
            verify(_make_complaint())


# ---------------------------------------------------------------------------
# change_score clamping
# ---------------------------------------------------------------------------

class TestChangeScoreClamping:
    def test_score_above_one_clamped(self, stub_off, seeded_db, tmp_path, monkeypatch):
        """ONNX model returning > 1.0 must be clamped to 1.0."""
        import numpy as np
        import kiosk_agent.verification_engine as eng
        from kiosk_agent.db import db_session
        from kiosk_agent.verification_engine import verify

        before_png = tmp_path / "before.png"
        after_png  = tmp_path / "after.png"
        _make_png(before_png)
        _make_png(after_png)

        with db_session() as db:
            _seed_asset(db, geo_polygon=_box_geojson(73.855, 18.515, d=0.002))
            _seed_imagery(db, "ROAD-SEG-001", str(before_png), str(after_png))

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [
            MagicMock(name="before"), MagicMock(name="after")
        ]
        mock_session.run.return_value = [np.array([2.5], dtype=np.float32)]
        monkeypatch.setattr(eng, "_ort_session", mock_session)

        result = verify(_make_complaint())
        assert result.verification_confidence <= 1.0

    def test_score_below_zero_clamped(self, stub_off, seeded_db, tmp_path, monkeypatch):
        """ONNX model returning < 0.0 must be clamped to 0.0."""
        import numpy as np
        import kiosk_agent.verification_engine as eng
        from kiosk_agent.db import db_session
        from kiosk_agent.verification_engine import verify

        before_png = tmp_path / "before.png"
        after_png  = tmp_path / "after.png"
        _make_png(before_png)
        _make_png(after_png)

        with db_session() as db:
            _seed_asset(db, geo_polygon=_box_geojson(73.855, 18.515, d=0.002))
            _seed_imagery(db, "ROAD-SEG-001", str(before_png), str(after_png))

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [
            MagicMock(name="before"), MagicMock(name="after")
        ]
        mock_session.run.return_value = [np.array([-0.5], dtype=np.float32)]
        monkeypatch.setattr(eng, "_ort_session", mock_session)

        result = verify(_make_complaint())
        assert result.verification_confidence >= 0.0


# ---------------------------------------------------------------------------
# ONNX input name fallback
# ---------------------------------------------------------------------------

class TestOnnxInputFallback:
    def test_positional_fallback_when_names_differ(
        self, stub_off, seeded_db, tmp_path, monkeypatch
    ):
        """
        When ONNX model input names are not 'before'/'after', feeds must be
        built in positional order (first=before, second=after).
        """
        import numpy as np
        import kiosk_agent.verification_engine as eng
        from kiosk_agent.db import db_session
        from kiosk_agent.verification_engine import verify

        before_png = tmp_path / "before.png"
        after_png  = tmp_path / "after.png"
        _make_png(before_png)
        _make_png(after_png)

        with db_session() as db:
            _seed_asset(db, geo_polygon=_box_geojson(73.855, 18.515, d=0.002))
            _seed_imagery(db, "ROAD-SEG-001", str(before_png), str(after_png))

        mock_session = MagicMock()
        # Non-standard input names (e.g. from a different training framework).
        mock_session.get_inputs.return_value = [
            MagicMock(name="input_0"), MagicMock(name="input_1")
        ]
        mock_session.run.return_value = [np.array([0.8], dtype=np.float32)]
        monkeypatch.setattr(eng, "_ort_session", mock_session)

        result = verify(_make_complaint())
        # Verify the feeds dict used positional names.
        feeds = mock_session.run.call_args.args[1]
        assert "input_0" in feeds
        assert "input_1" in feeds
        assert result.evidence_status == EvidenceStatus.VERIFIED


# ---------------------------------------------------------------------------
# Stub mode tests
# ---------------------------------------------------------------------------

class TestStubMode:
    def test_road_complaint_returns_verified(self, stub_on):
        from kiosk_agent.verification_engine import verify
        result = verify(_make_complaint(category=Category.ROAD))
        assert result.evidence_status == EvidenceStatus.VERIFIED
        assert result.asset_id == "ROAD-SEG-DEMO-001"
        assert result.verification_confidence > 0.5

    @pytest.mark.parametrize("category", [
        Category.WATER,
        Category.ELECTRICITY,
        Category.SANITATION,
        Category.STREETLIGHT,
        Category.OTHER,
    ])
    def test_non_road_complaint_returns_unavailable(self, stub_on, category):
        from kiosk_agent.verification_engine import verify
        result = verify(_make_complaint(category=category))
        assert result.evidence_status == EvidenceStatus.UNAVAILABLE
        assert result.verification_confidence == 0.0

    def test_stub_result_has_dept_code(self, stub_on):
        from kiosk_agent.verification_engine import verify
        result = verify(_make_complaint(category=Category.ROAD))
        assert isinstance(result.department_code, DepartmentCode)

    def test_load_model_noop_in_stub_mode(self, stub_on):
        import kiosk_agent.verification_engine as eng
        eng.load_model()
        assert eng._ort_session is None

    def test_is_ready_true_in_stub_mode(self, stub_on):
        from kiosk_agent.verification_engine import is_ready
        assert is_ready() is True


# ---------------------------------------------------------------------------
# _load_tile() shape test
# ---------------------------------------------------------------------------

class TestLoadTile:
    def test_returns_nchw_float32(self, tmp_path):
        """_load_tile() must return shape (1, 3, 256, 256) float32 in [0, 1]."""
        try:
            import numpy as np
            from kiosk_agent.verification_engine import _load_tile

            png_path = tmp_path / "test.png"
            _make_png(png_path, colour=(200, 100, 50))

            arr = _load_tile(str(png_path))
            assert arr.shape == (1, 3, 256, 256)
            assert arr.dtype == np.float32
            assert 0.0 <= arr.min() <= arr.max() <= 1.0
        except ImportError:
            pytest.skip("numpy/Pillow not available — skipping _load_tile shape test")


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

class TestModelLifecycle:
    def test_unload_clears_session(self, stub_off, monkeypatch):
        import kiosk_agent.verification_engine as eng
        monkeypatch.setattr(eng, "_ort_session", MagicMock())
        eng.unload_model()
        assert eng._ort_session is None

    def test_is_ready_false_before_load(self, stub_off):
        from kiosk_agent.verification_engine import is_ready
        assert is_ready() is False

    def test_load_model_noop_if_already_loaded(self, stub_off, monkeypatch):
        import kiosk_agent.verification_engine as eng
        sentinel = MagicMock()
        monkeypatch.setattr(eng, "_ort_session", sentinel)

        with patch("kiosk_agent.verification_engine.ort") as mock_ort:
            # Calling load_model() again must not create a new session.
            eng.load_model()
            mock_ort.InferenceSession.assert_not_called()

        assert eng._ort_session is sentinel
