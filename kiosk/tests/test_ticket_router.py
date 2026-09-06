"""
tests/test_ticket_router.py
-----------------------------
Unit tests for kiosk_agent/ticket_router.py.

Coverage targets:
  compute_urgency_score():
    - All weight components are exercised independently
    - Duplicate penalty lowers score by exactly W_DUPLICATE * 1.0
    - Score is clamped to [0.0, 1.0] — never negative, never > 1
    - Score is rounded to 4 decimal places
    - All 6 category weights produce expected relative ordering
    - recency_weight of 0.0 vs 1.0 produces expected delta

  _is_duplicate():
    - Returns False when no prior tickets exist
    - Returns False when prior ticket is SYNCED (not a duplicate)
    - Returns True when open ticket for same asset within 48h exists
    - Returns False when prior ticket is older than 48h
    - Returns False when asset_id is None (NO_ASSET_MATCH)

  _generate_ticket_id():
    - Format: GT-{KIOSK_ID}-{YYYYMMDD}-{SEQ:05d}
    - Sequence increments on each call within same DB

  _hash_image():
    - Returns None for None input
    - Returns None for non-existent file
    - Returns correct 64-char SHA-256 hex digest for a known file
    - Is deterministic (same file → same hash)

  route():
    - Writes exactly one TicketRow to the DB
    - TicketRow has correct urgency_score, status=QUEUED, dept, evidence fields
    - Returns FinalizeResponse with matching ticket_id
    - Raises TicketRouterError on DB failure
    - Duplicate ticket correctly reduces urgency_score

  Weight sum invariant:
    - Module-level assert fires if weights don't sum to 1.0
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kiosk_agent.schemas import (
    Category,
    DepartmentCode,
    EvidenceStatus,
    FinalizeResponse,
    StructuredComplaint,
    TicketStatus,
    VerificationResult,
)
from kiosk_agent.ticket_router import (
    CATEGORY_WEIGHT,
    DUPLICATE_WINDOW_HOURS,
    W_CATEGORY,
    W_DUPLICATE,
    W_RECENCY,
    W_VERIFICATION,
    compute_urgency_score,
)


# ---------------------------------------------------------------------------
# Shared fixtures & helpers
# ---------------------------------------------------------------------------

def _complaint(category: Category = Category.ROAD) -> StructuredComplaint:
    return StructuredComplaint(
        category=category,
        subcategory="Pothole",
        description="Test complaint.",
        location_hint="near community centre",
        reported_asset_type="road_segment",
        urgency_keywords=["pothole"],
        structuring_confidence=0.90,
    )


def _verification(
    asset_id: str | None = "ROAD-SEG-001",
    confidence: float | None = 0.80,
    evidence_status: EvidenceStatus = EvidenceStatus.VERIFIED,
    dept: DepartmentCode = DepartmentCode.ROAD,
) -> VerificationResult:
    if evidence_status not in (EvidenceStatus.VERIFIED, EvidenceStatus.CONTESTED):
        confidence = None
    return VerificationResult(
        asset_id=asset_id,
        department_code=dept,
        location_lat=18.515,
        location_lon=73.855,
        verification_confidence=confidence,
        evidence_status=evidence_status,
        evidence_image_path=None,
    )


@pytest.fixture()
def seeded_db(tmp_path, monkeypatch):
    """Fresh SQLite DB with schema, redirected for all kiosk_agent modules."""
    from kiosk_agent.config import settings
    from kiosk_agent.db import init_db

    db_path = tmp_path / "test.db"
    monkeypatch.setattr(settings, "db_path", db_path)
    monkeypatch.setattr(settings, "kiosk_id", "TEST-KIOSK")
    init_db()
    return db_path


def _insert_ticket(
    db,
    ticket_id: str,
    asset_id: str,
    kiosk_id: str = "TEST-KIOSK",
    created_at: str | None = None,
    status: str = TicketStatus.QUEUED.value,
) -> None:
    from sqlalchemy import text
    if created_at is None:
        created_at = datetime.now(UTC).isoformat()
    db.execute(
        text("""
            INSERT OR REPLACE INTO tickets
              (ticket_id, kiosk_id, created_at, language, raw_transcript,
               structured_complaint, category, department_code, evidence_status,
               urgency_score, status, sync_attempts, asset_id)
            VALUES
              (:tid, :kid, :cat, 'hi', 'raw transcript',
               '{}', 'ROAD', 'ROAD', 'verified', 0.7, :status, 0, :aid)
        """),
        {
            "tid": ticket_id, "kid": kiosk_id,
            "cat": created_at, "status": status, "aid": asset_id,
        },
    )


# ---------------------------------------------------------------------------
# Weight sum invariant
# ---------------------------------------------------------------------------

class TestWeightInvariant:
    def test_weights_sum_to_one(self):
        total = W_VERIFICATION + W_RECENCY + W_CATEGORY + W_DUPLICATE
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"


# ---------------------------------------------------------------------------
# compute_urgency_score() — pure function tests
# ---------------------------------------------------------------------------

class TestComputeUrgencyScore:
    def test_verified_road_no_duplicate_full_recency(self):
        """
        Known-good baseline:
          v=0.80, r=1.0, c=0.70 (ROAD), dup=0
          = 0.40*0.80 + 0.20*1.0 + 0.25*0.70 - 0.15*0.0
          = 0.32 + 0.20 + 0.175 - 0.0 = 0.695
        """
        score = compute_urgency_score(_complaint(), _verification(), is_duplicate=False)
        assert score == pytest.approx(0.695, abs=1e-4)

    def test_duplicate_penalty_applied(self):
        """
        Duplicate should reduce score by exactly W_DUPLICATE * 1.0 = 0.15
        compared to non-duplicate, all else equal.
        """
        no_dup  = compute_urgency_score(_complaint(), _verification(), is_duplicate=False)
        with_dup = compute_urgency_score(_complaint(), _verification(), is_duplicate=True)
        assert pytest.approx(no_dup - with_dup, abs=1e-6) == W_DUPLICATE

    def test_score_never_negative(self):
        """
        Worst case: zero verification confidence, worst category (OTHER),
        zero recency, plus duplicate penalty → raw < 0 → clamped to 0.0.
        """
        score = compute_urgency_score(
            _complaint(category=Category.OTHER),
            _verification(confidence=0.0, evidence_status=EvidenceStatus.NO_ASSET_MATCH),
            is_duplicate=True,
            recency_weight=0.0,
        )
        assert score >= 0.0

    def test_score_never_above_one(self):
        """
        Best case: max verification (1.0), max recency (1.0),
        max category (WATER=1.0), no duplicate → raw can exceed 1.0 → clamped.
        """
        score = compute_urgency_score(
            _complaint(category=Category.WATER),
            _verification(confidence=1.0, dept=DepartmentCode.WATER),
            is_duplicate=False,
            recency_weight=1.0,
        )
        assert score <= 1.0

    def test_score_rounded_to_4_decimal_places(self):
        score = compute_urgency_score(_complaint(), _verification(), is_duplicate=False)
        assert score == round(score, 4)

    def test_recency_weight_zero_reduces_score(self):
        full_recency = compute_urgency_score(
            _complaint(), _verification(), is_duplicate=False, recency_weight=1.0
        )
        zero_recency = compute_urgency_score(
            _complaint(), _verification(), is_duplicate=False, recency_weight=0.0
        )
        assert full_recency - zero_recency == pytest.approx(W_RECENCY * 1.0, abs=1e-6)

    def test_category_weight_ordering(self):
        """
        Urgency must decrease in the order:
          WATER > ELECTRICITY > SANITATION > ROAD > STREETLIGHT > OTHER
        (all other parameters held constant).
        """
        verification = _verification(confidence=0.5, dept=DepartmentCode.ROAD)
        ordered_cats = [
            Category.WATER,
            Category.ELECTRICITY,
            Category.SANITATION,
            Category.ROAD,
            Category.STREETLIGHT,
            Category.OTHER,
        ]
        scores = [
            compute_urgency_score(
                _complaint(category=cat), verification,
                is_duplicate=False, recency_weight=0.5,
            )
            for cat in ordered_cats
        ]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1], (
                f"{ordered_cats[i].value} score {scores[i]:.4f} ≤ "
                f"{ordered_cats[i+1].value} score {scores[i+1]:.4f}"
            )

    def test_none_verification_confidence_treated_as_zero(self):
        """verification_confidence=None (NO_ASSET_MATCH) must not crash."""
        ver = _verification(
            asset_id=None,
            confidence=0.0,
            evidence_status=EvidenceStatus.NO_ASSET_MATCH,
        )
        # Override confidence to None to simulate a code path where it's unset.
        ver = ver.model_copy(update={"verification_confidence": None})
        score = compute_urgency_score(_complaint(), ver, is_duplicate=False)
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("cat,expected_cat_w", [
        (Category.WATER,       1.00),
        (Category.ELECTRICITY, 0.90),
        (Category.SANITATION,  0.80),
        (Category.ROAD,        0.70),
        (Category.STREETLIGHT, 0.55),
        (Category.OTHER,       0.40),
    ])
    def test_category_weight_values(self, cat, expected_cat_w):
        assert CATEGORY_WEIGHT[cat] == pytest.approx(expected_cat_w)


# ---------------------------------------------------------------------------
# _is_duplicate() tests
# ---------------------------------------------------------------------------

class TestIsDuplicate:
    def test_no_prior_tickets(self, seeded_db):
        from kiosk_agent.ticket_router import _is_duplicate
        assert _is_duplicate("ROAD-SEG-001", "TEST-KIOSK") is False

    def test_returns_false_for_none_asset_id(self, seeded_db):
        """NO_ASSET_MATCH complaints (asset_id=None) are never duplicates."""
        from kiosk_agent.ticket_router import _is_duplicate
        assert _is_duplicate(None, "TEST-KIOSK") is False

    def test_returns_true_for_open_ticket_within_window(self, seeded_db):
        """Open ticket for same asset within 48h → duplicate."""
        from kiosk_agent.db import db_session
        from kiosk_agent.ticket_router import _is_duplicate

        with db_session() as db:
            _insert_ticket(db, "GT-TEST-001", "ROAD-SEG-001",
                           status=TicketStatus.QUEUED.value)

        assert _is_duplicate("ROAD-SEG-001", "TEST-KIOSK") is True

    def test_returns_false_for_synced_ticket(self, seeded_db):
        """A SYNCED ticket must NOT trigger the duplicate penalty."""
        from kiosk_agent.db import db_session
        from kiosk_agent.ticket_router import _is_duplicate

        with db_session() as db:
            _insert_ticket(db, "GT-TEST-002", "ROAD-SEG-001",
                           status=TicketStatus.SYNCED.value)

        assert _is_duplicate("ROAD-SEG-001", "TEST-KIOSK") is False

    def test_returns_false_for_ticket_outside_window(self, seeded_db):
        """Ticket older than DUPLICATE_WINDOW_HOURS must not block new report."""
        from kiosk_agent.db import db_session
        from kiosk_agent.ticket_router import _is_duplicate

        old_time = (
            datetime.now(UTC) - timedelta(hours=DUPLICATE_WINDOW_HOURS + 1)
        ).isoformat()

        with db_session() as db:
            _insert_ticket(db, "GT-TEST-003", "ROAD-SEG-001",
                           created_at=old_time, status=TicketStatus.QUEUED.value)

        assert _is_duplicate("ROAD-SEG-001", "TEST-KIOSK") is False

    def test_different_kiosk_not_a_duplicate(self, seeded_db):
        """Open ticket from a different kiosk for the same asset ≠ duplicate."""
        from kiosk_agent.db import db_session
        from kiosk_agent.ticket_router import _is_duplicate

        with db_session() as db:
            _insert_ticket(db, "GT-OTHER-001", "ROAD-SEG-001",
                           kiosk_id="OTHER-KIOSK",
                           status=TicketStatus.QUEUED.value)

        assert _is_duplicate("ROAD-SEG-001", "TEST-KIOSK") is False

    def test_different_asset_not_a_duplicate(self, seeded_db):
        """Open ticket for a different asset is not a duplicate."""
        from kiosk_agent.db import db_session
        from kiosk_agent.ticket_router import _is_duplicate

        with db_session() as db:
            _insert_ticket(db, "GT-TEST-004", "WATER-PUMP-001",
                           status=TicketStatus.QUEUED.value)

        assert _is_duplicate("ROAD-SEG-001", "TEST-KIOSK") is False


# ---------------------------------------------------------------------------
# _hash_image() tests
# ---------------------------------------------------------------------------

class TestHashImage:
    def test_none_input_returns_none(self):
        from kiosk_agent.ticket_router import _hash_image
        assert _hash_image(None) is None

    def test_nonexistent_file_returns_none(self, tmp_path):
        from kiosk_agent.ticket_router import _hash_image
        assert _hash_image(str(tmp_path / "missing.png")) is None

    def test_returns_64_char_hex_digest(self, tmp_path):
        from kiosk_agent.ticket_router import _hash_image
        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG" + b"\x00" * 100)
        result = _hash_image(str(f))
        assert result is not None
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_hash_is_deterministic(self, tmp_path):
        from kiosk_agent.ticket_router import _hash_image
        f = tmp_path / "img.png"
        f.write_bytes(b"consistent content")
        h1 = _hash_image(str(f))
        h2 = _hash_image(str(f))
        assert h1 == h2

    def test_hash_matches_sha256(self, tmp_path):
        from kiosk_agent.ticket_router import _hash_image
        content = b"GroundTruth evidence photo"
        f = tmp_path / "evidence.png"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert _hash_image(str(f)) == expected

    def test_different_files_different_hashes(self, tmp_path):
        from kiosk_agent.ticket_router import _hash_image
        f1 = tmp_path / "a.png"
        f2 = tmp_path / "b.png"
        f1.write_bytes(b"file A content")
        f2.write_bytes(b"file B content")
        assert _hash_image(str(f1)) != _hash_image(str(f2))


# ---------------------------------------------------------------------------
# _generate_ticket_id() tests
# ---------------------------------------------------------------------------

class TestGenerateTicketId:
    def test_format_matches_spec(self, seeded_db):
        """Format must be GT-{KIOSK_ID}-{YYYYMMDD}-{SEQ:05d}."""
        from kiosk_agent.db import db_session
        from kiosk_agent.ticket_router import _generate_ticket_id

        with db_session() as db:
            ticket_id = _generate_ticket_id(db, "TEST-KIOSK")

        today = datetime.now(UTC).strftime("%Y%m%d")
        assert ticket_id.startswith(f"GT-TEST-KIOSK-{today}-")
        seq_part = ticket_id.split("-")[-1]
        assert len(seq_part) == 5
        assert seq_part.isdigit()

    def test_sequence_increments(self, seeded_db):
        """Each call within the same DB session must produce a higher SEQ."""
        from kiosk_agent.db import db_session
        from kiosk_agent.ticket_router import _generate_ticket_id

        with db_session() as db:
            id1 = _generate_ticket_id(db, "TEST-KIOSK")
            id2 = _generate_ticket_id(db, "TEST-KIOSK")

        seq1 = int(id1.split("-")[-1])
        seq2 = int(id2.split("-")[-1])
        assert seq2 > seq1


# ---------------------------------------------------------------------------
# route() integration tests
# ---------------------------------------------------------------------------

class TestRoute:
    def test_route_writes_ticket_to_db(self, seeded_db, monkeypatch):
        """route() must persist exactly one TicketRow with status=QUEUED."""
        from kiosk_agent.db import db_session
        from kiosk_agent.ticket_router import route
        from kiosk_agent.db import TicketRow
        from sqlalchemy import select

        result = route(
            complaint=_complaint(),
            verification=_verification(),
            raw_transcript="Test transcript.",
            language="hi",
        )

        with db_session() as db:
            rows = db.execute(select(TicketRow)).scalars().all()

        assert len(rows) == 1
        row = rows[0]
        assert row.ticket_id == result.ticket_id
        assert row.status == TicketStatus.QUEUED.value
        assert row.kiosk_id == "TEST-KIOSK"
        assert row.category == "ROAD"
        assert row.urgency_score == pytest.approx(result.urgency_score)

    def test_route_returns_finalize_response(self, seeded_db):
        from kiosk_agent.ticket_router import route

        result = route(
            complaint=_complaint(),
            verification=_verification(),
            raw_transcript="transcript",
            language="hi",
        )
        assert isinstance(result, FinalizeResponse)
        assert result.ticket_id.startswith("GT-TEST-KIOSK-")
        assert result.category == Category.ROAD
        assert result.department_code == DepartmentCode.ROAD
        assert 0.0 <= result.urgency_score <= 1.0
        assert result.evidence_status == EvidenceStatus.VERIFIED

    def test_route_duplicate_reduces_urgency(self, seeded_db):
        """Second ticket for same asset should have lower urgency_score."""
        from kiosk_agent.db import db_session
        from kiosk_agent.ticket_router import route

        # First ticket.
        r1 = route(_complaint(), _verification(), "t1", "hi")

        # Second ticket — same asset, same kiosk → is_duplicate=True.
        r2 = route(_complaint(), _verification(), "t2", "hi")

        assert r2.urgency_score < r1.urgency_score
        assert pytest.approx(r1.urgency_score - r2.urgency_score, abs=1e-4) == W_DUPLICATE

    def test_route_uses_verification_dept_code(self, seeded_db):
        """department_code on the ticket must come from VerificationResult, not category mapping."""
        from kiosk_agent.ticket_router import route
        from kiosk_agent.db import db_session
        from kiosk_agent.db import TicketRow

        ver = _verification(dept=DepartmentCode.WATER)  # overridden dept
        res = route(_complaint(category=Category.ROAD), ver, "t", "hi")

        with db_session() as db:
            row = db.get(TicketRow, res.ticket_id)
        # VerificationResult dept (WATER) takes precedence over ROAD's default.
        assert row.department_code == DepartmentCode.WATER.value

    def test_route_raises_on_db_failure(self, seeded_db, monkeypatch):
        """DB write failure must raise TicketRouterError (not a raw SQLAlchemy error)."""
        from kiosk_agent import ticket_router as tr_mod
        from kiosk_agent.ticket_router import route, TicketRouterError

        # Patch db_session to raise on __enter__.
        class _BrokenCtx:
            def __enter__(self): raise OSError("disk full")
            def __exit__(self, *a): pass

        monkeypatch.setattr(tr_mod, "db_session", lambda: _BrokenCtx())

        with pytest.raises(TicketRouterError, match="DB write failed"):
            route(_complaint(), _verification(), "t", "hi")

    def test_route_stores_raw_transcript(self, seeded_db):
        from kiosk_agent.ticket_router import route
        from kiosk_agent.db import db_session
        from kiosk_agent.db import TicketRow

        transcript = "सड़क पर गड्ढा है।"
        res = route(_complaint(), _verification(), transcript, "hi")

        with db_session() as db:
            row = db.get(TicketRow, res.ticket_id)
        assert row.raw_transcript == transcript

    def test_route_no_asset_match_sets_correct_status(self, seeded_db):
        """NO_ASSET_MATCH evidence_status must be stored correctly on the row."""
        from kiosk_agent.ticket_router import route
        from kiosk_agent.db import db_session
        from kiosk_agent.db import TicketRow

        ver = _verification(
            asset_id=None,
            confidence=None,
            evidence_status=EvidenceStatus.NO_ASSET_MATCH,
        )
        res = route(_complaint(), ver, "t", "hi")

        with db_session() as db:
            row = db.get(TicketRow, res.ticket_id)
        assert row.evidence_status == EvidenceStatus.NO_ASSET_MATCH.value
        assert row.asset_id is None
