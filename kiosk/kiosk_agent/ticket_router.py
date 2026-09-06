"""
kiosk_agent/ticket_router.py
------------------------------
Ticket Router — computes urgency score, writes a Ticket to the local queue,
and returns a fully populated Ticket ready for sync.

Specification: architecture.md §4.4, implementation.md §4.5

Urgency Score Formula (architecture.md §4.4):
  urgency_score = w_v * verification_confidence
                + w_r * recency_weight
                + w_c * category_weight[category]
                - w_d * duplicate_penalty

  where:
    w_v = 0.40  (verification confidence weight)
    w_r = 0.20  (recency weight — higher for fresher complaints)
    w_c = 0.25  (category weight — see CATEGORY_WEIGHT table)
    w_d = 0.15  (duplicate penalty — subtracted if a recent open ticket
                  for the same asset exists within DUPLICATE_WINDOW_HOURS)

  Result is clamped to [0.0, 1.0] and rounded to 4 decimal places.

Category weights (implementation.md §4.5):
  WATER       : 1.0  (highest — health/life impact)
  ELECTRICITY : 0.90
  SANITATION  : 0.80
  ROAD        : 0.70
  STREETLIGHT : 0.55
  OTHER       : 0.40  (lowest)

Recency weight:
  The kiosk records complaints round-the-clock. "Recency" here measures
  how recently the kiosk itself received the complaint relative to the
  current UTC hour within a 24-hour sliding window.
  For the MVP, recency_weight = 1.0 (all complaints are fresh — the queue
  is consumed by sync within hours). Future versions can discount stale
  queue entries.

Duplicate detection:
  Queries the local ticket queue for tickets with the same asset_id
  created within DUPLICATE_WINDOW_HOURS (default 48h) that are not yet
  SYNCED. If found, duplicate_penalty = 1.0; else 0.0.

Ticket ID format (implementation.md §4.5):
  GT-{KIOSK_ID}-{YYYYMMDD}-{SEQ:05d}
  e.g. GT-DEV-KIOSK-001-20260904-00001
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Optional

from kiosk_agent.config import settings
from kiosk_agent.db import (
    TicketRow,
    db_session,
    next_ticket_seq,
    queued_ticket_count,
)
from kiosk_agent.schemas import (
    CATEGORY_TO_DEPARTMENT,
    Category,
    DepartmentCode,
    EvidenceStatus,
    FinalizeResponse,
    StructuredComplaint,
    TicketStatus,
    VerificationResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Urgency scoring constants
# ---------------------------------------------------------------------------

# Weights must sum to 1.0.
W_VERIFICATION = 0.40
W_RECENCY      = 0.20
W_CATEGORY     = 0.25
W_DUPLICATE    = 0.15

assert abs(W_VERIFICATION + W_RECENCY + W_CATEGORY + W_DUPLICATE - 1.0) < 1e-9, (
    "Urgency weight components must sum to 1.0"
)

# Category urgency weights (architecture.md §4.4).
CATEGORY_WEIGHT: dict[Category, float] = {
    Category.WATER:       1.00,
    Category.ELECTRICITY: 0.90,
    Category.SANITATION:  0.80,
    Category.ROAD:        0.70,
    Category.STREETLIGHT: 0.55,
    Category.OTHER:       0.40,
}

# Hours within which a same-asset open ticket is considered a duplicate.
DUPLICATE_WINDOW_HOURS = 48


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class TicketRouterError(Exception):
    """Raised on an unrecoverable error during ticket creation."""


# ---------------------------------------------------------------------------
# Urgency scoring
# ---------------------------------------------------------------------------

def compute_urgency_score(
    complaint: StructuredComplaint,
    verification: VerificationResult,
    is_duplicate: bool,
    recency_weight: float = 1.0,
) -> float:
    """
    Compute the urgency score for a complaint.

    Parameters
    ----------
    complaint : StructuredComplaint
        Structured complaint output from the Structuring Engine.
    verification : VerificationResult
        Output from the Verification Engine.
    is_duplicate : bool
        True if an open ticket for the same asset exists within
        DUPLICATE_WINDOW_HOURS.
    recency_weight : float
        Weight in [0, 1] expressing how recently the complaint was filed.
        Defaults to 1.0 (all MVP complaints are fresh).

    Returns
    -------
    float
        Urgency score ∈ [0.0, 1.0], rounded to 4 decimal places.
    """
    cat_w  = CATEGORY_WEIGHT.get(complaint.category, 0.40)
    dup_p  = 1.0 if is_duplicate else 0.0
    v_conf = verification.verification_confidence or 0.0

    raw = (
        W_VERIFICATION * v_conf
        + W_RECENCY     * recency_weight
        + W_CATEGORY    * cat_w
        - W_DUPLICATE   * dup_p
    )

    score = max(0.0, min(1.0, raw))
    return round(score, 4)


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def _is_duplicate(asset_id: Optional[str], kiosk_id: str) -> bool:
    """
    Return True if a ticket for the same asset_id (not yet SYNCED) exists
    in the local queue created within DUPLICATE_WINDOW_HOURS.

    An asset_id of None (NO_ASSET_MATCH) is never a duplicate.
    """
    if asset_id is None:
        return False

    cutoff = (
        datetime.now(UTC) - timedelta(hours=DUPLICATE_WINDOW_HOURS)
    ).isoformat()

    from sqlalchemy import select, and_  # noqa: PLC0415

    with db_session() as db:
        existing = db.execute(
            select(TicketRow).where(
                and_(
                    TicketRow.asset_id == asset_id,
                    TicketRow.kiosk_id == kiosk_id,
                    TicketRow.created_at >= cutoff,
                    TicketRow.status != TicketStatus.SYNCED.value,
                )
            )
        ).scalars().first()

    is_dup = existing is not None
    if is_dup:
        logger.info(
            "Duplicate detected: asset=%s  existing_ticket=%s",
            asset_id, existing.ticket_id,
        )
    return is_dup


# ---------------------------------------------------------------------------
# Ticket ID generation
# ---------------------------------------------------------------------------

def _generate_ticket_id(db, kiosk_id: str) -> str:
    """
    Generate the next ticket ID for this kiosk.
    Format: GT-{KIOSK_ID}-{YYYYMMDD}-{SEQ:05d}
    Sequence is per-kiosk-per-day (resets daily is acceptable for MVP;
    the full tuple (kiosk_id, date, seq) is globally unique).
    """
    today = datetime.now(UTC).strftime("%Y%m%d")
    seq = next_ticket_seq(db, kiosk_id)
    return f"GT-{kiosk_id}-{today}-{seq:05d}"


# ---------------------------------------------------------------------------
# Core routing function
# ---------------------------------------------------------------------------

def route(
    complaint: StructuredComplaint,
    verification: VerificationResult,
    raw_transcript: str,
    language: str,
    audio_path: Optional[str] = None,
) -> FinalizeResponse:
    """
    Compute urgency, write the ticket to the local SQLite queue, and
    return a FinalizeResponse for the Local UI confirmation screen.

    Parameters
    ----------
    complaint : StructuredComplaint
        Output of the Structuring Engine.
    verification : VerificationResult
        Output of the Verification Engine.
    raw_transcript : str
        Original ASR transcript (stored verbatim for auditability).
    language : str
        ISO-639-1 language code of the complaint.
    audio_path : str | None
        Path to the saved audio recording (stored for future re-processing).

    Returns
    -------
    FinalizeResponse
        Contains ticket_id, category, department_code, urgency_score,
        evidence_status — used by the Local UI confirmation screen and
        the Sync Daemon.

    Raises
    ------
    TicketRouterError
        On DB write failure or any unrecoverable error.
    """
    kiosk_id = settings.kiosk_id

    try:
        # ── Duplicate check ───────────────────────────────────────────────────
        is_dup = _is_duplicate(verification.asset_id, kiosk_id)

        # ── Urgency score ─────────────────────────────────────────────────────
        urgency_score = compute_urgency_score(
            complaint=complaint,
            verification=verification,
            is_duplicate=is_dup,
        )
        logger.info(
            "Urgency score: %.4f  (category=%s  verified=%s  duplicate=%s)",
            urgency_score,
            complaint.category.value,
            verification.evidence_status == EvidenceStatus.VERIFIED,
            is_dup,
        )

        # ── Determine department_code ─────────────────────────────────────────
        dept_code: DepartmentCode = (
            verification.department_code
            or CATEGORY_TO_DEPARTMENT.get(complaint.category, DepartmentCode.GEN)
        )

        # ── Persist ticket ────────────────────────────────────────────────────
        created_at = datetime.now(UTC).isoformat()

        with db_session() as db:
            ticket_id = _generate_ticket_id(db, kiosk_id)

            row = TicketRow(
                ticket_id=ticket_id,
                kiosk_id=kiosk_id,
                created_at=created_at,
                language=language,
                raw_transcript=raw_transcript,
                structured_complaint=complaint.model_dump_json(),
                category=complaint.category.value,
                department_code=dept_code.value,
                asset_id=verification.asset_id,
                location_lat=verification.location_lat,
                location_lon=verification.location_lon,
                evidence_image_path=verification.evidence_image_path,
                evidence_image_hash=_hash_image(verification.evidence_image_path),
                verification_confidence=verification.verification_confidence,
                evidence_status=verification.evidence_status.value,
                urgency_score=urgency_score,
                status=TicketStatus.QUEUED.value,
                audio_path=audio_path,
                sync_attempts=0,
                last_sync_attempt=None,
            )
            db.add(row)

    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to write ticket to DB: %s", exc, exc_info=True)
        raise TicketRouterError(f"DB write failed: {exc}") from exc

    logger.info(
        "Ticket created: %s  urgency=%.4f  dept=%s  evidence=%s",
        ticket_id, urgency_score, dept_code.value,
        verification.evidence_status.value,
    )

    return FinalizeResponse(
        ticket_id=ticket_id,
        category=complaint.category,
        department_code=dept_code,
        urgency_score=urgency_score,
        evidence_status=verification.evidence_status,
    )


# ---------------------------------------------------------------------------
# Image hashing (for deduplication on the backend)
# ---------------------------------------------------------------------------

def _hash_image(image_path: Optional[str]) -> Optional[str]:
    """
    Compute the SHA-256 hex digest of the evidence image file.
    Used by the Municipal Backend to deduplicate evidence photos.
    Returns None if image_path is None or the file does not exist.
    """
    if not image_path:
        return None

    import hashlib  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    p = Path(image_path)
    if not p.exists():
        return None

    h = hashlib.sha256()
    try:
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as exc:
        logger.warning("Could not hash evidence image %s: %s", image_path, exc)
        return None
