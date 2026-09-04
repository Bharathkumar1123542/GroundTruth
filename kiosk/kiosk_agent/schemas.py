"""
kiosk_agent/schemas.py
-----------------------
All Pydantic v2 models for:
  - Engine inputs/outputs (ASR, Structuring, Verification, Ticket)
  - Kiosk Internal API request/response shapes (architecture.md §12.1)
  - Sync API payload shapes (architecture.md §12.2)

These are the SINGLE source of truth for data shapes in the kiosk codebase.
The db.py SQLAlchemy models are derived from these schemas — not the other way around.
Do NOT hand-roll parallel schema definitions elsewhere.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


# ===========================================================================
# Enumerations — sourced from architecture.md §4.2, §6.2
# ===========================================================================

class Category(StrEnum):
    """Complaint categories — exactly the six values enforced by the GBNF grammar."""
    ROAD        = "ROAD"
    WATER       = "WATER"
    ELECTRICITY = "ELECTRICITY"
    SANITATION  = "SANITATION"
    STREETLIGHT = "STREETLIGHT"
    OTHER       = "OTHER"


class DepartmentCode(StrEnum):
    """Default department codes (architecture.md §4.6)."""
    ROAD  = "ROAD"
    WATER = "WATER"
    ELEC  = "ELEC"
    SANI  = "SANI"
    LIGHT = "LIGHT"
    GEN   = "GEN"


class EvidenceStatus(StrEnum):
    """Verification result status values (architecture.md §4.4)."""
    VERIFIED       = "verified"
    UNAVAILABLE    = "unavailable"
    NO_ASSET_MATCH = "no_asset_match"
    CONTESTED      = "contested"


class TicketStatus(StrEnum):
    """Ticket lifecycle states (architecture.md §6.2)."""
    CREATED      = "CREATED"
    QUEUED       = "QUEUED"
    SYNCED       = "SYNCED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED     = "RESOLVED"
    REJECTED     = "REJECTED"


# Department code lookup by category (architecture.md §4.6)
CATEGORY_TO_DEPARTMENT: dict[Category, DepartmentCode] = {
    Category.ROAD:        DepartmentCode.ROAD,
    Category.WATER:       DepartmentCode.WATER,
    Category.ELECTRICITY: DepartmentCode.ELEC,
    Category.SANITATION:  DepartmentCode.SANI,
    Category.STREETLIGHT: DepartmentCode.LIGHT,
    Category.OTHER:       DepartmentCode.GEN,
}


# ===========================================================================
# Engine output models
# ===========================================================================

class AsrResult(BaseModel):
    """Output of ASR Engine (architecture.md §4.1)."""
    transcript:          str
    language_detected:   str   # ISO-639-1 code, e.g. "hi", "mr", "ta"
    asr_confidence:      Annotated[float, Field(ge=0.0, le=1.0)]
    segment_confidences: list[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(default_factory=list)


class StructuredComplaint(BaseModel):
    """
    Schema-conformant structured complaint (architecture.md §6.1).
    This exact shape is enforced by the GBNF grammar at generation time;
    validation here is a belt-and-suspenders defence for stub/test paths.
    """
    category:               Category
    subcategory:            Annotated[str, Field(max_length=64)]
    description:            Annotated[str, Field(max_length=280)]
    location_hint:          str
    reported_asset_type:    str
    urgency_keywords:       Annotated[list[str], Field(min_length=0, max_length=5)] = Field(
                                default_factory=list
                            )
    structuring_confidence: Annotated[float, Field(ge=0.0, le=1.0)]

    @field_validator("urgency_keywords")
    @classmethod
    def _max_five_keywords(cls, v: list[str]) -> list[str]:
        if len(v) > 5:
            raise ValueError("urgency_keywords must contain at most 5 elements")
        return v


class VerificationResult(BaseModel):
    """Output of the Verification Engine (architecture.md §4.4)."""
    asset_id:                str | None
    department_code:         DepartmentCode
    location_lat:            float | None = None
    location_lon:            float | None = None
    verification_confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    evidence_status:         EvidenceStatus
    evidence_image_path:     str | None = None

    @model_validator(mode="after")
    def _confidence_only_when_evidence_available(self) -> "VerificationResult":
        if self.evidence_status != EvidenceStatus.VERIFIED and self.verification_confidence is not None:
            # contested is the one case where confidence can be present but below threshold
            if self.evidence_status != EvidenceStatus.CONTESTED:
                raise ValueError(
                    "verification_confidence must be None when evidence_status is not "
                    "'verified' or 'contested'"
                )
        return self


class Ticket(BaseModel):
    """
    Full ticket record (architecture.md §6.2).
    Maps 1-to-1 with the `tickets` SQLite table.
    """
    ticket_id:               str   # GT-{KIOSK_ID}-{YYYYMMDD}-{SEQ:05d}
    kiosk_id:                str
    created_at:              datetime
    language:                str   # ISO-639-1
    raw_transcript:          str
    structured_complaint:    StructuredComplaint
    category:                Category
    department_code:         DepartmentCode
    asset_id:                str | None
    location_lat:            float | None
    location_lon:            float | None
    evidence_image_path:     str | None
    evidence_image_hash:     str | None   # SHA-256
    verification_confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None
    evidence_status:         EvidenceStatus
    urgency_score:           Annotated[float, Field(ge=0.0, le=1.0)]
    status:                  TicketStatus = TicketStatus.QUEUED
    sync_attempts:           int = 0
    last_sync_attempt:       datetime | None = None


# ===========================================================================
# Kiosk Internal API — request / response models (architecture.md §12.1)
# ===========================================================================

class RecordRequest(BaseModel):
    """POST /v1/complaints/record — body (language selection from UI)."""
    language: str = Field(
        description="ISO-639-1 language code selected by the resident.",
        examples=["hi", "mr", "ta"],
    )

    @field_validator("language")
    @classmethod
    def _supported_language(cls, v: str) -> str:
        supported = {"hi", "mr", "ta"}
        if v not in supported:
            raise ValueError(f"Language '{v}' is not supported. Choose from: {supported}")
        return v


class RecordResponse(BaseModel):
    """POST /v1/complaints/record — response."""
    session_id: UUID


class FinalizeResponse(BaseModel):
    """POST /v1/complaints/{session_id}/finalize — response (architecture.md §12.1)."""
    ticket_id:       str
    category:        Category
    department_code: DepartmentCode
    urgency_score:   float
    evidence_status: EvidenceStatus


class TicketSummary(BaseModel):
    """One entry in GET /v1/tickets list response."""
    ticket_id:       str
    created_at:      datetime
    category:        Category
    department_code: DepartmentCode
    urgency_score:   float
    evidence_status: EvidenceStatus
    status:          TicketStatus


class HealthResponse(BaseModel):
    """GET /v1/health — (architecture.md §12.1)."""
    asr:               str   # "ok" | "unavailable"
    structuring:       str
    verification:      str
    storage_pct_used:  float
    unsynced_tickets:  int
    last_sync_at:      datetime | None = None


class ErrorResponse(BaseModel):
    """Structured error returned to the Local UI on pipeline failure (implementation.md §4.1)."""
    error_code:    str
    message:       str
    retry_allowed: bool = True


class SyncTriggerResponse(BaseModel):
    """POST /v1/sync/trigger — response."""
    triggered: bool


# ===========================================================================
# Sync API payload shapes (architecture.md §12.2)
# Used by sync_daemon/sync_client.py — defined here to share with backend schemas.
# ===========================================================================

class TicketSyncPayload(BaseModel):
    """
    Wire representation of a single ticket in a sync batch push.
    Structured complaint is serialised as a dict for transport; backend
    validates it against its own StructuredComplaint schema on ingestion.
    """
    ticket_id:               str
    kiosk_id:                str
    created_at:              datetime
    language:                str
    raw_transcript:          str
    structured_complaint:    dict[str, Any]   # JSON-serialised StructuredComplaint
    category:                str
    department_code:         str
    asset_id:                str | None
    location_lat:            float | None
    location_lon:            float | None
    evidence_image_hash:     str | None
    verification_confidence: float | None
    evidence_status:         str
    urgency_score:           float


class BatchPushRequest(BaseModel):
    """POST /sync/v1/tickets/batch — request body (architecture.md §12.2)."""
    kiosk_id: str
    tickets:  list[TicketSyncPayload]
    # Evidence images are sent as multipart form fields alongside this JSON body.


class BatchPushResponse(BaseModel):
    """POST /sync/v1/tickets/batch — response."""
    acknowledged:    list[str]          # ticket_ids accepted
    queue_positions: dict[str, int]     # ticket_id → position in dept queue


class RegistryDeltaResponse(BaseModel):
    """GET /sync/v1/registry/delta — response (architecture.md §12.2)."""
    current_version:   int
    updated_assets:    list[dict[str, Any]]
    deleted_asset_ids: list[str]


class HeartbeatRequest(BaseModel):
    """POST /sync/v1/heartbeat — request body (architecture.md §12.2)."""
    kiosk_id:            str
    storage_pct_used:    float
    unsynced_ticket_count: int
    last_boot_at:        datetime


class HeartbeatResponse(BaseModel):
    """POST /sync/v1/heartbeat — response."""
    acknowledged: bool
