"""
kiosk_agent/orchestrator.py
--------------------------
Pipeline Orchestrator — chains ASR, Structuring, Verification, and Ticket Routing
into a unified end-to-end processing pipeline.

Specification: architecture.md §4, implementation.md §4.6

Pipeline Flow:
  1. ASR Engine:
     audio_bytes (WebM/OGG/PCM) → asr_engine.transcribe() → AsrResult(transcript, confidence)
  2. Structuring Engine:
     transcript + language → structuring_engine.structure() → StructuredComplaint
  3. Verification Engine:
     StructuredComplaint → verification_engine.verify() → VerificationResult
     (Non-fatal fallback: if verification fails unexpectedly, marks UNAVAILABLE
      so the citizen's complaint ticket is never discarded).
  4. Ticket Router:
     StructuredComplaint + VerificationResult + transcript → ticket_router.route()
     → FinalizeResponse (persisted to SQLite queue).

Error Handling:
  All domain exceptions from downstream engines are mapped to PipelineExecutionError
  carrying structured error_code, user-friendly message, retry_allowed flag, and HTTP status.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from kiosk_agent import asr_engine, structuring_engine, ticket_router, verification_engine
from kiosk_agent.config import settings
from kiosk_agent.schemas import (
    DepartmentCode,
    ErrorResponse,
    EvidenceStatus,
    FinalizeResponse,
    StructuredComplaint,
    VerificationResult,
    CATEGORY_TO_DEPARTMENT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline Exceptions
# ---------------------------------------------------------------------------

class PipelineExecutionError(Exception):
    """
    Raised when a step in the complaint processing pipeline fails.
    Carries UI-facing error metadata and HTTP status code.
    """
    def __init__(
        self,
        error_code: str,
        message: str,
        retry_allowed: bool = True,
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.retry_allowed = retry_allowed
        self.status_code = status_code

    def to_error_response(self) -> ErrorResponse:
        """Convert to Pydantic ErrorResponse schema."""
        return ErrorResponse(
            error_code=self.error_code,
            message=self.message,
            retry_allowed=self.retry_allowed,
        )


# ---------------------------------------------------------------------------
# Model Lifespan Initializer
# ---------------------------------------------------------------------------

def load_all_models() -> None:
    """
    Initialize all offline ML engines once at process startup.
    In stub mode (STUB_STRUCTURING=true), logs a notice and skips heavy loading.
    """
    if settings.stub_structuring:
        logger.info("STUB_STRUCTURING=true — skipping physical ML model loading.")
        return

    logger.info("Loading offline ML models into memory...")
    t0 = time.perf_counter()

    try:
        asr_engine.load_model()
        structuring_engine.load_model()
        verification_engine.load_model()
        elapsed_s = time.perf_counter() - t0
        logger.info("All ML models loaded successfully in %.2fs", elapsed_s)
    except Exception as exc:
        logger.error("Failed to initialize ML models: %s", exc, exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Pipeline Orchestrator Function
# ---------------------------------------------------------------------------

def run_pipeline(
    session_id: str,
    audio_bytes: bytes,
    language: str,
    audio_path: Optional[str] = None,
) -> FinalizeResponse:
    """
    Executes the full offline civic complaint pipeline:
      ASR -> Structuring -> Verification -> Ticket Routing & Persistence.

    Parameters
    ----------
    session_id : str
        Active recording session ID.
    audio_bytes : bytes
        Recorded audio bytes from the kiosk UI.
    language : str
        ISO-639-1 language code selected by citizen (hi, mr, ta).
    audio_path : str | None
        Optional filesystem path where the raw audio was saved.

    Returns
    -------
    FinalizeResponse
        Finalized ticket metadata and routing response.

    Raises
    ------
    PipelineExecutionError
        On any unrecoverable pipeline failure with mapped HTTP/UI details.
    """
    total_start = time.perf_counter()
    logger.info(
        "Pipeline START: session=%s  language=%s  audio_size=%d bytes",
        session_id, language, len(audio_bytes),
    )

    # ------------------------------------------------------------------
    # Step 1: ASR Engine (Whisper)
    # ------------------------------------------------------------------
    t_stage = time.perf_counter()
    try:
        asr_res = asr_engine.transcribe(
            audio_bytes=audio_bytes,
            language_hint=language,
        )
        logger.info(
            "ASR finished in %.2fs — confidence=%.3f, transcript_len=%d",
            time.perf_counter() - t_stage,
            asr_res.asr_confidence,
            len(asr_res.transcript),
        )
    except asr_engine.AsrInputTooLongError as exc:
        logger.warning("Pipeline ASR error (input too long): %s", exc)
        raise PipelineExecutionError(
            error_code="ASR_INPUT_TOO_LONG",
            message=f"Audio recording was too long ({exc.duration_s:.1f}s). Maximum allowed is {exc.max_s}s. Please record again.",
            retry_allowed=True,
            status_code=400,
        ) from exc
    except asr_engine.AsrLowConfidenceError as exc:
        logger.warning("Pipeline ASR error (low confidence): %s", exc)
        raise PipelineExecutionError(
            error_code="ASR_LOW_CONFIDENCE",
            message="Could not clearly understand the audio. Please speak clearly and closer to the microphone.",
            retry_allowed=True,
            status_code=422,
        ) from exc
    except asr_engine.AsrModelNotReadyError as exc:
        logger.error("Pipeline ASR error (model not ready): %s", exc)
        raise PipelineExecutionError(
            error_code="ASR_NOT_READY",
            message="ASR speech recognition engine is not ready.",
            retry_allowed=False,
            status_code=503,
        ) from exc
    except Exception as exc:
        logger.error("Unexpected ASR failure: %s", exc, exc_info=True)
        raise PipelineExecutionError(
            error_code="ASR_FAILED",
            message="An unexpected error occurred during speech transcription. Please try recording again.",
            retry_allowed=True,
            status_code=500,
        ) from exc

    transcript = asr_res.transcript.strip()
    if not transcript:
        logger.warning("ASR returned empty transcript for session %s", session_id)
        raise PipelineExecutionError(
            error_code="ASR_EMPTY_TRANSCRIPT",
            message="No speech detected in the audio. Please try recording your complaint again.",
            retry_allowed=True,
            status_code=422,
        )

    # ------------------------------------------------------------------
    # Step 2: Structuring Engine (Gemma-2-2b + GBNF)
    # ------------------------------------------------------------------
    t_stage = time.perf_counter()
    try:
        structured: StructuredComplaint = structuring_engine.structure(
            transcript=transcript,
            language=language,
        )
        logger.info(
            "Structuring finished in %.2fs — category=%s, confidence=%.3f",
            time.perf_counter() - t_stage,
            structured.category.value,
            structured.structuring_confidence,
        )
    except structuring_engine.StructuringTimeoutError as exc:
        logger.warning("Pipeline Structuring error (timeout): %s", exc)
        raise PipelineExecutionError(
            error_code="STRUCTURING_TIMEOUT",
            message="Complaint analysis timed out. Please try submitting again.",
            retry_allowed=True,
            status_code=504,
        ) from exc
    except structuring_engine.StructuringModelNotReadyError as exc:
        logger.error("Pipeline Structuring error (model not ready): %s", exc)
        raise PipelineExecutionError(
            error_code="STRUCTURING_NOT_READY",
            message="Language structuring engine is not ready.",
            retry_allowed=False,
            status_code=503,
        ) from exc
    except structuring_engine.StructuringOutputInvalidError as exc:
        logger.error("Pipeline Structuring error (invalid output): %s", exc)
        raise PipelineExecutionError(
            error_code="STRUCTURING_INVALID",
            message="Unable to structure the complaint. Please try recording again.",
            retry_allowed=True,
            status_code=422,
        ) from exc
    except Exception as exc:
        logger.error("Unexpected Structuring failure: %s", exc, exc_info=True)
        raise PipelineExecutionError(
            error_code="STRUCTURING_FAILED",
            message="An unexpected error occurred while analyzing the complaint.",
            retry_allowed=True,
            status_code=500,
        ) from exc

    # ------------------------------------------------------------------
    # Step 3: Verification Engine (Geo / ONNX Change Detection)
    # ------------------------------------------------------------------
    t_stage = time.perf_counter()
    try:
        verification: VerificationResult = verification_engine.verify(structured)
        logger.info(
            "Verification finished in %.2fs — evidence_status=%s, asset_id=%s, conf=%s",
            time.perf_counter() - t_stage,
            verification.evidence_status.value,
            verification.asset_id,
            verification.verification_confidence,
        )
    except Exception as exc:
        # Non-fatal: If verification fails, default to UNAVAILABLE so citizen complaint is never lost.
        logger.warning(
            "Verification engine encountered non-fatal error (%s) — falling back to UNAVAILABLE.",
            exc,
            exc_info=True,
        )
        dept_code = CATEGORY_TO_DEPARTMENT.get(structured.category, DepartmentCode.GEN)
        verification = VerificationResult(
            asset_id=None,
            department_code=dept_code,
            location_lat=settings.geofence_centroid_lat,
            location_lon=settings.geofence_centroid_lon,
            verification_confidence=None,
            evidence_status=EvidenceStatus.UNAVAILABLE,
            evidence_image_path=None,
        )

    # ------------------------------------------------------------------
    # Step 4: Ticket Router & Local Persistence
    # ------------------------------------------------------------------
    t_stage = time.perf_counter()
    try:
        finalize_resp: FinalizeResponse = ticket_router.route(
            complaint=structured,
            verification=verification,
            raw_transcript=transcript,
            language=language,
            audio_path=audio_path,
        )
        logger.info(
            "Ticket routing finished in %.2fs — ticket_id=%s, urgency=%.4f",
            time.perf_counter() - t_stage,
            finalize_resp.ticket_id,
            finalize_resp.urgency_score,
        )
    except ticket_router.TicketRouterError as exc:
        logger.error("Ticket router error: %s", exc, exc_info=True)
        raise PipelineExecutionError(
            error_code="TICKET_ROUTING_FAILED",
            message="Failed to generate and save the complaint ticket. Please try again.",
            retry_allowed=True,
            status_code=500,
        ) from exc
    except Exception as exc:
        logger.error("Unexpected ticket routing failure: %s", exc, exc_info=True)
        raise PipelineExecutionError(
            error_code="TICKET_ROUTING_FAILED",
            message="An internal error occurred while saving the complaint ticket.",
            retry_allowed=False,
            status_code=500,
        ) from exc

    total_elapsed = time.perf_counter() - total_start
    logger.info(
        "Pipeline SUCCESS: session=%s  ticket_id=%s  total_time=%.2fs",
        session_id, finalize_resp.ticket_id, total_elapsed,
    )

    return finalize_resp
