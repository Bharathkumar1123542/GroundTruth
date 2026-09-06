"""
kiosk_agent/main.py
--------------------
FastAPI application — Kiosk Agent entry-point.

Responsibilities:
  1. Serves the Local UI (Jinja2 templates) at the root path.
  2. Exposes the Kiosk Internal API (architecture.md §12.1):
       POST /v1/complaints/record
       POST /v1/complaints/{session_id}/finalize
       GET  /v1/tickets/{ticket_id}
       GET  /v1/tickets
       POST /v1/sync/trigger
       GET  /v1/health
       GET  /v1/health/partial    (HTMX partial for the status bar)
  3. Initialises the database (init_db) and loads engine modules on startup.
  4. Binds only to loopback (127.0.0.1) — never exposed beyond localhost
     (architecture.md §12.1).

Pipeline integration (Phase 2–6):
  The /v1/complaints/{session_id}/finalize endpoint currently returns a STUB
  response when STUB_STRUCTURING=true (default for dev). Once the real engine
  modules (asr_engine, structuring_engine, verification_engine, ticket_router)
  are implemented in Phases 2–5, the orchestrator.py module (Phase 6) will
  replace the stub call below.

systemd entrypoint:
  The `start()` function at the bottom is referenced by pyproject.toml's
  [tool.poetry.scripts] kiosk-agent entry point.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from kiosk_agent import orchestrator
from kiosk_agent.config import settings
from kiosk_agent.db import (
    AssetRegistryRow,
    ImageryRow,
    SyncLogRow,
    TicketRow,
    db_session,
    get_storage_pct_used,
    init_db,
    next_ticket_seq,
    queued_ticket_count,
)
from kiosk_agent.schemas import (
    Category,
    DepartmentCode,
    EvidenceStatus,
    FinalizeResponse,
    HealthResponse,
    RecordResponse,
    SyncTriggerResponse,
    Ticket,
    TicketStatus,
    TicketSummary,
    ErrorResponse,
    StructuredComplaint,
    VerificationResult,
    CATEGORY_TO_DEPARTMENT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_UI_DIR = _HERE.parent / "local_ui"
_TEMPLATES_DIR = _UI_DIR / "templates"
_STATIC_DIR = _UI_DIR / "static"

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# ---------------------------------------------------------------------------
# In-memory session store
# A real production system would persist sessions in SQLite.
# For the MVP, sessions are transient (lost on restart — acceptable since
# a restart discards any recording-in-progress anyway, implementation.md §10).
# ---------------------------------------------------------------------------

_sessions: dict[str, dict] = {}  # session_id → {language, created_at, status}


# ---------------------------------------------------------------------------
# Application lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup: initialise DB, pre-load engine models.
    Shutdown: flush logs.
    """
    logger.info("Kiosk Agent starting — KIOSK_ID=%s", settings.kiosk_id)

    # Initialise SQLite schema (idempotent).
    init_db()

    # Pre-load all ML models (ASR, Structuring, Verification)
    try:
        orchestrator.load_all_models()
    except Exception as exc:
        logger.error("Failed to load ML models during startup: %s", exc)

    yield  # application runs

    logger.info("Kiosk Agent shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GroundTruth Kiosk Agent",
    description="On-device API for the GroundTruth civic complaint kiosk.",
    version="0.1.0",
    docs_url=None,    # disable Swagger UI in kiosk mode
    redoc_url=None,
    lifespan=lifespan,
)

# Serve static files (CSS, JS, vendor libs) from /static.
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Template context helpers
# ---------------------------------------------------------------------------

def _base_ctx(request: Request) -> dict:
    """Common context injected into every template render."""
    return {
        "request": request,
        "kiosk_id": settings.kiosk_id,
    }


# ---------------------------------------------------------------------------
# UI Routes — Local UI screens
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def language_select(request: Request) -> HTMLResponse:
    """Screen 1 — Language selection."""
    return templates.TemplateResponse(
        "language_select.html",
        {**_base_ctx(request)},
    )


@app.get("/record/{session_id}", response_class=HTMLResponse, include_in_schema=False)
async def record_screen(request: Request, session_id: str) -> HTMLResponse:
    """Screen 2 — Recording. Reached via HX-Redirect after POST /v1/complaints/record."""
    session = _sessions.get(session_id)
    if not session:
        # Session expired or invalid — redirect to language select.
        return HTMLResponse(
            content="",
            status_code=302,
            headers={"Location": "/"},
        )
    return templates.TemplateResponse(
        "record.html",
        {
            **_base_ctx(request),
            "session_id": session_id,
            "language": session["language"],
            "max_duration": settings.asr_max_duration_s,
        },
    )


@app.get("/confirmation/{ticket_id}", response_class=HTMLResponse, include_in_schema=False)
async def confirmation_screen(request: Request, ticket_id: str) -> HTMLResponse:
    """Screen 3 — Confirmation. Reached via HX-Redirect after finalize."""
    with db_session() as session:
        row = session.get(TicketRow, ticket_id)
        if not row:
            return HTMLResponse(content="", status_code=302, headers={"Location": "/"})

    return templates.TemplateResponse(
        "confirmation.html",
        {
            **_base_ctx(request),
            "ticket_id": row.ticket_id,
            "category": row.category,
            "department_code": row.department_code,
            "urgency_score": row.urgency_score,
            "evidence_status": row.evidence_status,
            "language": row.language,
        },
    )


# ---------------------------------------------------------------------------
# Kiosk Internal API — architecture.md §12.1
# ---------------------------------------------------------------------------

@app.post(
    "/v1/complaints/record",
    response_model=RecordResponse,
    summary="Begin a complaint recording session",
)
async def start_record(
    language: str = Form(..., description="ISO-639-1 language code: hi | mr | ta"),
) -> Response:
    """
    Validates the selected language, creates an in-memory session, and
    returns an HX-Redirect header pointing the Local UI to /record/{session_id}.

    architecture.md §12.1:
      POST /v1/complaints/record → { session_id }
    """
    # Validate language (mirrors RecordRequest validator).
    supported = {"hi", "mr", "ta"}
    if language not in supported:
        err = ErrorResponse(
            error_code="UNSUPPORTED_LANGUAGE",
            message=f"Language '{language}' is not supported. Choose from: {', '.join(sorted(supported))}.",
            retry_allowed=True,
        )
        return JSONResponse(status_code=422, content=err.model_dump())

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "language": language,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "RECORDING",
    }
    logger.info("Session created: %s  language=%s", session_id, language)

    # HTMX follows HX-Redirect for full-page navigation.
    return Response(
        content=RecordResponse(session_id=session_id).model_dump_json(),  # type: ignore[arg-type]
        status_code=200,
        media_type="application/json",
        headers={"HX-Redirect": f"/record/{session_id}"},
    )


@app.post(
    "/v1/complaints/{session_id}/finalize",
    summary="Run the full ASR → Structuring → Verification → Ticket pipeline",
)
async def finalize_complaint(
    session_id: str,
    language: str = Form(...),
    audio_data: UploadFile | None = File(default=None),
) -> Response:
    """
    Drives the complaint pipeline for a session using orchestrator.run_pipeline().

    architecture.md §12.1:
      POST /v1/complaints/{session_id}/finalize →
        { ticket_id, category, department_code, urgency_score, evidence_status }
    """
    session = _sessions.get(session_id)
    if not session:
        err = ErrorResponse(
            error_code="SESSION_NOT_FOUND",
            message="Recording session expired. Please start again.",
            retry_allowed=False,
        )
        return JSONResponse(status_code=404, content=err.model_dump())

    # Read audio bytes (passed to orchestrator / ASR).
    audio_bytes: bytes = b""
    if audio_data:
        audio_bytes = await audio_data.read()
        logger.info(
            "Audio received: session=%s  size=%d bytes  content_type=%s",
            session_id, len(audio_bytes), audio_data.content_type,
        )
    else:
        logger.warning("No audio_data in finalize request for session=%s", session_id)

    try:
        finalize_resp = orchestrator.run_pipeline(
            session_id=session_id,
            audio_bytes=audio_bytes,
            language=language,
        )
        _sessions.pop(session_id, None)  # clean up session

        return Response(
            content=finalize_resp.model_dump_json(),
            status_code=200,
            media_type="application/json",
            headers={"HX-Redirect": f"/confirmation/{finalize_resp.ticket_id}"},
        )
    except orchestrator.PipelineExecutionError as exc:
        logger.warning("Pipeline execution failed for session %s: %s", session_id, exc)
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_error_response().model_dump(),
        )
    except Exception as exc:
        logger.error("Unhandled error during pipeline finalize: %s", exc, exc_info=True)
        err = ErrorResponse(
            error_code="INTERNAL_ERROR",
            message="An unexpected system error occurred. Please try again.",
            retry_allowed=True,
        )
        return JSONResponse(status_code=500, content=err.model_dump())



# ---------------------------------------------------------------------------
# GET /v1/tickets/{ticket_id}
# ---------------------------------------------------------------------------

@app.get(
    "/v1/tickets/{ticket_id}",
    summary="Retrieve a ticket from the local queue",
)
async def get_ticket(ticket_id: str) -> JSONResponse:
    """architecture.md §12.1 — GET /v1/tickets/{ticket_id}"""
    with db_session() as db:
        row = db.get(TicketRow, ticket_id)
        if not row:
            raise HTTPException(status_code=404, detail="Ticket not found.")
    return JSONResponse(content=_ticket_row_to_dict(row))


# ---------------------------------------------------------------------------
# GET /v1/tickets
# ---------------------------------------------------------------------------

@app.get(
    "/v1/tickets",
    summary="List all tickets in the local queue",
)
async def list_tickets() -> JSONResponse:
    """architecture.md §12.1 — GET /v1/tickets"""
    from sqlalchemy import select
    with db_session() as db:
        rows = db.execute(
            select(TicketRow).order_by(TicketRow.urgency_score.desc())
        ).scalars().all()
    return JSONResponse(content=[_ticket_row_to_dict(r) for r in rows])


def _ticket_row_to_dict(row: TicketRow) -> dict:
    return {
        "ticket_id":              row.ticket_id,
        "kiosk_id":               row.kiosk_id,
        "created_at":             row.created_at,
        "language":               row.language,
        "category":               row.category,
        "department_code":        row.department_code,
        "asset_id":               row.asset_id,
        "evidence_status":        row.evidence_status,
        "verification_confidence":row.verification_confidence,
        "urgency_score":          row.urgency_score,
        "status":                 row.status,
        "sync_attempts":          row.sync_attempts,
        "last_sync_attempt":      row.last_sync_attempt,
    }


# ---------------------------------------------------------------------------
# POST /v1/sync/trigger
# ---------------------------------------------------------------------------

@app.post(
    "/v1/sync/trigger",
    response_model=SyncTriggerResponse,
    summary="Manually trigger a sync attempt",
)
async def trigger_sync() -> SyncTriggerResponse:
    """
    architecture.md §12.1 — POST /v1/sync/trigger
    Signals the Sync Daemon to attempt an immediate push.
    MVP: writes a flag file that the Sync Daemon polls.
    Phase 7 (sync_daemon): replace with an IPC signal / named pipe.
    """
    flag_path = settings.db_path.parent / ".sync_trigger"
    try:
        flag_path.touch(exist_ok=True)
        logger.info("Sync trigger flag written to %s", flag_path)
    except OSError as exc:
        logger.warning("Could not write sync trigger flag: %s", exc)
    return SyncTriggerResponse(triggered=True)


# ---------------------------------------------------------------------------
# GET /v1/health  (JSON — used by systemd health-check timer)
# ---------------------------------------------------------------------------

@app.get(
    "/v1/health",
    response_model=HealthResponse,
    summary="Health of all on-device engines",
)
async def health() -> HealthResponse:
    """architecture.md §12.1 — GET /v1/health"""
    storage_pct = get_storage_pct_used()
    with db_session() as db:
        unsynced = queued_ticket_count(db)

    # Engine status — "ok" always in Phase 1 (stub mode).
    # Phase 2–5 will query actual engine readiness flags.
    engine_status = "ok" if settings.stub_structuring else "ok"  # updated per phase

    return HealthResponse(
        asr=engine_status,
        structuring=engine_status,
        verification=engine_status,
        storage_pct_used=storage_pct,
        unsynced_tickets=unsynced,
        last_sync_at=None,  # Phase 7: Sync Daemon writes this to a state file
    )


# ---------------------------------------------------------------------------
# GET /v1/health/partial  (HTMX partial — polled by status bar in base.html)
# ---------------------------------------------------------------------------

@app.get("/v1/health/partial", response_class=HTMLResponse, include_in_schema=False)
async def health_partial(request: Request) -> HTMLResponse:
    """
    Returns a minimal HTML fragment (the status-bar div) polled every 30s
    by HTMX in base.html. Avoids a full-page reload just to update indicators.

    Thresholds from implementation.md §4.7:
      Storage: green <80%, yellow 80-95%, red >95%
      Sync age: green <1hr, yellow >6hr, red >24hr  (MVP: always green — Phase 7)
    """
    storage_pct = get_storage_pct_used()
    with db_session() as db:
        unsynced = queued_ticket_count(db)

    # Determine storage CSS class.
    if storage_pct >= settings.storage_critical_pct:
        storage_cls  = "crit"
        storage_label = f"Storage {storage_pct:.0f}%"
    elif storage_pct >= settings.storage_warn_pct:
        storage_cls  = "warn"
        storage_label = f"Storage {storage_pct:.0f}%"
    else:
        storage_cls  = "ok"
        storage_label = f"Storage {storage_pct:.0f}%"

    # Sync status — Phase 7 will compute sync age from sync_log.
    # For Phase 1, show unsynced count with green indicator.
    sync_cls   = "ok"
    sync_label = f"Sync ({unsynced} queued)" if unsynced else "Sync ✓"

    html = f"""
<div
  class="status-bar"
  role="status"
  aria-live="polite"
  aria-label="System status"
  id="status-bar"
  hx-get="/v1/health/partial"
  hx-trigger="every 30s"
  hx-swap="outerHTML"
  hx-target="#status-bar"
>
  <div class="status-indicator status-indicator--{storage_cls}"
       aria-label="Storage status: {storage_label}">
    <span class="status-indicator__dot"></span>
    <span>{storage_label}</span>
  </div>
  <div class="status-indicator status-indicator--{sync_cls}"
       aria-label="Sync status: {sync_label}">
    <span class="status-indicator__dot"></span>
    <span>{sync_label}</span>
  </div>
</div>
""".strip()
    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# systemd entry-point
# ---------------------------------------------------------------------------

def start() -> None:
    """
    Entry-point called by `poetry run kiosk-agent` (pyproject.toml §scripts).
    Binds to loopback only — never exposed beyond localhost (architecture.md §12.1).
    """
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    uvicorn.run(
        "kiosk_agent.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,      # never reload in production; use systemd restart
        access_log=True,
    )


if __name__ == "__main__":
    start()
