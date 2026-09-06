"""
tests/test_orchestrator.py
--------------------------
Unit and integration tests for the Pipeline Orchestrator (kiosk_agent/orchestrator.py).

Test coverage:
  - Model loading lifecycle in stub and real modes
  - Full end-to-end pipeline execution (Stub mode & mocked engines)
  - Multi-language support (Hindi, Marathi, Tamil)
  - Mapping of ASR errors (TooLong, LowConfidence, NotReady, Empty)
  - Mapping of Structuring errors (Timeout, NotReady, InvalidOutput)
  - Non-fatal graceful degradation on Verification errors
  - Handling of TicketRouter persistence errors
  - ErrorResponse serialization
  - FastAPI /v1/complaints/{session_id}/finalize endpoint integration
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from kiosk_agent import asr_engine, orchestrator, structuring_engine, ticket_router, verification_engine
from kiosk_agent.config import settings
from kiosk_agent.db import TicketRow, db_session, init_db
from kiosk_agent.main import _sessions, app
from kiosk_agent.schemas import (
    Category,
    DepartmentCode,
    EvidenceStatus,
    FinalizeResponse,
    StructuredComplaint,
    VerificationResult,
)


@pytest.fixture(autouse=True)
def _setup_test_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate SQLite database and test settings for each test."""
    test_db = tmp_path / "test_orchestrator.db"
    monkeypatch.setattr(settings, "db_path", test_db)
    monkeypatch.setattr(settings, "stub_structuring", True)
    monkeypatch.setattr(settings, "kiosk_id", "TEST-KIOSK-01")
    init_db()
    _sessions.clear()
    yield
    _sessions.clear()


# ===========================================================================
# Model Loading Lifecycle Tests
# ===========================================================================

def test_load_all_models_stub_mode(monkeypatch: pytest.MonkeyPatch):
    """When STUB_STRUCTURING=true, load_all_models logs and returns cleanly without touching model files."""
    monkeypatch.setattr(settings, "stub_structuring", True)
    # Should not raise
    orchestrator.load_all_models()


def test_load_all_models_real_mode(monkeypatch: pytest.MonkeyPatch):
    """When STUB_STRUCTURING=false, load_all_models delegates to all three engine loaders."""
    monkeypatch.setattr(settings, "stub_structuring", False)

    with (
        patch("kiosk_agent.asr_engine.load_model") as mock_asr_load,
        patch("kiosk_agent.structuring_engine.load_model") as mock_struct_load,
        patch("kiosk_agent.verification_engine.load_model") as mock_verif_load,
    ):
        orchestrator.load_all_models()
        mock_asr_load.assert_called_once()
        mock_struct_load.assert_called_once()
        mock_verif_load.assert_called_once()


# ===========================================================================
# Successful Pipeline Execution Tests
# ===========================================================================

def test_run_pipeline_success_stub():
    """Stub mode end-to-end pipeline execution creates ticket and returns FinalizeResponse."""
    audio_bytes = b"\x00\x00" * 16000  # 1s dummy audio
    resp = orchestrator.run_pipeline(
        session_id="sess-001",
        audio_bytes=audio_bytes,
        language="hi",
    )

    assert isinstance(resp, FinalizeResponse)
    assert resp.ticket_id.startswith("GT-TEST-KIOSK-01-")
    assert resp.category == Category.ROAD
    assert resp.department_code == DepartmentCode.ROAD
    assert resp.urgency_score > 0.0

    # Verify ticket written to DB
    with db_session() as db:
        row = db.get(TicketRow, resp.ticket_id)
        assert row is not None
        assert row.language == "hi"
        assert row.kiosk_id == "TEST-KIOSK-01"


@pytest.mark.parametrize("lang", ["hi", "mr", "ta"])
def test_run_pipeline_all_supported_languages(lang: str):
    """Pipeline operates across all supported regional languages."""
    audio_bytes = b"\x00\x00" * 8000
    resp = orchestrator.run_pipeline(
        session_id=f"sess-{lang}",
        audio_bytes=audio_bytes,
        language=lang,
    )
    assert resp.ticket_id is not None
    with db_session() as db:
        row = db.get(TicketRow, resp.ticket_id)
        assert row is not None
        assert row.language == lang


# ===========================================================================
# ASR Error Mapping Tests
# ===========================================================================

def test_asr_input_too_long_mapping():
    """AsrInputTooLongError maps to PipelineExecutionError(400, retry_allowed=True)."""
    with patch(
        "kiosk_agent.asr_engine.transcribe",
        side_effect=asr_engine.AsrInputTooLongError(duration_s=95.0, max_s=90),
    ):
        with pytest.raises(orchestrator.PipelineExecutionError) as exc_info:
            orchestrator.run_pipeline("s1", b"dummy", "hi")

        err = exc_info.value
        assert err.error_code == "ASR_INPUT_TOO_LONG"
        assert err.status_code == 400
        assert err.retry_allowed is True
        assert "95.0" in err.message


def test_asr_low_confidence_mapping():
    """AsrLowConfidenceError maps to PipelineExecutionError(422, retry_allowed=True)."""
    with patch(
        "kiosk_agent.asr_engine.transcribe",
        side_effect=asr_engine.AsrLowConfidenceError(confidence=0.35, threshold=0.55),
    ):
        with pytest.raises(orchestrator.PipelineExecutionError) as exc_info:
            orchestrator.run_pipeline("s2", b"dummy", "mr")

        err = exc_info.value
        assert err.error_code == "ASR_LOW_CONFIDENCE"
        assert err.status_code == 422
        assert err.retry_allowed is True


def test_asr_model_not_ready_mapping():
    """AsrModelNotReadyError maps to PipelineExecutionError(503, retry_allowed=False)."""
    with patch(
        "kiosk_agent.asr_engine.transcribe",
        side_effect=asr_engine.AsrModelNotReadyError("Model not loaded"),
    ):
        with pytest.raises(orchestrator.PipelineExecutionError) as exc_info:
            orchestrator.run_pipeline("s3", b"dummy", "ta")

        err = exc_info.value
        assert err.error_code == "ASR_NOT_READY"
        assert err.status_code == 503
        assert err.retry_allowed is False


def test_asr_empty_transcript_mapping():
    """Empty or whitespace-only transcript maps to PipelineExecutionError(422)."""
    mock_res = MagicMock()
    mock_res.transcript = "   \n\t  "
    mock_res.asr_confidence = 0.8

    with patch("kiosk_agent.asr_engine.transcribe", return_value=mock_res):
        with pytest.raises(orchestrator.PipelineExecutionError) as exc_info:
            orchestrator.run_pipeline("s4", b"dummy", "hi")

        err = exc_info.value
        assert err.error_code == "ASR_EMPTY_TRANSCRIPT"
        assert err.status_code == 422
        assert err.retry_allowed is True


# ===========================================================================
# Structuring Error Mapping Tests
# ===========================================================================

def test_structuring_timeout_mapping():
    """StructuringTimeoutError maps to PipelineExecutionError(504, retry_allowed=True)."""
    with patch(
        "kiosk_agent.structuring_engine.structure",
        side_effect=structuring_engine.StructuringTimeoutError(timeout_s=20),
    ):
        with pytest.raises(orchestrator.PipelineExecutionError) as exc_info:
            orchestrator.run_pipeline("s5", b"dummy", "hi")

        err = exc_info.value
        assert err.error_code == "STRUCTURING_TIMEOUT"
        assert err.status_code == 504
        assert err.retry_allowed is True


def test_structuring_model_not_ready_mapping():
    """StructuringModelNotReadyError maps to PipelineExecutionError(503, retry_allowed=False)."""
    with patch(
        "kiosk_agent.structuring_engine.structure",
        side_effect=structuring_engine.StructuringModelNotReadyError("Not ready"),
    ):
        with pytest.raises(orchestrator.PipelineExecutionError) as exc_info:
            orchestrator.run_pipeline("s6", b"dummy", "hi")

        err = exc_info.value
        assert err.error_code == "STRUCTURING_NOT_READY"
        assert err.status_code == 503
        assert err.retry_allowed is False


def test_structuring_invalid_output_mapping():
    """StructuringOutputInvalidError maps to PipelineExecutionError(422, retry_allowed=True)."""
    with patch(
        "kiosk_agent.structuring_engine.structure",
        side_effect=structuring_engine.StructuringOutputInvalidError("Invalid JSON"),
    ):
        with pytest.raises(orchestrator.PipelineExecutionError) as exc_info:
            orchestrator.run_pipeline("s7", b"dummy", "hi")

        err = exc_info.value
        assert err.error_code == "STRUCTURING_INVALID"
        assert err.status_code == 422
        assert err.retry_allowed is True


# ===========================================================================
# Verification Non-Fatal Degradation Tests
# ===========================================================================

def test_verification_failure_graceful_degradation():
    """If Verification Engine raises any unexpected exception, ticket is still written with UNAVAILABLE status."""
    with patch(
        "kiosk_agent.verification_engine.verify",
        side_effect=RuntimeError("ONNX engine crashed"),
    ):
        resp = orchestrator.run_pipeline("s8", b"dummy", "hi")

        assert isinstance(resp, FinalizeResponse)
        assert resp.evidence_status == EvidenceStatus.UNAVAILABLE
        assert resp.ticket_id is not None

        with db_session() as db:
            row = db.get(TicketRow, resp.ticket_id)
            assert row is not None
            assert row.evidence_status == EvidenceStatus.UNAVAILABLE.value


# ===========================================================================
# Ticket Router Failure Tests
# ===========================================================================

def test_ticket_router_failure_mapping():
    """TicketRouterError maps to PipelineExecutionError(500)."""
    with patch(
        "kiosk_agent.ticket_router.route",
        side_effect=ticket_router.TicketRouterError("Disk full"),
    ):
        with pytest.raises(orchestrator.PipelineExecutionError) as exc_info:
            orchestrator.run_pipeline("s9", b"dummy", "hi")

        err = exc_info.value
        assert err.error_code == "TICKET_ROUTING_FAILED"
        assert err.status_code == 500


# ===========================================================================
# Error Schema Serialization Test
# ===========================================================================

def test_pipeline_execution_error_to_error_response():
    err = orchestrator.PipelineExecutionError(
        error_code="TEST_CODE",
        message="Test message",
        retry_allowed=True,
        status_code=400,
    )
    schema = err.to_error_response()
    assert schema.error_code == "TEST_CODE"
    assert schema.message == "Test message"
    assert schema.retry_allowed is True


# ===========================================================================
# FastAPI /v1/complaints/{session_id}/finalize Integration Tests
# ===========================================================================

def test_api_finalize_success():
    """Client POST to /v1/complaints/{session_id}/finalize returns 200 with HX-Redirect."""
    client = TestClient(app)

    # 1. Start record session
    rec_res = client.post("/v1/complaints/record", data={"language": "hi"})
    assert rec_res.status_code == 200
    session_id = rec_res.json()["session_id"]

    # 2. Finalize
    audio_file = io.BytesIO(b"\x00\x00" * 8000)
    fin_res = client.post(
        f"/v1/complaints/{session_id}/finalize",
        data={"language": "hi"},
        files={"audio_data": ("audio.webm", audio_file, "audio/webm")},
    )

    assert fin_res.status_code == 200
    data = fin_res.json()
    assert "ticket_id" in data
    assert fin_res.headers.get("HX-Redirect") == f"/confirmation/{data['ticket_id']}"


def test_api_finalize_session_expired():
    """Finalize on non-existent session returns 404."""
    client = TestClient(app)
    fin_res = client.post(
        "/v1/complaints/non-existent-session/finalize",
        data={"language": "hi"},
    )
    assert fin_res.status_code == 404
    assert fin_res.json()["error_code"] == "SESSION_NOT_FOUND"


def test_api_finalize_pipeline_error():
    """Pipeline errors in endpoint return corresponding HTTP status and ErrorResponse body."""
    client = TestClient(app)

    rec_res = client.post("/v1/complaints/record", data={"language": "hi"})
    session_id = rec_res.json()["session_id"]

    with patch(
        "kiosk_agent.structuring_engine.structure",
        side_effect=structuring_engine.StructuringTimeoutError(20),
    ):
        fin_res = client.post(
            f"/v1/complaints/{session_id}/finalize",
            data={"language": "hi"},
        )
        assert fin_res.status_code == 504
        body = fin_res.json()
        assert body["error_code"] == "STRUCTURING_TIMEOUT"
        assert body["retry_allowed"] is True
