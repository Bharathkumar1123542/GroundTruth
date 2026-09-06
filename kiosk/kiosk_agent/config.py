"""
kiosk_agent/config.py
---------------------
All runtime configuration is driven by environment variables (never hardcoded).
In production, these are set in /etc/groundtruth/kiosk.env and loaded by the
systemd unit via EnvironmentFile= (architecture.md §7.2, implementation.md §11).

For local development, copy .env.example → .env and load with:
    export $(grep -v '^#' .env | xargs)
or use `poetry run` which picks up .env automatically via uvicorn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class KioskSettings(BaseSettings):
    """
    Typed, validated configuration for the Kiosk Agent and Sync Daemon.
    Every field maps to an env var of the same name (uppercased by pydantic-settings).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",          # silently ignore unknown env vars
    )

    # ------------------------------------------------------------------
    # Kiosk identity (implementation.md §11)
    # ------------------------------------------------------------------
    kiosk_id: str = Field(
        default="DEV-KIOSK-001",
        description="Unique kiosk identifier — e.g. MH-PNQ-014. "
                    "Set via KIOSK_ID in /etc/groundtruth/kiosk.env.",
    )

    # GeoJSON polygon string defining this kiosk's village geofence.
    # Used by the Verification Engine for landmark→coordinate resolution.
    # (architecture.md §4.3)
    kiosk_geofence_geojson: str = Field(
        default='{"type":"Polygon","coordinates":[[[73.85,18.51],[73.86,18.51],[73.86,18.52],[73.85,18.52],[73.85,18.51]]]}',
        description="GeoJSON Polygon (WGS84) bounding the kiosk's service area.",
    )
    geofence_centroid_lat: float = Field(
        default=18.515,
        description="Latitude centroid of the kiosk geofence.",
    )
    geofence_centroid_lon: float = Field(
        default=73.855,
        description="Longitude centroid of the kiosk geofence.",
    )

    # ------------------------------------------------------------------
    # Sync / Gateway (architecture.md §4.7, §12.2)
    # ------------------------------------------------------------------
    gateway_url: str = Field(
        default="https://gateway.groundtruth.example",
        description="Base URL of the Regional Gateway. Set via GATEWAY_URL.",
    )
    kiosk_cert_path: Path = Field(
        default=Path("/etc/groundtruth/certs/kiosk.crt"),
        description="Path to the kiosk's X.509 client certificate (mTLS).",
    )
    kiosk_key_path: Path = Field(
        default=Path("/etc/groundtruth/certs/kiosk.key"),
        description="Path to the kiosk's private key (mTLS).",
    )
    # Set SKIP_MTLS=true for hackathon demo / local dev (architecture.md decision Q3).
    skip_mtls: bool = Field(
        default=False,
        description="Disable mTLS and fall back to HTTPS-only for local dev/demo.",
    )
    sync_poll_interval_s: int = Field(
        default=60,
        description="Seconds between Sync Daemon reachability checks (architecture.md §4.7).",
    )
    sync_base_backoff_s: int = Field(default=5)
    sync_max_backoff_s: int = Field(default=300)
    sync_max_attempts: int = Field(default=5)

    # ------------------------------------------------------------------
    # Model paths (implementation.md §11, §2.1)
    # ------------------------------------------------------------------
    asr_model_path: Path = Field(
        default=Path("models/asr/ggml-small-q8_0.bin"),
        description="Path to the quantized Whisper GGML model binary.",
    )
    llm_model_path: Path = Field(
        default=Path("models/llm/gemma-2-2b-it-groundtruth-lora.Q4_K_M.gguf"),
        description="Path to the merged+quantized Structuring Engine GGUF.",
    )
    cv_model_path: Path = Field(
        default=Path("models/cv/change-detection-unet-mbv3.onnx"),
        description="Path to the change-detection ONNX model.",
    )
    grammar_path: Path = Field(
        default=Path("kiosk_agent/grammar/structured_complaint.gbnf"),
        description="Path to the GBNF grammar file for constrained decoding.",
    )
    prompts_dir: Path = Field(
        default=Path("kiosk_agent/prompts"),
        description="Directory containing per-language prompt templates ({lang}.txt).",
    )

    # ------------------------------------------------------------------
    # ASR settings (architecture.md §4.1)
    # ------------------------------------------------------------------
    asr_max_duration_s: int = Field(
        default=90,
        description="Maximum complaint recording duration in seconds.",
    )
    asr_confidence_threshold: float = Field(
        default=0.55,
        description="ASR confidence below which a re-record prompt is shown.",
    )
    asr_language_hint: str | None = Field(
        default=None,
        description="Optional ISO-639-1 language hint passed to Whisper.",
    )

    # ------------------------------------------------------------------
    # Structuring Engine settings (architecture.md §4.2, implementation.md §4.3)
    # ------------------------------------------------------------------
    structuring_timeout_s: int = Field(
        default=20,
        description="Max seconds allowed for GBNF-constrained LLM generation.",
    )
    # Stub flag: when true, structuring_engine returns a hardcoded schema-conformant
    # object without invoking llama.cpp. Useful before the GGUF binary is available.
    stub_structuring: bool = Field(
        default=False,
        description="Return a stub StructuredComplaint without running the LLM. "
                    "Set STUB_STRUCTURING=true for dev/demo without the model binary.",
    )

    # ------------------------------------------------------------------
    # Verification Engine settings (architecture.md §4.4)
    # ------------------------------------------------------------------
    # Per-category change_score thresholds (architecture.md §4.4 rule 1).
    # Stored as a JSON string so they live in one env var; parsed below.
    verification_thresholds_json: str = Field(
        default='{"ROAD":0.4,"WATER":0.45,"ELECTRICITY":0.4,"SANITATION":0.35,"STREETLIGHT":0.4,"OTHER":0.3}',
        description="JSON object mapping category → change_score threshold.",
    )

    # ------------------------------------------------------------------
    # Urgency scoring weights (architecture.md §4.6)
    # ------------------------------------------------------------------
    urgency_weights_json: str = Field(
        default='{"severity":0.40,"recency":0.20,"category":0.25,"duplicate":0.15}',
        description="JSON object: urgency formula component weights.",
    )
    # Per-category weights (architecture.md §4.6)
    category_weights_json: str = Field(
        default='{"WATER":0.9,"ELECTRICITY":0.85,"ROAD":0.7,"SANITATION":0.6,"STREETLIGHT":0.5,"OTHER":0.3}',
        description="JSON object mapping category → category_weight in urgency formula.",
    )
    urgency_sla_breach_hours: int = Field(
        default=72,
        description="Hours unsynced before a +0.3 SLA-breach bonus is added to recency_weight.",
    )

    # ------------------------------------------------------------------
    # Storage / UI (architecture.md §10, implementation.md §4.7)
    # ------------------------------------------------------------------
    db_path: Path = Field(
        default=Path("groundtruth_kiosk.db"),
        description="Path to the local SQLite database file.",
    )
    # SQLCipher encryption key — MUST be set via env var in production.
    # If empty, falls back to unencrypted SQLite (dev/CI only).
    db_encryption_key: str = Field(
        default="",
        description="SQLCipher passphrase. Leave empty for unencrypted dev DB. "
                    "Set DB_ENCRYPTION_KEY in production.",
    )
    storage_warn_pct: int = Field(default=80, description="Yellow indicator threshold (%).")
    storage_critical_pct: int = Field(default=95, description="Red indicator threshold (%).")
    sync_warn_age_hours: int = Field(default=6, description="Yellow sync-age threshold (hours).")
    sync_critical_age_hours: int = Field(default=24, description="Red sync-age threshold (hours).")

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------
    host: str = Field(default="127.0.0.1", description="Bind address — loopback only in prod.")
    port: int = Field(default=8000)
    log_level: Literal["debug", "info", "warning", "error"] = Field(default="info")

    # ------------------------------------------------------------------
    # Parsed / derived properties
    # ------------------------------------------------------------------
    @field_validator("asr_confidence_threshold")
    @classmethod
    def _validate_asr_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("asr_confidence_threshold must be in [0.0, 1.0]")
        return v

    @model_validator(mode="after")
    def _parse_json_fields(self) -> "KioskSettings":
        # Eagerly parse JSON fields so callers get dicts, not raw strings.
        # Errors surface at startup, not mid-request.
        try:
            object.__setattr__(
                self,
                "_verification_thresholds",
                json.loads(self.verification_thresholds_json),
            )
            object.__setattr__(
                self,
                "_urgency_weights",
                json.loads(self.urgency_weights_json),
            )
            object.__setattr__(
                self,
                "_category_weights",
                json.loads(self.category_weights_json),
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in settings: {exc}") from exc
        return self

    @property
    def verification_thresholds(self) -> dict[str, float]:
        return self._verification_thresholds  # type: ignore[attr-defined]

    @property
    def urgency_weights(self) -> dict[str, float]:
        return self._urgency_weights  # type: ignore[attr-defined]

    @property
    def category_weights(self) -> dict[str, float]:
        return self._category_weights  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere instead of instantiating
# KioskSettings directly.  Tests can monkeypatch `config.settings` freely.
# ---------------------------------------------------------------------------
settings = KioskSettings()
