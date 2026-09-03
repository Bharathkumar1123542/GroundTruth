# GroundTruth — Implementation

**Track:** OpenInnovation (open domain, local/open-source model at the core)
**Document Version:** 1.0
**Status:** Hackathon MVP Specification — Approved for Build
**Last Updated:** 2026-09-01
**Owner:** Bharath
**Companion Documents:** `project_overview.md`, `architecture.md`

---

## 1. Implementation Overview

This document specifies exactly what to build, in what order, to produce the GroundTruth hackathon MVP defined in `project_overview.md` §7.1, implementing the architecture defined in `architecture.md`. It is scoped for a single kiosk demo (reference hardware or a laptop-simulated kiosk) plus a minimal Municipal Backend sufficient to demonstrate sync, routing, and the Department Agent dashboard. Every module below maps directly to a component in `architecture.md` §3.

## 2. Prerequisites & Environment Setup

### 2.1 Kiosk Development Environment
```bash
# OS baseline: Ubuntu 22.04 LTS (or JetPack 6 on Jetson hardware)
sudo apt update && sudo apt install -y python3.11 python3.11-venv git cmake build-essential \
  libsqlite3-dev portaudio19-dev chromium-browser

# Python dependency management
curl -sSL https://install.python-poetry.org | python3.11 -
poetry --version   # verify install

# Project init
mkdir groundtruth-kiosk && cd groundtruth-kiosk
poetry init --name groundtruth-kiosk --python "^3.11"
poetry add fastapi uvicorn[standard] sqlalchemy sqlcipher3-binary jinja2 python-multipart \
  onnxruntime numpy pillow pydantic
poetry add pywhispercpp llama-cpp-python --no-cache

# Model assets (quantized, downloaded once, stored under models/)
mkdir -p models/asr models/llm models/cv
# ASR: ggml-small-q8_0.bin (whisper.cpp quantized multilingual small model)
# LLM: gemma-2-2b-it-groundtruth-lora.Q4_K_M.gguf (base + LoRA merged, quantized)
# CV:  change-detection-unet-mbv3.onnx
```

### 2.2 Backend Development Environment
```bash
mkdir groundtruth-backend && cd groundtruth-backend
poetry init --name groundtruth-backend --python "^3.11"
poetry add fastapi uvicorn[standard] sqlalchemy psycopg2-binary geoalchemy2 nats-py minio pydantic

# Local dev infra via Docker Compose: PostgreSQL+PostGIS, NATS, MinIO
docker compose up -d postgis nats minio
```

### 2.3 Required External Assets (must be sourced before implementation begins)
| Asset | Source | Purpose |
|---|---|---|
| `whisper-small` multilingual checkpoint | OpenAI Whisper open weights | ASR Engine base model |
| `Gemma-2-2b-it` base weights | Google (open weights, Gemma Terms of Use) | Structuring Engine base model |
| LoRA fine-tuning dataset: 300+ labeled `(transcript, structured_complaint)` pairs across Hindi/Marathi/Tamil | Manually authored/collected for this project | Structuring Engine fine-tune (§8) |
| Sample satellite/drone imagery tile pairs (before/after) for demo assets | Publicly available open imagery (e.g. Sentinel-2 or equivalent open dataset) cropped to demo geofence | Verification Engine demo dataset |
| Demo Asset Registry seed data (10–20 assets with geo-polygons) | Manually authored for the demo geofence | Asset Registry |

## 3. Repository Structure

```
groundtruth/
├── kiosk/
│   ├── kiosk_agent/
│   │   ├── main.py                # systemd entrypoint, FastAPI app
│   │   ├── orchestrator.py        # session state machine
│   │   ├── asr_engine.py
│   │   ├── structuring_engine.py
│   │   ├── verification_engine.py
│   │   ├── ticket_router.py
│   │   ├── db.py                  # SQLite/SQLCipher models + session
│   │   └── grammar/
│   │       └── structured_complaint.gbnf
│   ├── sync_daemon/
│   │   ├── main.py
│   │   └── sync_client.py
│   ├── local_ui/
│   │   ├── templates/
│   │   └── static/
│   ├── models/                    # quantized model binaries (gitignored, fetched via script)
│   ├── scripts/
│   │   └── fetch_models.sh
│   └── pyproject.toml
├── backend/
│   ├── regional_gateway/
│   ├── core_services/
│   │   ├── models.py              # SQLAlchemy + GeoAlchemy2 models
│   │   ├── routing.py
│   │   └── dedupe.py
│   ├── department_agent/
│   │   ├── consumer.py
│   │   └── dashboard/
│   ├── docker-compose.yml
│   └── pyproject.toml
├── training/
│   ├── lora_finetune.py           # Structuring Engine LoRA training script
│   ├── dataset/                   # labeled (transcript, structured_complaint) pairs
│   └── eval/
│       ├── asr_wer_eval.py
│       └── structuring_accuracy_eval.py
├── project_overview.md
├── architecture.md
└── implementation.md
```

## 4. Module-by-Module Implementation Spec

### 4.1 `kiosk_agent/orchestrator.py` — Kiosk Agent
- Implements a session state machine with states: `IDLE → RECORDING → TRANSCRIBING → STRUCTURING → VERIFYING → ROUTING → QUEUED → DONE`, plus an `ERROR` state reachable from any step.
- `POST /v1/complaints/record` transitions `IDLE → RECORDING`, returns a `session_id` (UUID4).
- `POST /v1/complaints/{session_id}/finalize` drives the session through `TRANSCRIBING → STRUCTURING → VERIFYING → ROUTING → QUEUED`, calling each engine module in sequence and persisting the resulting ticket via `db.py`.
- On any engine raising an exception or returning a confidence below its defined threshold (ASR §4.2 below), transition to `ERROR` and return a structured error to the Local UI (`{ error_code, message, retry_allowed: true }`) rather than propagating a raw exception.

### 4.2 `kiosk_agent/asr_engine.py` — ASR Engine
```python
def transcribe(audio_pcm16_16khz: bytes, language_hint: str | None) -> AsrResult:
    """
    Runs ggml-small-q8_0 via pywhispercpp.
    Returns AsrResult(transcript: str, language_detected: str,
                       asr_confidence: float, segment_confidences: list[float]).
    Raises AsrLowConfidenceError if asr_confidence < 0.55 (per architecture.md §4.1).
    Max input duration: 90 seconds — longer input is rejected with AsrInputTooLongError
    before inference begins.
    """
```
- Model load happens once at process start (not per-request) to keep p95 latency within the 90s end-to-end budget (`architecture.md` §9).

### 4.3 `kiosk_agent/structuring_engine.py` — Structuring Engine
```python
def structure(transcript: str, language: str) -> StructuredComplaint:
    """
    Loads structured_complaint.gbnf as the llama.cpp grammar and runs
    gemma-2-2b-it-groundtruth-lora.Q4_K_M.gguf with grammar-constrained decoding.
    Prompt template is language-tagged (see prompts/{language}.txt) but the
    grammar file is shared across all languages — only the prompt varies.
    Returns StructuredComplaint conforming exactly to architecture.md §6.1.
    Raises StructuringTimeoutError if generation exceeds 20 seconds.
    """
```
- The GBNF grammar file enumerates the six category values verbatim; the model architecture cannot emit a seventh.

### 4.4 `kiosk_agent/verification_engine.py` — Verification Engine
```python
def verify(structured_complaint: StructuredComplaint,
           kiosk_geofence: Polygon) -> VerificationResult:
    """
    1. Resolve location_hint to a coordinate within kiosk_geofence via fuzzy
       match against known landmark labels in the cached Asset Registry.
    2. Query cached asset_registry for the containing geo_polygon -> asset_id, department_code.
    3. If asset_id resolved and imagery exists in the Imagery Store for it:
         run change-detection ONNX model on the before/after tile pair,
         compare change_score against the category-specific threshold
         (architecture.md §4.4) to set verification_confidence + evidence_status.
       Else: set evidence_status per architecture.md §4.4 rules 2/3, verification_confidence = None.
    Returns VerificationResult(asset_id, department_code, verification_confidence,
                                evidence_status, evidence_image_path).
    """
```

### 4.5 `kiosk_agent/ticket_router.py` — Ticket Router
```python
def route(structured: StructuredComplaint, verification: VerificationResult,
          kiosk_id: str) -> Ticket:
    """
    Computes urgency_score per the formula in architecture.md §4.6.
    Assigns ticket_id = f"GT-{kiosk_id}-{today:%Y%m%d}-{next_seq:05d}".
    Persists the Ticket row (schema: architecture.md §6.2) with status='QUEUED'.
    """
```
- `next_seq` is read from a per-kiosk, per-day counter table (`ticket_seq`) incremented atomically within the same SQLite transaction as the ticket insert, preventing ID collisions under concurrent finalize calls.

### 4.6 `sync_daemon/main.py` — Sync Daemon
- Runs as an independent `systemd` service (not a thread inside `kiosk_agent`), polling reachability every 60 seconds per `architecture.md` §4.7.
- On reachability, calls `sync_client.push_batch()` and `sync_client.pull_registry_delta()` per the contracts in `architecture.md` §12.2.
- Retry policy implemented exactly as specified in `architecture.md` §4.7 (exponential backoff, base 5s, cap 5min, max 5 attempts/cycle).

### 4.7 `local_ui/` — Local UI
- Single-page kiosk interface served by the Kiosk Agent's FastAPI app at `/`.
- Screens: **Language Select** → **Record** (large touch-target microphone button, live waveform indicator, 90s countdown) → **Confirmation** (shows `category`, `department_code`, `ticket_id`, and a human-readable status: "Verified with photo evidence" / "Recorded — evidence pending" depending on `evidence_status`) → **Idle** (auto-return after 10s).
- Storage/sync status indicator in the header: green (storage <80%, last sync <1hr), yellow (storage 80–95% or last sync >6hr), red (storage >95% or last sync >24hr) — thresholds sourced from `architecture.md` §10.

## 5. Backend Module Implementation

### 5.1 `core_services/models.py`
- SQLAlchemy models mirroring `architecture.md` §6.2/§6.3, with `geo_polygon` mapped via GeoAlchemy2 `Geometry('POLYGON', srid=4326)` for native PostGIS queries.

### 5.2 `core_services/dedupe.py`
```python
def dedupe_incoming(ticket: Ticket) -> Ticket | None:
    """
    Query existing tickets with the same asset_id, created within the last
    15 minutes, department_code matching. If found: merge — retain the
    ticket with higher verification_confidence as canonical, append the
    other's evidence_image_path to a supplementary evidence list, discard
    the duplicate as a standalone queue entry. Returns None if merged
    into an existing ticket, else returns the ticket to be inserted fresh.
    """
```

### 5.3 `core_services/routing.py`
- Re-computes `duplicate_weight` using fleet-wide counts (all kiosks, all districts) per `architecture.md` §4.6, then re-computes `urgency_score`, then publishes to the NATS subject `tickets.{department_code}`.

### 5.4 `department_agent/consumer.py` + `dashboard/`
- Subscribes to its department's NATS subject.
- Dashboard: server-rendered (Jinja2 or equivalent) table sorted by `urgency_score` descending, each row expandable to show transcript, structured complaint JSON, evidence image (served from MinIO via a signed URL), and a status-update control (`ACKNOWLEDGED → RESOLVED / REJECTED`, written back to Core Services).

## 6. Data Models & Schemas

All schemas are defined authoritatively in `architecture.md` §6 (Structured Complaint JSON Schema §6.1, Ticket table DDL §6.2, Asset Registry DDL §6.3, Sync Log DDL §6.4). Implementation must not diverge from these without a corresponding update to `architecture.md`.

### 6.1 GBNF Grammar File (`kiosk_agent/grammar/structured_complaint.gbnf`)
The grammar must enforce, at minimum:
- `category` restricted to exactly: `ROAD`, `WATER`, `ELECTRICITY`, `SANITATION`, `STREETLIGHT`, `OTHER`.
- `structuring_confidence` restricted to a decimal between `0.0` and `1.0` inclusive.
- `subcategory` and `description` restricted to printable UTF-8 strings up to their schema-defined max lengths (64 and 280 characters respectively).
- `urgency_keywords` restricted to a JSON array of 0–5 string elements.

## 7. API Specifications

Implement exactly the endpoints defined in `architecture.md` §12.1 (Kiosk Internal API) and §12.2 (Sync API). Request/response payloads must validate against Pydantic models generated directly from those schemas — do not hand-roll parallel schema definitions in the backend that could drift from `architecture.md`.

## 8. Model Training & Fine-Tuning Pipeline

### 8.1 Dataset Requirements
- Minimum 300 labeled `(transcript, structured_complaint)` pairs for LoRA fine-tuning, distributed across the 3 MVP languages (target: 100+ per language), covering all 6 categories with at least 20 examples each.
- Each example: raw transcript (as a native speaker would plausibly phrase a complaint) paired with a hand-labeled, schema-conformant JSON target.

### 8.2 Training Script (`training/lora_finetune.py`)
```python
"""
Fine-tunes Gemma-2-2b-it with LoRA (rank=16, alpha=32, target_modules=["q_proj","v_proj"])
on the (transcript, structured_complaint) dataset.
Training config: 3 epochs, learning_rate=2e-4, batch_size=4, gradient_accumulation_steps=4.
Output: LoRA adapter weights, subsequently merged into the base model and
quantized to Q4_K_M GGUF via llama.cpp's convert + quantize scripts for on-device deployment.
"""
```

### 8.3 Evaluation Scripts
- `training/eval/asr_wer_eval.py`: computes word error rate against 50 held-out field-recorded complaints per language; fails the build gate if WER exceeds 25% for any MVP language.
- `training/eval/structuring_accuracy_eval.py`: computes exact-match accuracy on `category` and `department_code` against a held-out 200-example labeled validation set; fails the build gate if accuracy is below 90%.

## 9. Testing Strategy

| Test Type | Scope | Tooling |
|---|---|---|
| Unit tests | Each engine module in isolation (`asr_engine`, `structuring_engine`, `verification_engine`, `ticket_router`) with mocked model outputs | `pytest` |
| Schema validation tests | Every Structuring Engine output (including adversarial/edge-case transcripts) validated against the JSON schema in `architecture.md` §6.1 | `pytest` + `pydantic` |
| Integration tests | Full pipeline run with network interface disabled, asserting a valid ticket is produced end-to-end | `pytest` + `docker network disconnect` in CI |
| Sync reconciliation tests | Simulate offline queue buildup, then reconnect and verify `QUEUED → SYNCED → ACKNOWLEDGED` transitions and correct dedupe behavior at Backend | `pytest` against `docker compose` backend stack |
| Model quality gates | ASR WER and Structuring accuracy thresholds (§8.3) | CI job blocking merge if thresholds are not met |
| Field pilot testing (post-MVP) | 3 kiosks, 2-week resident-facing pilot per `project_overview.md` §13 (M1) | Manual data collection + `structuring_accuracy_eval.py` re-run against pilot data |

## 10. Deployment & Provisioning

### 10.1 Kiosk Imaging
1. Flash reference OS image (Ubuntu 22.04 + JetPack 6, or Raspberry Pi OS) to the device's NVMe/SD storage.
2. Run `scripts/fetch_models.sh` to download and verify (SHA-256 checksum) all quantized model binaries into `kiosk/models/`.
3. Provision a unique X.509 device certificate (per `architecture.md` §8) and place it under `/etc/groundtruth/certs/`.
4. Install `kiosk_agent` and `sync_daemon` as `systemd` services (`Restart=on-failure`, `WantedBy=multi-user.target`).
5. Configure Chromium to launch in kiosk mode pointed at `http://localhost:8000/` on boot via an `.xinitrc`/`systemd` autostart entry.
6. Seed the local Asset Registry cache and Imagery Store with the demo dataset (§2.3) for the initial offline period before the first sync.

### 10.2 Backend Provisioning
1. `docker compose up -d` to bring up PostgreSQL+PostGIS, NATS, and MinIO.
2. Run Alembic migrations to create the Core Services schema (mirroring `architecture.md` §6.2/§6.3 with PostGIS geometry types).
3. Seed the authoritative Asset Registry with the demo dataset.
4. Issue and register kiosk device certificates against the Regional Gateway's trusted CA.

## 11. Configuration Management

| Setting | Location | Example |
|---|---|---|
| Kiosk ID | `/etc/groundtruth/kiosk.env` | `KIOSK_ID=MH-PNQ-014` |
| Kiosk geofence polygon | `/etc/groundtruth/kiosk.env` | GeoJSON polygon string |
| Regional Gateway URL | `/etc/groundtruth/kiosk.env` | `GATEWAY_URL=https://gateway.groundtruth.example` |
| ASR/LLM/CV model paths | `kiosk_agent/config.py`, defaults under `models/` | overridable via env var for testing with smaller stub models |
| Category thresholds (verification) | `verification_engine.py` constants, sourced from `architecture.md` §4.4 | `{"ROAD": 0.4, "SANITATION": 0.35, ...}` |
| Urgency scoring weights | `ticket_router.py` constants, sourced from `architecture.md` §4.6 | `{"severity": 0.40, "recency": 0.20, "category": 0.25, "duplicate": 0.15}` |
| Sync retry policy | `sync_daemon/config.py` | `{"base_backoff_s": 5, "max_backoff_s": 300, "max_attempts": 5}` |

## 12. Monitoring & Logging

- Every engine module logs structured JSON log lines (`timestamp`, `session_id`, `ticket_id`, `component`, `event`, `duration_ms`) to a local rotating log file (`/var/log/groundtruth/kiosk-agent.log`).
- `GET /v1/health` (architecture.md §12.1) is polled locally every 5 minutes by a `systemd` timer; a failing health check writes a `WARN`-level log entry and surfaces a visible status change in the Local UI header.
- `sync_log` table (architecture.md §6.4) serves as the durable audit trail for every sync attempt; the Backend's Regional Gateway additionally logs every ingested batch with kiosk ID, ticket count, and result for cross-kiosk observability.
- Department Agent dashboard surfaces a rolling count of tickets by `evidence_status` so staff can see verification coverage at a glance.

## 13. Implementation Timeline (Hackathon Build Plan, 36-hour window)

| Hours | Milestone |
|---|---|
| 0–4 | Environment setup (§2); fetch/quantize model assets; scaffold repository structure (§3); stub FastAPI apps for Kiosk Agent and Backend |
| 4–10 | ASR Engine implementation and integration (§4.2); Local UI recording flow skeleton (§4.7, Language Select + Record screens) |
| 10–18 | GBNF grammar (§6.1) + Structuring Engine (§4.3); LoRA fine-tuning pipeline run on the seed dataset (§8); integrate structuring into the orchestrator |
| 18–24 | Asset Registry seed data + Verification Engine (§4.4) with the pre-loaded demo imagery pairs |
| 24–30 | Ticket Router + urgency scoring (§4.5, §4.6); Local SQLite Queue; Confirmation screen in Local UI |
| 30–34 | Sync Daemon (§4.6) + minimal Backend (Regional Gateway, Core Services, single Department Agent dashboard) demonstrating end-to-end sync |
| 34–36 | Offline-mode demo rehearsal (network disabled end-to-end); polish Local UI and dashboard; prepare demo script |

## 14. Acceptance Criteria / Definition of Done

A build is considered demo-ready when all of the following hold:

1. A complaint recorded in each of the 3 MVP languages produces a valid, schema-conformant ticket with the network interface disabled throughout.
2. Every produced ticket has a non-null `urgency_score` and an explicit `evidence_status` (never null).
3. At least one demo asset with pre-loaded before/after imagery produces `evidence_status: "verified"` with a non-null `verification_confidence` and an attached evidence image.
4. At least one demo complaint with no matching asset produces `evidence_status: "no_asset_match"` and still reaches `status: QUEUED` (never silently dropped).
5. Re-enabling network connectivity triggers the Sync Daemon to push all `QUEUED` tickets, and the Department Agent dashboard displays the synced ticket with its transcript, structured summary, and evidence image within 60 seconds of sync completion.
6. `structuring_accuracy_eval.py` reports ≥90% category/department exact-match accuracy on the held-out validation set (§8.3).
7. `asr_wer_eval.py` reports <25% WER for each of the 3 MVP languages (§8.3).
8. End-to-end local latency (speech end → confirmation screen) is <90 seconds p95 across a 10-run timed benchmark on reference hardware.
