# GroundTruth — Architecture

**Track:** OpenInnovation (open domain, local/open-source model at the core)
**Document Version:** 1.0
**Status:** Hackathon MVP Specification — Approved for Build
**Last Updated:** 2026-09-01
**Owner:** Bharath
**Companion Documents:** `project_overview.md`, `implementation.md`

---

## 1. Architecture Principles

1. **Offline-first, sync-second.** Every component required to produce a valid, routable ticket runs on-device with no network dependency. Connectivity only affects *when* a ticket reaches the Municipal Backend, never *whether* it can be created.
2. **Open-weight, on-device models only, in the complaint-creation path.** No proprietary or hosted-API model may sit between "resident speaks" and "ticket queued locally." This is a hard constraint, not a preference.
3. **Structured output by construction, not by validation-after-the-fact.** The Structuring Engine uses grammar-constrained decoding (GBNF) so the LLM cannot emit a token sequence outside the defined JSON schema. Downstream components never receive malformed structured data.
4. **Evidence over assertion.** Every ticket carries an explicit `verification_confidence` and `evidence_status` field. A complaint that cannot be corroborated is never silently upgraded to "verified"; it is explicitly marked `unverified` and still routed.
5. **Degrade, never drop.** Missing imagery, missing asset match, or low ASR confidence each produce a lower-confidence ticket with an explicit flag — never a discarded complaint.

## 2. System Context Diagram

```
                     ┌────────────────────────────────────────────┐
                     │                  RESIDENT                   │
                     └──────────────────────┬───────────────────────┘
                                             │ speech (local language)
                                             ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                                   KIOSK                                     │
│  (Jetson Orin Nano 8GB / Raspberry Pi 5 8GB — offline-capable edge device)  │
│                                                                              │
│   Local UI ─▶ Kiosk Agent ─▶ ASR Engine ─▶ Structuring Engine ─▶            │
│                    ▲                                     │                  │
│                    │                                     ▼                  │
│                    │                          Verification Engine ◀──────┐  │
│                    │                                     │               │  │
│                    │                                     ▼               │  │
│                    │                              Ticket Router          │  │
│                    │                                     │               │  │
│                    │                                     ▼               │  │
│                    └──────────────────────────  Local SQLite Queue       │  │
│                                                           │               │  │
│                                                           ▼               │  │
│                                                     Sync Daemon           │  │
│                                             Asset Registry (cached) ──────┘  │
│                                             Imagery Store (cached)            │
└──────────────────────────────────┬───────────────────────────────────────┘
                                    │ HTTPS + mTLS (opportunistic)
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                             MUNICIPAL BACKEND                                │
│  Regional Gateway ─▶ Core Services (PostgreSQL+PostGIS, urgency re-score,   │
│  dedupe) ─▶ Message Bus (NATS) ─▶ Department Agent(s) ─▶ Staff Dashboard    │
│  Object Storage (MinIO, evidence images)   Asset Registry (source of truth) │
└────────────────────────────────────────────────────────────────────────────┘
```

## 3. Component Inventory

| Component | Location | Responsibility |
|---|---|---|
| Local UI | Kiosk | Touchscreen interface: record button, language selector, confirmation screen, storage/sync status indicators |
| Kiosk Agent | Kiosk | Orchestrates the full pipeline for one complaint session; owns session state machine |
| ASR Engine | Kiosk | Speech-to-text transcription, on-device, quantized Whisper |
| Structuring Engine | Kiosk | Transcript → schema-conformant structured complaint, quantized LoRA-fine-tuned LLM with grammar-constrained decoding |
| Asset Registry (cached) | Kiosk | Read-only local SQLite replica of municipal asset ownership data, refreshed via Sync Daemon |
| Imagery Store (cached) | Kiosk | Local store of pre-loaded satellite/drone change-detection results, keyed by asset/geohash |
| Verification Engine | Kiosk | Cross-checks structured complaint against Asset Registry and Imagery Store; produces `verification_confidence` and `evidence_status` |
| Ticket Router | Kiosk | Assembles final ticket, computes local `urgency_score`, writes to Local SQLite Queue |
| Local SQLite Queue | Kiosk | Durable, offline-capable ticket store (WAL mode) |
| Sync Daemon | Kiosk | Detects connectivity, transmits queued tickets, pulls Asset Registry/Imagery deltas |
| Regional Gateway | Municipal Backend | mTLS termination, kiosk authentication, batch ticket ingestion, rate limiting |
| Core Services | Municipal Backend | Authoritative ticket store, urgency re-scoring, geospatial dedupe, Asset Registry source of truth |
| Message Bus | Municipal Backend | Decouples Core Services from Department Agents (NATS) |
| Department Agent | Municipal Backend | Per-department ticket queue consumer and staff dashboard |
| Object Storage | Municipal Backend | Durable evidence image storage (MinIO) |

## 4. Component Detail

### 4.1 ASR Engine
- **Model:** `whisper-small` (244M params), multilingual checkpoint, quantized to INT8 GGML (`ggml-small-q8_0`, ~500MB) via `whisper.cpp`.
- **Input:** 16kHz mono PCM audio buffer captured by Local UI (max 90 seconds per complaint recording, enforced by Kiosk Agent).
- **Output:** `{ transcript: string, language_detected: string, segment_confidences: [float], asr_confidence: float }`.
- **Behavior on low confidence:** if `asr_confidence < 0.55`, the Kiosk Agent routes back to the Local UI for a re-record prompt before invoking the Structuring Engine.

### 4.2 Structuring Engine
- **Model:** `Gemma-2-2b-it`, LoRA-fine-tuned (rank 16, target modules: `q_proj`, `v_proj`) on a labeled complaint-structuring dataset, quantized to Q4_K_M GGUF (~1.6GB), served via `llama.cpp`.
- **Decoding constraint:** GBNF grammar enforces output conformance to the Structured Complaint Schema (§6.1) — the model cannot emit a token outside the grammar's production rules.
- **Input:** ASR transcript + detected language code.
- **Output:** Structured Complaint object (schema in §6.1).
- **Category enum (fixed, MVP):** `ROAD`, `WATER`, `ELECTRICITY`, `SANITATION`, `STREETLIGHT`, `OTHER`.

### 4.3 Asset Registry (cached)
- **Storage:** local SQLite table, read-only from the Kiosk Agent's perspective; writable only by the Sync Daemon during a registry-delta pull.
- **Match strategy:** structured complaint's `location_hint` is resolved to an approximate coordinate (kiosk-configured village geofence + resident-provided landmark text, matched via fuzzy string match against known asset labels within the geofence); resolved coordinate is matched against Asset Registry geo-polygons to find the owning `asset_id` and `department_code`.
- **No-match behavior:** if no asset polygon contains the resolved coordinate, `asset_id` is `null` and `department_code` falls back to the Structuring Engine's predicted category-to-department mapping (§4.6 table).

### 4.4 Imagery Store (cached) + Verification Engine
- **Imagery Store:** pre-loaded pairs of "before/after" geotagged raster tiles per known asset, refreshed periodically by the Sync Daemon from the Municipal Backend's imagery pipeline (production roadmap: live drone ingestion; MVP: pre-loaded sample dataset per `project_overview.md` §7.1).
- **Change-detection model:** lightweight U-Net (MobileNetV3 encoder), ONNX Runtime, produces a per-pixel change mask and a scalar `change_score` (0–1) for the asset's tile pair.
- **Verification logic:**
  1. If `asset_id` is resolved and imagery exists for it: compare `change_score` against a category-specific threshold (e.g. `ROAD` potholes: 0.4; `SANITATION` garbage accumulation: 0.35) to produce `verification_confidence`.
  2. If `asset_id` is resolved but no imagery exists: `verification_confidence = null`, `evidence_status = "unavailable"`.
  3. If `asset_id` could not be resolved: `verification_confidence = null`, `evidence_status = "no_asset_match"`.
- **Output:** `{ verification_confidence: float|null, evidence_status: "verified"|"unavailable"|"no_asset_match"|"contested", evidence_image_path: string|null }`. `"contested"` is set when `change_score` is below threshold despite a resolved asset match (imagery does not support the claim); the ticket is still created, flagged for department review rather than rejected.

### 4.5 Ticket Router
- Assembles the final ticket record (schema in §6.2) from: Structured Complaint, Verification result, Asset Registry match, and computed `urgency_score`.
- Assigns `ticket_id` using the scheme `GT-{KIOSK_ID}-{YYYYMMDD}-{SEQ:05d}` (e.g. `GT-MH-PNQ-014-20260901-00042`), where `SEQ` is a per-kiosk, per-day monotonic counter.
- Writes the ticket to the Local SQLite Queue with `status = QUEUED`.

### 4.6 Urgency Scoring Formula
```
urgency_score = (0.40 × severity_confidence)
              + (0.20 × recency_weight)
              + (0.25 × category_weight)
              + (0.15 × duplicate_weight)
```
- `severity_confidence`: the Verification Engine's `change_score` if available, else `0.5` (neutral) if `evidence_status = "unavailable"`, else `0.3` if `evidence_status = "no_asset_match"`.
- `recency_weight`: `min(1.0, days_since_last_ticket_for_asset / 30)` — assets with no recent complaint history score lower urgency by default; resets to a full `+0.3` flat bonus if the *current* ticket has been locally queued unsynced for more than 72 hours (SLA-breach escalation).
- `category_weight`: fixed per category — `WATER: 0.9`, `ELECTRICITY: 0.85`, `ROAD: 0.7`, `SANITATION: 0.6`, `STREETLIGHT: 0.5`, `OTHER: 0.3` (reflects public-health/safety priority ordering).
- `duplicate_weight`: `min(1.0, duplicate_complaint_count_for_asset / 5)` — the local kiosk counts prior tickets it has raised for the same `asset_id` within 30 days; the Municipal Backend re-computes this across all kiosks at sync time.
- **Category → default department code mapping** (used when no asset match exists): `ROAD → ROAD`, `WATER → WATER`, `ELECTRICITY → ELEC`, `SANITATION → SANI`, `STREETLIGHT → LIGHT`, `OTHER → GEN`.

### 4.7 Sync Daemon
- Polls network reachability every 60 seconds (HTTPS `HEAD` request to Regional Gateway health endpoint).
- On reachability: pushes all `QUEUED` tickets in FIFO-by-`urgency_score`-descending order via `POST /sync/v1/tickets/batch` (contract in §12.2); pulls Asset Registry and Imagery Store deltas via `GET /sync/v1/registry/delta`.
- Retry policy: exponential backoff (base 5s, cap 5min), max 5 attempts per sync cycle before marking the cycle failed and logging to `sync_log`; next reachability check resumes the cycle.
- Ticket status transitions on successful push: `QUEUED → SYNCED`. On Municipal Backend acknowledgment of department receipt: `SYNCED → ACKNOWLEDGED`.

### 4.8 Municipal Backend — Regional Gateway
- Terminates mutual TLS using per-kiosk device certificates (provisioned at kiosk imaging time, §7).
- Enforces rate limiting per kiosk (default: 100 tickets/hour, configurable) to bound backend load from a single misbehaving device.
- Forwards authenticated, validated batches to Core Services.

### 4.9 Municipal Backend — Core Services
- **Datastore:** PostgreSQL 16 with PostGIS extension.
- **Responsibilities:** authoritative ticket persistence; re-computation of `urgency_score` using fleet-wide `duplicate_weight` (all kiosks, not just the originating one); geospatial dedupe (tickets for the same `asset_id` within a 15-minute ingestion window are merged, retaining the highest `verification_confidence` and the union of evidence images); publishes routed tickets to the Message Bus keyed by `department_code`.

### 4.10 Municipal Backend — Department Agent
- Consumes its department's topic from the Message Bus.
- Persists to a per-department queue view (ordered by `urgency_score` descending).
- Serves the Staff Dashboard (read-only web UI: ticket list, transcript, structured summary, evidence image, verification confidence, status update controls).

## 5. Data Flow — End-to-End Sequence (Happy Path)

```
Resident          Local UI        Kiosk Agent      ASR Engine   Structuring Engine   Verification Engine   Ticket Router   Local Queue   Sync Daemon   Municipal Backend
   │ speaks           │                │               │               │                    │                  │              │             │                │
   │──audio──────────▶│                │               │               │                    │                  │              │             │                │
   │                  │──start session▶│               │               │                    │                  │              │             │                │
   │                  │                │──raw audio───▶│               │                    │                  │              │             │                │
   │                  │                │◀─transcript───│               │                    │                  │              │             │                │
   │                  │                │──transcript──────────────────▶│                    │                  │              │             │                │
   │                  │                │◀─structured complaint─────────│                    │                  │              │             │                │
   │                  │                │──structured complaint────────────────────────────▶│                  │              │             │                │
   │                  │                │◀─verification result─────────────────────────────│                  │              │             │                │
   │                  │                │──verified complaint──────────────────────────────────────────────▶│              │             │                │
   │                  │                │                                                                     │──ticket────▶│             │                │
   │                  │◀─confirmation screen (ticket_id, category, department)──────────────────────────────│              │             │                │
   │                  │                │                                                                     │              │◀─QUEUED tickets─│             │
   │                  │                │                                                                     │              │             │──batch push───▶│
   │                  │                │                                                                     │              │             │◀─ack + queue pos│
   │                  │                │                                                                     │              │◀─SYNCED status──│                │
```

## 6. Data Models

### 6.1 Structured Complaint Schema (Structuring Engine output, GBNF-enforced)
```json
{
  "category": "ROAD | WATER | ELECTRICITY | SANITATION | STREETLIGHT | OTHER",
  "subcategory": "string, free text, max 64 chars",
  "description": "string, normalized complaint summary, max 280 chars",
  "location_hint": "string, resident-described landmark or address fragment",
  "reported_asset_type": "string, e.g. 'handpump', 'streetlight_pole', 'road_segment'",
  "urgency_keywords": ["string", "..."],
  "structuring_confidence": "float, 0.0-1.0"
}
```

### 6.2 Ticket Schema (Local SQLite `tickets` table)
```sql
CREATE TABLE tickets (
  ticket_id             TEXT PRIMARY KEY,   -- GT-{KIOSK_ID}-{YYYYMMDD}-{SEQ:05d}
  kiosk_id              TEXT NOT NULL,
  created_at            TEXT NOT NULL,      -- ISO-8601 UTC
  language              TEXT NOT NULL,      -- e.g. 'hi', 'mr', 'ta'
  raw_transcript         TEXT NOT NULL,
  structured_complaint   TEXT NOT NULL,      -- JSON, conforms to §6.1
  category               TEXT NOT NULL,
  department_code        TEXT NOT NULL,      -- ROAD | WATER | ELEC | SANI | LIGHT | GEN
  asset_id                TEXT,               -- FK asset_registry.asset_id, nullable
  location_lat            REAL,
  location_lon            REAL,
  evidence_image_path     TEXT,
  evidence_image_hash     TEXT,               -- SHA-256, integrity check
  verification_confidence REAL,               -- nullable
  evidence_status          TEXT NOT NULL,      -- verified | unavailable | no_asset_match | contested
  urgency_score            REAL NOT NULL,
  status                   TEXT NOT NULL,      -- CREATED | QUEUED | SYNCED | ACKNOWLEDGED | RESOLVED | REJECTED
  sync_attempts             INTEGER NOT NULL DEFAULT 0,
  last_sync_attempt         TEXT
);
```

### 6.3 Asset Registry Schema (Local cache and Backend source of truth)
```sql
CREATE TABLE asset_registry (
  asset_id           TEXT PRIMARY KEY,   -- {DEPT_CODE}-{ASSET_TYPE}-{DISTRICT_CODE}-{SEQ}
  asset_type         TEXT NOT NULL,
  department_code    TEXT NOT NULL,
  district_code      TEXT NOT NULL,
  geo_polygon        TEXT NOT NULL,      -- GeoJSON polygon, WGS84
  registry_version   INTEGER NOT NULL,
  last_synced_at     TEXT                -- kiosk-local cache only
);
```

### 6.4 Sync Log Schema
```sql
CREATE TABLE sync_log (
  sync_id        INTEGER PRIMARY KEY AUTOINCREMENT,
  ticket_id      TEXT NOT NULL,
  sync_method    TEXT NOT NULL,   -- HTTPS (MVP); USB, MESH (production roadmap)
  initiated_at   TEXT NOT NULL,
  completed_at   TEXT,
  result         TEXT NOT NULL,   -- SUCCESS | FAILED | CONFLICT
  error_detail   TEXT
);
```

## 7. Deployment Architecture

### 7.1 Kiosk Hardware (Reference Configuration)
| Component | Specification |
|---|---|
| Compute | NVIDIA Jetson Orin Nano 8GB Developer Kit (primary); Raspberry Pi 5 8GB + Coral USB Accelerator (budget fallback) |
| OS | Ubuntu 22.04 LTS + JetPack 6 (Jetson) / Raspberry Pi OS 64-bit (Pi) |
| Audio input | 4-microphone circular far-field USB array |
| Display | 10.1" resistive/capacitive touchscreen |
| Storage | 256GB NVMe SSD |
| Power | 20W solar panel + 20,000mAh UPS power bank, rated for 8-hour continuous offline operation |
| Connectivity | Onboard Wi-Fi/Ethernet + optional 4G/LTE USB dongle for opportunistic sync |

### 7.2 Kiosk Software Stack
- Python 3.11, dependency management via Poetry.
- `whisper.cpp` (via `pywhispercpp` bindings) for ASR.
- `llama.cpp` (via `llama-cpp-python`) for the Structuring Engine, with GBNF grammar file loaded at process start.
- ONNX Runtime for the Verification Engine's change-detection model.
- SQLite (WAL mode) for the Local SQLite Queue and cached Asset Registry.
- FastAPI serving a localhost-only internal API (§12.1); no port exposed beyond loopback.
- Local UI: FastAPI + Jinja2 + HTMX + Alpine.js, rendered fullscreen in Chromium kiosk mode (`--kiosk` flag) on the touchscreen.
- Each on-device process (`kiosk-agent`, `sync-daemon`) managed as a `systemd` unit with `Restart=on-failure`.

### 7.3 Municipal Backend Deployment
- Regional Gateway: containerized, horizontally scalable behind a load balancer, mTLS termination.
- Core Services: PostgreSQL 16 + PostGIS, single primary with read replicas for the Staff Dashboard.
- Message Bus: NATS cluster, one subject per `department_code`.
- Object Storage: MinIO, bucket-per-district.
- All Backend components are open-source and self-hostable — no managed proprietary cloud service is required by the architecture, consistent with the OpenInnovation constraint on the pipeline's model layer and the project's broader open-source posture.

## 8. Security Architecture

- **Device identity:** each kiosk is provisioned with a unique X.509 client certificate at imaging time (§`implementation.md` §10), used for mutual TLS on every sync connection. A kiosk without a valid, non-revoked certificate cannot push tickets.
- **Data at rest:** Local SQLite Queue and Asset Registry cache are encrypted at rest using SQLCipher with a key derived from a hardware-backed secret where available (TPM/secure element on Jetson), falling back to a filesystem-permission-restricted key file on Pi-class hardware.
- **PII minimization:** raw audio is discarded immediately after successful transcription by default (configurable retention for debugging); transcripts and structured complaints are retained for 90 days; a regex + lightweight NER pass redacts phone numbers and full names mentioned in transcripts before they are persisted or synced.
- **Evidence integrity:** every evidence image is hashed (SHA-256) at capture time; the hash is stored alongside the ticket and re-verified at the Municipal Backend on ingestion to detect tampering in transit.
- **Backend access control:** Department Agent dashboards are scoped per department via role-based access control; District Administrators have cross-department read access only, not write access to other departments' queues.

## 9. Scalability & Performance

| Dimension | Target | Basis |
|---|---|---|
| End-to-end local latency | <90s p95, resident speech end → confirmation screen | ASR (~8s for 30s audio on Jetson GPU) + Structuring (~15s for 200-token structured output) + Verification (~5s) + overhead |
| Kiosk offline autonomy | 30 days of local queue capacity at 50 complaints/day | ~1,500 tickets × ~10MB avg (transcript + structured JSON + evidence image) ≈ 15GB, well within 256GB storage |
| Kiosk power draw | <15W idle, <45W peak during inference | Reference hardware (Jetson Orin Nano) TDP envelope |
| Backend ticket ingestion | 10,000 tickets/hour per Regional Gateway instance | Horizontal scaling via additional Gateway instances behind load balancer; Core Services writes are the eventual bottleneck, mitigated by PostgreSQL connection pooling and batched inserts |
| Department Agent read latency | <500ms p95 for queue view render | Read replica + indexed `urgency_score` column |

## 10. Failure Modes & Resilience

| Failure | Detection | Recovery |
|---|---|---|
| ASR/Structuring model fails to load at boot | `kiosk-agent` health check on `systemd` start | `systemd` `Restart=on-failure` with exponential backoff; Local UI shows "kiosk unavailable" state rather than a silent hang |
| Kiosk loses power mid-complaint | Session not finalized in Local SQLite Queue (`status` remains absent/`CREATED`) | On next boot, Kiosk Agent discards incomplete sessions older than 5 minutes; resident is prompted to re-record — no partial/corrupt ticket is ever synced |
| Sync push fails after max retries | `sync_log.result = FAILED` | Ticket remains `status = QUEUED`; next reachability check re-attempts automatically; no ticket is ever marked `SYNCED` without Backend acknowledgment |
| Regional Gateway unreachable for extended period (production) | Sync Daemon reachability check consistently fails | Production roadmap: sneakernet (USB export/import) and LoRa mesh relay through the nearest reachable kiosk, both consuming the same `POST /sync/v1/tickets/batch` contract, just over a different transport (interface stable, transport swappable) |
| Duplicate tickets for the same real-world issue from multiple kiosks | Detected at Core Services via geospatial + category + time-window clustering | Automatic merge at Backend (§4.9); originating kiosks are not required to coordinate with each other |
| Asset Registry cache is stale relative to Backend source of truth | `registry_version` mismatch detected on next sync | Sync Daemon pulls incremental delta (`GET /sync/v1/registry/delta?since_version=N`); kiosk continues operating on its (possibly stale) cache in the interim rather than blocking complaint intake |

## 11. Technology Stack Summary

| Layer | Technology | License |
|---|---|---|
| ASR | Whisper (small, multilingual), whisper.cpp | MIT |
| Structuring LLM | Gemma-2-2b-it + LoRA fine-tune, llama.cpp | Gemma Terms of Use / MIT (runtime) |
| Change detection | ONNX Runtime, MobileNetV3-encoder U-Net | Apache 2.0 |
| Kiosk backend | Python 3.11, FastAPI, SQLite/SQLCipher | MIT / BSD |
| Kiosk frontend | Jinja2, HTMX, Alpine.js, Chromium kiosk mode | MIT |
| Backend datastore | PostgreSQL 16 + PostGIS | PostgreSQL License |
| Message bus | NATS | Apache 2.0 |
| Object storage | MinIO | AGPLv3 |
| Transport security | mTLS (X.509 device certs) | n/a |

## 12. API Contracts

### 12.1 Kiosk Internal API (localhost-only, consumed by Local UI)
| Method & Path | Purpose | Response |
|---|---|---|
| `POST /v1/complaints/record` | Begin a recording session | `{ session_id }` |
| `POST /v1/complaints/{session_id}/finalize` | Trigger ASR → Structuring → Verification → Ticket Router pipeline | `{ ticket_id, category, department_code, urgency_score, evidence_status }` |
| `GET /v1/tickets/{ticket_id}` | Retrieve ticket status | Ticket record (§6.2, minus internal fields) |
| `GET /v1/tickets` | List local queue | Array of ticket summaries |
| `POST /v1/sync/trigger` | Manually force a sync attempt | `{ triggered: true }` |
| `GET /v1/health` | Health of all on-device engines | `{ asr: "ok", structuring: "ok", verification: "ok", storage_pct_used: float }` |

### 12.2 Kiosk ↔ Municipal Backend Sync API (HTTPS + mTLS)
| Method & Path | Purpose | Request | Response |
|---|---|---|---|
| `POST /sync/v1/tickets/batch` | Push queued tickets | `{ kiosk_id, tickets: [ticket...], evidence_images: [multipart] }` | `{ acknowledged: [ticket_id...], queue_positions: { ticket_id: int } }` |
| `GET /sync/v1/registry/delta?since_version=N` | Pull Asset Registry updates | — | `{ current_version: int, updated_assets: [asset...], deleted_asset_ids: [string...] }` |
| `GET /sync/v1/imagery/delta?since_version=N` | Pull Imagery Store updates | — | `{ current_version: int, updated_tiles: [tile_ref...] }` |
| `POST /sync/v1/heartbeat` | Kiosk telemetry | `{ kiosk_id, storage_pct_used, unsynced_ticket_count, last_boot_at }` | `{ acknowledged: true }` |

## 13. Non-Functional Requirements

| Category | Requirement |
|---|---|
| Availability | Kiosk complaint-intake path must have zero dependency on Backend availability; Backend availability affects only sync timing |
| Durability | No ticket transitions to `SYNCED` without an explicit Backend acknowledgment; local queue survives kiosk power loss (SQLite WAL) |
| Observability | Every state transition (`CREATED → QUEUED → SYNCED → ACKNOWLEDGED`) is timestamped and queryable via `GET /v1/tickets` |
| Portability | Kiosk software stack must run unmodified on both Jetson Orin Nano and Raspberry Pi 5 reference hardware |
| Localization | Adding a new supported language requires only a new Whisper language code entry and a language-tagged LoRA prompt template — no code change to the pipeline orchestration logic |

## 14. Architecture Decision Records

**ADR-001: On-device quantized open models instead of cloud inference APIs.**
*Decision:* Run ASR, structuring, and change-detection entirely on-device using quantized open-weight models.
*Rationale:* The offline requirement (project_overview.md §4, G1) makes any cloud-API dependency in the complaint-creation path a hard failure condition, not a degraded-mode fallback. The OpenInnovation track further requires a local/open-source model at the core.
*Alternative considered:* Hybrid mode calling a cloud LLM when connectivity is present, falling back to a smaller local model offline. Rejected because it introduces two divergent code paths with different accuracy/behavior profiles for the same feature, complicating testing and validation.

**ADR-002: Grammar-constrained decoding (GBNF) for the Structuring Engine.**
*Decision:* Constrain LLM output generation to a fixed grammar rather than validating free-form output after generation.
*Rationale:* Post-hoc validation-and-retry wastes latency budget and can still fail repeatedly on ambiguous transcripts. Grammar constraints make invalid output structurally unreachable, guaranteeing G3/G4 (project_overview.md §4) latency and correctness targets are achievable.
*Alternative considered:* Few-shot prompting with JSON-mode and a validation retry loop. Rejected due to non-deterministic failure modes under the target 90-second latency budget.

**ADR-003: SQLite (WAL mode) for the Local SQLite Queue and Asset Registry cache.**
*Decision:* Use SQLite rather than an embedded key-value store (e.g. LevelDB/RocksDB) for kiosk-local persistence.
*Rationale:* The Ticket Router and Verification Engine require relational queries (geo-polygon containment via cached bounding-box pre-filter, duplicate counting by `asset_id` and time window) that a pure key-value store would require re-implementing in application code. WAL mode gives crash-safe durability without a separate write-ahead-log implementation.
*Alternative considered:* PostgreSQL running locally on the kiosk. Rejected as unnecessary operational overhead for a single-writer, resource-constrained edge device.

**ADR-004: Interface-stable, transport-swappable sync (HTTPS now, USB/mesh later).**
*Decision:* Define the sync contract (§12.2) as transport-agnostic; the MVP implements only the HTTPS transport, but USB and LoRa mesh transports (production roadmap) will serialize the same request/response contract.
*Rationale:* Avoids building three divergent sync implementations; the Ticket Router and Sync Daemon logic remain unchanged regardless of which transport carries the batch.

**ADR-005: PostGIS for the Municipal Backend's authoritative Asset Registry.**
*Decision:* Use PostgreSQL + PostGIS rather than a specialized geospatial database or a plain relational table with manually computed bounding boxes.
*Rationale:* Native polygon-containment queries (`ST_Contains`) are required both for asset resolution during registry updates and for the Core Services' geospatial deduplication step; PostGIS is open-source, self-hostable, and consistent with the project's open-source-stack posture.
