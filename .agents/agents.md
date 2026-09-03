# GroundTruth — Agent Guide

## Project Overview
GroundTruth is an offline-first civic complaint kiosk built for the OpenInnovation hackathon track. Residents speak complaints in Hindi, Marathi, or Tamil into a village kiosk; an on-device ASR + LoRA-fine-tuned Gemma-2-2b pipeline structures the speech into a formal complaint, verifies it against imagery and a municipal asset registry, and routes it to the correct department — entirely offline until sync.

## Architecture Map
- Kiosk ASR/NLU — transcribes speech (Whisper) and structures it into a formal complaint record via grammar-constrained decoding (LoRA-fine-tuned Gemma-2-2b) — talks to: Verification Engine
- Verification Engine — cross-checks the structured complaint against satellite/drone imagery change-detection and a cached municipal asset registry — talks to: Kiosk ASR/NLU, Ticket Router
- Ticket Router — scores urgency and routes the verified complaint to the correct municipal department — talks to: Verification Engine, Sync Layer
- Sync Layer — transmits queued tickets to the Municipal Backend whenever connectivity is available; pulls asset registry/imagery updates — talks to: Ticket Router, Municipal Backend
- Municipal Backend — authoritative store and department routing target (PostgreSQL + PostGIS, NATS, MinIO) — talks to: Sync Layer

## Tech Stack
- On-device pipeline: Python-based ML inference for the Whisper ASR model and the LoRA-fine-tuned Gemma-2-2b model with grammar-constrained decoding
- Verification Engine: local imagery change-detection model + cached asset registry
- Municipal Backend: PostgreSQL + PostGIS (ticket/asset storage), NATS (message bus), MinIO (evidence object storage)
- Sync transport: HTTP(S) between kiosk and Municipal Backend

## Repo & Directory Conventions
```
groundtruth/
├── kiosk/        # on-device pipeline: ASR/NLU, Verification Engine, Ticket Router
├── sync/          # Sync Layer client + protocol
├── backend/       # Municipal Backend services (Postgres/PostGIS, NATS, MinIO)
├── schemas/        # shared complaint/ticket/asset schemas used by kiosk and backend
├── docs/            # project_overview.md, architecture.md, implementation.md
└── agents.md
```

## Setup Commands
```bash
# Kiosk pipeline
[PACKAGE_MANAGER] install                 # Python version: [PYTHON_VERSION]
[COMMAND_TO_FETCH_MODELS]                  # download/quantize Whisper + Gemma-2-2b LoRA weights

# Backend services
[PACKAGE_MANAGER] install                  # backend runtime/version: [BACKEND_RUNTIME_AND_VERSION]
[COMMAND_TO_START_LOCAL_INFRA]              # e.g. docker compose up for Postgres+PostGIS, NATS, MinIO
```

## Coding Standards
- Every network call in the kiosk path must be non-blocking and wrapped in explicit no-connectivity handling — never assume the Municipal Backend is reachable.
- The complaint-creation path (ASR → structuring → verification → routing) must complete with zero network calls.
- Fail loud, not silent: a missing verification result or registry match must set an explicit status field, never be dropped.
- Keep on-device inference code resource-aware — no unbounded memory or storage growth assumptions on kiosk hardware.

## Testing Instructions
- Write unit tests per component (ASR/NLU, Verification Engine, Ticket Router) before integration tests — no CI exists yet, so agents must run tests locally and report results in the PR description.
- Any offline-first test must simulate disabled connectivity and assert the full complaint-creation path still completes and produces a valid ticket.
- Sync Layer changes must be tested against both connected and disconnected states, including reconnection after an extended offline period.

## Data & Privacy Constraints
- The kiosk must never call an external or cloud service to process complaint audio, transcripts, or images — all inference stays on-device.
- Raw audio and complaint data must never be committed to the repository, including as test fixtures.
- PII appearing in transcripts (names, phone numbers) is handled only at the Municipal Backend layer — never assume it is safe to log verbatim on-device.

## PR / Commit Instructions
- Commit format: `[component]: short imperative summary` (e.g. `verification-engine: add imagery threshold check`).
- Any change touching the Sync Layer or offline queueing must confirm, in the PR description, that the complaint-creation path still works with connectivity disabled.
- Do not merge Municipal Backend schema changes without confirming the Kiosk ASR/NLU and Sync Layer contracts still match.

## Do Not
- Do not remove or bypass the offline fallback path in the kiosk pipeline, because offline operation is the project's core constraint.
- Do not swap Gemma-2-2b or Whisper for a cloud-hosted or closed-weight model, because the complaint-creation path must run fully on-device.
- Do not commit real or resident-identifiable complaint audio, transcripts, or images as test fixtures, because this is resident data, not sample data.
- Do not invent new architecture components or change the five defined components without explicit approval, because this spec is the approved build target.
