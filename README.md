# GroundTruth

> **Offline-first civic complaint kiosk** — voice in, verified ticket out, zero internet required.

[![Track](https://img.shields.io/badge/Track-OpenInnovation-blue)](https://github.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-yellow)](https://python.org)

---

## What it does

A resident speaks a complaint in Hindi, Marathi, or Tamil into a village kiosk.
With **no internet connection**, the kiosk:

1. **Transcribes** speech on-device (quantized Whisper)
2. **Structures** the complaint into a schema-conformant JSON record (LoRA-fine-tuned Gemma-2-2b-it with GBNF grammar constraints — the model physically cannot emit malformed output)
3. **Verifies** the complaint against a cached asset registry and satellite/drone imagery change-detection
4. **Scores urgency** and writes a ticket to a local SQLite queue
5. **Syncs** to the Municipal Backend when connectivity returns, where department staff see the ticket in a priority-ordered dashboard with transcript, structured summary, and photo evidence

See [`project_overview.md`](project_overview.md), [`architecture.md`](architecture.md), and [`implementation.md`](implementation.md) for full specification.

---

## Repository Structure

```
groundtruth/
├── kiosk/                    # On-device kiosk software
│   ├── kiosk_agent/          # FastAPI app + pipeline orchestrator
│   │   ├── grammar/          # GBNF grammar for constrained LLM decoding
│   │   └── prompts/          # Per-language prompt templates
│   ├── sync_daemon/          # Background sync service
│   ├── local_ui/             # Jinja2 templates + Alpine.js frontend
│   ├── models/               # Quantized model binaries (gitignored)
│   ├── scripts/              # fetch_models.sh, seed_demo_data.py
│   └── tests/                # pytest unit + integration tests
├── backend/                  # Municipal Backend (sync gateway + dashboard)
│   ├── regional_gateway/     # mTLS ingestion endpoint
│   ├── core_services/        # PostgreSQL+PostGIS, dedupe, urgency re-score
│   ├── department_agent/     # NATS consumer + staff dashboard
│   └── docker-compose.yml
├── training/                 # LoRA fine-tuning + eval scripts
├── .env.example              # All required env vars (copy → .env)
├── architecture.md
├── implementation.md
└── project_overview.md
```

---

## Quick Start — Kiosk (Local Dev / Hackathon Demo)

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- Ubuntu 22.04 or Windows (WSL2 recommended for production targets)

### 1. Clone & install

```bash
git clone https://github.com/your-org/groundtruth.git
cd groundtruth/kiosk
poetry install
```

### 2. Configure environment

```bash
cp ../.env.example .env
# Edit .env — at minimum set KIOSK_ID and confirm STUB_STRUCTURING=true
# for dev (real GGUF model not required for UI/pipeline development)
```

### 3. Fetch model binaries *(skip if using stub mode)*

```bash
bash scripts/fetch_models.sh
```

> Models are **not** committed to git (`.gitignore`d). `fetch_models.sh` downloads
> and SHA-256 verifies each binary from the project's model registry.

### 4. Seed demo data

```bash
poetry run python scripts/seed_demo_data.py
```

Generates a synthetic demo asset registry (10 assets with geo-polygons) and
synthetic before/after imagery pairs for the verification demo.

### 5. Run the Kiosk Agent

```bash
poetry run kiosk-agent
# Kiosk UI available at http://127.0.0.1:8000/
```

### 6. Run the Sync Daemon *(separate terminal)*

```bash
poetry run sync-daemon
```

---

## Quick Start — Municipal Backend

```bash
cd backend
docker compose up -d          # starts PostgreSQL+PostGIS, NATS, MinIO
poetry install
poetry run python -m alembic upgrade head   # run schema migrations
poetry run python scripts/seed_backend.py  # seed demo asset registry
poetry run uvicorn regional_gateway.main:app --port 8080
poetry run uvicorn department_agent.dashboard.main:app --port 8081
```

Department dashboard: `http://127.0.0.1:8081/`

---

## Running Tests

```bash
cd kiosk
poetry run pytest                        # all tests
poetry run pytest tests/test_asr_engine.py -v   # one module
poetry run pytest --cov=kiosk_agent --cov-report=term-missing
```

---

## Environment Variables

All variables are documented in [`.env.example`](.env.example).
Key variables for getting started:

| Variable | Default | Purpose |
|---|---|---|
| `KIOSK_ID` | `DEV-KIOSK-001` | Unique kiosk identifier |
| `STUB_STRUCTURING` | `true` | Skip LLM inference (dev without GGUF) |
| `SKIP_MTLS` | `true` | Use plain HTTPS for dev (disable mTLS) |
| `DB_ENCRYPTION_KEY` | *(empty)* | SQLCipher key — leave empty for dev |
| `GATEWAY_URL` | `https://gateway.groundtruth.example` | Backend sync endpoint |
| `LOG_LEVEL` | `info` | `debug` / `info` / `warning` / `error` |

---

## Model Assets

| Model | File | Size | Source |
|---|---|---|---|
| ASR | `models/asr/ggml-small-q8_0.bin` | ~500 MB | [OpenAI Whisper](https://github.com/openai/whisper) via whisper.cpp |
| Structuring LLM | `models/llm/gemma-2-2b-it-groundtruth-lora.Q4_K_M.gguf` | ~1.6 GB | Gemma-2-2b-it + LoRA fine-tune, quantized via llama.cpp |
| Change Detection | `models/cv/change-detection-unet-mbv3.onnx` | ~15 MB | MobileNetV3-U-Net, ONNX Runtime |

> **Memory budget:** All three models fit within 8 GB unified memory (Jetson Orin Nano constraint — `architecture.md §9`).

---

## Offline Demo Walkthrough

1. Start the kiosk agent (Step 5 above).
2. **Disable your network interface** (`sudo ip link set eth0 down` or disconnect Wi-Fi).
3. Open `http://127.0.0.1:8000/` in a browser (simulating the kiosk touchscreen).
4. Select a language → record a complaint → confirm the ticket is generated with a ticket ID.
5. Re-enable network → the Sync Daemon pushes the ticket automatically within 60 seconds.
6. Check the department dashboard at `http://127.0.0.1:8081/`.

This demonstrates the full offline → queue → sync → dashboard flow per
`implementation.md §14` acceptance criteria.

---

## Architecture Decisions

Key design decisions (ADRs) are documented in [`architecture.md §14`](architecture.md#14-architecture-decision-records):

- **ADR-001:** On-device quantized open models (no cloud API dependency)
- **ADR-002:** Grammar-constrained decoding (GBNF) — invalid LLM output is structurally unreachable
- **ADR-003:** SQLite WAL mode — crash-safe, no separate DB process on edge hardware
- **ADR-004:** Transport-stable sync contract — HTTPS now, USB/mesh later, same API

---

## Acceptance Criteria (MVP)

See `implementation.md §14` for the full Definition of Done. Summary:

- [ ] All 3 languages produce valid tickets **with network disabled**
- [ ] Every ticket has a non-null `urgency_score` and explicit `evidence_status`
- [ ] At least one demo asset produces `evidence_status: verified` with photo evidence
- [ ] Sync pushes queued tickets; dashboard shows them within 60 seconds of reconnect
- [ ] Structuring accuracy ≥ 90% on held-out validation set
- [ ] ASR word error rate < 25% per language
- [ ] End-to-end latency < 90 seconds p95

---

## Licence

MIT — see [LICENSE](LICENSE).
All model weights carry their own licences (Whisper: MIT; Gemma: [Gemma Terms of Use](https://ai.google.dev/gemma/terms); ONNX Runtime: Apache 2.0).
