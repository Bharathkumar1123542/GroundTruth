# GroundTruth — Project Overview

**Track:** OpenInnovation (open domain, local/open-source model at the core)
**Document Version:** 1.0
**Status:** Hackathon MVP Specification — Approved for Build
**Last Updated:** 2026-09-01
**Owner:** Bharath
**Companion Documents:** `architecture.md`, `implementation.md`

---

## 1. Executive Summary

GroundTruth is an offline-first civic complaint kiosk. A resident speaks a complaint in a local language into a village kiosk; an on-device, fully open-source ASR + LLM pipeline transcribes and structures the complaint into a schema-conformant record; a local computer-vision Verification Engine cross-checks the complaint against satellite/drone imagery change-detection and a cached municipal asset ownership registry; a Ticket Router assigns the correct department, attaches visual evidence, computes an urgency score, and either queues the ticket locally (no connectivity) or negotiates its position in the department's live priority queue (connectivity available). No step in the complaint-to-ticket path requires an internet connection or a proprietary API.

## 2. Problem Statement

Rural and peri-urban residents in low-connectivity areas file civic complaints (broken roads, non-functional handpumps, damaged streetlights, uncollected garbage, water leakage) through channels that fail them in three specific ways:

1. **Language exclusion.** Complaint intake systems (call centers, web/app forms) are built around a small set of dominant languages and literate, form-filling users.
2. **Unverifiable claims.** Complaints are taken at the resident's word. Departments deprioritize or ignore complaints they cannot independently corroborate, and duplicate or contradictory complaints for the same asset are not deduplicated.
3. **Misrouting and connectivity dependence.** Complaint systems assume constant connectivity to a central server to determine which department owns an asset and where a complaint should queue. In villages with intermittent or no connectivity, this either blocks complaint filing entirely or forces batch upload with no verification, no routing intelligence, and no evidence.

The result is a measurable trust gap: residents stop reporting because reports go nowhere, and departments deprioritize channels with high noise-to-signal ratio.

## 3. Solution Summary

A kiosk device (fixed installation, e.g. at a panchayat office or community center) runs GroundTruth entirely on local compute:

1. Resident speaks a complaint in their language into the kiosk microphone.
2. The **ASR Engine** (quantized Whisper) transcribes speech to text locally, on-device.
3. The **Structuring Engine** (quantized, LoRA-fine-tuned Gemma/Llama) converts the free-text transcript into a schema-conformant structured complaint (category, subcategory, description, location hint, urgency signals) using grammar-constrained decoding — the model is physically incapable of emitting malformed output.
4. The **Verification Engine** matches the structured complaint's location and category against (a) the locally cached municipal **Asset Registry** to identify the owning department and asset, and (b) locally stored satellite/drone imagery change-detection results to corroborate or contest the claim with a confidence score.
5. The **Ticket Router** assembles the final ticket (transcript, structured fields, asset match, verification confidence, evidence image, computed urgency score) and either enqueues it in the local SQLite queue (offline) or, once the **Sync Daemon** detects connectivity, transmits it to the **Municipal Backend**, which re-scores it against the live department queue and negotiates its priority position.
6. Department staff see the ticket, its transcript, its structured summary, and its supporting evidence in the **Department Agent** dashboard — never a raw, unverified voice note.

Every model in the pipeline (ASR, structuring LLM, change-detection CNN) is open-source, quantized, and runs on commodity edge hardware with no outbound API call required to produce a ticket.

## 4. Goals

| # | Goal | Metric |
|---|------|--------|
| G1 | Complaint filing works with zero connectivity | 100% of core pipeline (record → transcribe → structure → verify → queue) functions with network disabled |
| G2 | Multilingual voice-first intake | MVP supports 3 languages (Hindi, Marathi, Tamil) with <25% word error rate on field-recorded audio |
| G3 | Verified, evidence-backed tickets | ≥90% of tickets carry a non-null verification confidence score and, where imagery is available, an attached evidence image |
| G4 | Correct department routing | ≥90% of tickets on a 200-complaint labeled validation set route to the correct department code without human correction |
| G5 | End-to-end local latency | <90 seconds from end-of-speech to ticket confirmation screen on reference hardware (Jetson Orin Nano 8GB) |
| G6 | Graceful degradation under partial data | A complaint with no asset-registry match or no imagery still produces a valid, routable ticket with a lower urgency score and an explicit `unverified` flag — it is never silently dropped |

## 5. Non-Goals (MVP)

- **Not a general-purpose chatbot.** The Structuring Engine only ever emits the fixed complaint schema; it does not hold open-ended conversations.
- **Not a real-time video surveillance system.** Verification uses periodically refreshed (not live-streamed) satellite/drone imagery snapshots.
- **Not a payments, grievance-appeal, or legal-escalation system.** GroundTruth stops at ticket creation, routing, and status visibility.
- **Not responsible for last-mile repair execution.** Once a ticket reaches a Department Agent, physical repair work is outside GroundTruth's scope.
- **Not building custom ASR/LLM models from scratch.** MVP fine-tunes (LoRA) existing open checkpoints; it does not pretrain.

## 6. Target Users & Personas

| Persona | Description | Primary Need |
|---|---|---|
| **Resident (Complainant)** | Village resident, may be non-literate, speaks a regional language, limited or no smartphone/data access | Report a problem by speaking, in their language, and trust it reaches the right people |
| **Kiosk Operator** | Panchayat staff or community volunteer responsible for the physical kiosk | Kiosk stays powered, storage doesn't fill up, occasional manual sync via USB when needed |
| **Department Field Staff** | Municipal department employee (Roads, Water, Electricity, Sanitation, Streetlighting) | Receive only credible, correctly-routed, evidence-backed tickets — not raw noise |
| **District Administrator** | Oversees multiple kiosks and department queues | Visibility into ticket volume, verification rates, and department SLA compliance across the district |

## 7. Scope

### 7.1 In Scope — Hackathon MVP
- On-device ASR + structuring pipeline for 3 languages.
- Local SQLite ticket queue with offline-first operation.
- Asset Registry as a locally cached, periodically synced SQLite read replica.
- Verification Engine using a pre-loaded sample imagery change-detection dataset (live drone ingestion is out of scope for the MVP demo; the interface contract for it is fully specified in `architecture.md`).
- Ticket Router with local urgency scoring.
- Sync Daemon supporting HTTPS sync when connectivity is available.
- Department Agent web dashboard (single department, demo data) showing incoming tickets with evidence.

### 7.2 In Scope — Post-Hackathon Production Roadmap
- Live satellite/drone imagery ingestion pipeline and scheduled re-verification of unresolved tickets.
- USB "sneakernet" and LoRa mesh sync fallback (interfaces specified now, implementation deferred — see `architecture.md` §10).
- Additional language coverage beyond the initial 3.
- Multi-kiosk, multi-district Municipal Backend with horizontal scaling.
- Resident-facing ticket status lookup (SMS-based, since it must not assume smartphone access).

### 7.3 Explicitly Out of Scope
- Video calling or live agent escalation.
- Any cloud-hosted, closed-weight, or pay-per-call LLM/ASR API anywhere in the complaint path (violates the OpenInnovation track constraint and the offline requirement).
- Biometric identification of complainants.

## 8. Constraints & Assumptions

**Constraints:**
- All models in the complaint-creation path must be open-source/open-weight and must run fully on-device (no network call permitted in that path).
- Kiosk hardware must run on battery/solar power for a minimum 8-hour offline session (see `architecture.md` §9 for the power budget).
- Total on-device model footprint (ASR + LLM + CV) must fit within 8GB unified memory (reference hardware constraint).

**Assumptions:**
- At least one kiosk operator visits each kiosk periodically to perform maintenance (storage checks, manual USB sync if needed) — GroundTruth does not assume a fully unattended device.
- Satellite/drone imagery for a given district is refreshed on a periodic (not real-time) cadence and delivered to the Municipal Backend by an external data source; GroundTruth consumes this imagery, it does not capture it.
- The municipal asset ownership registry is maintained as source-of-truth by the Municipal Backend; kiosks hold a read-only cached copy.

## 9. High-Level System Diagram

```
┌─────────────────────────── KIOSK (offline-capable) ───────────────────────────┐
│                                                                                 │
│   Resident ──speaks──▶ [Local UI] ──▶ [Kiosk Agent orchestrator]               │
│                                            │                                    │
│                                            ▼                                    │
│                                     [ASR Engine]                                │
│                                            │ transcript                         │
│                                            ▼                                    │
│                                  [Structuring Engine] ── structured complaint    │
│                                            │                                    │
│                                            ▼                                    │
│              [Asset Registry (cached)] ◀── [Verification Engine] ──▶ [Imagery store] │
│                                            │ verified complaint + confidence     │
│                                            ▼                                    │
│                                     [Ticket Router] ── ticket ──▶ [Local SQLite queue] │
│                                            │                                    │
└────────────────────────────────────────────┼──────────────────────────────────┘
                                              │ (when connectivity available)
                                              ▼
                                       [Sync Daemon]
                                              │ HTTPS / mTLS
                                              ▼
                             ┌──────── MUNICIPAL BACKEND ────────┐
                             │  [Regional Gateway]                │
                             │        │                           │
                             │        ▼                           │
                             │  [Core Services: PostgreSQL+PostGIS]│
                             │        │  urgency re-score, dedupe │
                             │        ▼                           │
                             │  [Message Bus] ──▶ [Department Agent(s)] │
                             └────────────────────────────────────┘
```

## 10. Success Metrics / KPIs

| Metric | Target | Measured By |
|---|---|---|
| Offline pipeline completion rate | 100% | Automated test: full pipeline run with network interface disabled |
| ASR word error rate (per language) | <25% | Benchmark against 50 field-recorded held-out complaints/language |
| Structuring schema-validity rate | 100% | Every LLM output validated against JSON schema before ticket creation; grammar-constrained decoding makes invalid output structurally impossible |
| Routing accuracy | ≥90% | 200-complaint labeled validation set, category + department code exact match |
| End-to-end local latency | <90s (p95) | Timed from end-of-speech event to ticket confirmation screen render |
| Verification coverage | ≥90% of tickets carry a confidence score | Ticket table `verification_confidence IS NOT NULL` |
| Sync success rate (when connectivity present) | ≥95% within 3 retry attempts | `sync_log` table success/failure ratio |

## 11. Glossary

| Term | Definition |
|---|---|
| **Kiosk** | The physical edge device installed at a village location running the full offline pipeline |
| **Kiosk Agent** | The on-device orchestrator process coordinating ASR, Structuring, Verification, and Ticket Router |
| **Structured Complaint** | The JSON-schema-conformant object produced by the Structuring Engine from a raw transcript |
| **Asset Registry** | The database of municipal infrastructure assets and their owning department, cached locally on each kiosk and authoritative at the Municipal Backend |
| **Verification Confidence** | A 0–1 score produced by the Verification Engine expressing how strongly available imagery and registry data corroborate a complaint |
| **Urgency Score** | A 0–1 computed value used to order tickets within a department's priority queue (formula defined in `architecture.md` §4.6) |
| **Sync Daemon** | The on-device process responsible for detecting connectivity and transmitting queued tickets to the Municipal Backend |
| **Department Agent** | The Municipal Backend service (plus staff-facing dashboard) that receives routed tickets for one department |
| **Sneakernet Sync** | Manual ticket export/import via USB drive, used when no network connectivity is available for extended periods (production roadmap item) |

## 12. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| LLM produces plausible-sounding but incorrect structured data (hallucinated category/location) | Misrouted tickets, resident distrust | Grammar-constrained decoding restricts output to the fixed schema and enumerated category list; Verification Engine cross-checks category against Asset Registry before final routing; low-confidence tickets are flagged `unverified` rather than auto-routed silently |
| Kiosk storage fills during an extended offline period | New complaints cannot be recorded | Local SQLite queue capacity sized for 30 days at expected volume (see `architecture.md` §9); Kiosk Agent surfaces a storage-warning state in the Local UI at 80% capacity |
| ASR accuracy degrades for underrepresented dialects within a supported language | Complaints transcribed incorrectly, structuring fails downstream | Field-recorded validation set per language before language is marked "supported"; Structuring Engine flags low ASR-confidence segments for operator review rather than silently structuring garbled text |
| No imagery available for a given asset/location | Complaint cannot be visually corroborated | Verification Engine returns `verification_confidence: null` with `evidence_status: "unavailable"` rather than blocking ticket creation; urgency scoring formula treats missing evidence as neutral, not negative |
| Kiosk device is offline for longer than assumed maintenance interval | Backlog of unsynced tickets grows unbounded | Sync Daemon retries on every network-availability event; sneakernet/mesh fallback defined for production roadmap; kiosk operator alerted via Local UI when unsynced backlog exceeds threshold |

## 13. Roadmap / Milestones

| Phase | Scope | Timeframe |
|---|---|---|
| M0 — Hackathon MVP | Sections 7.1, demoable on reference hardware or laptop-simulated kiosk | Hackathon build window (see `implementation.md` §13 for hour-by-hour plan) |
| M1 — Field Pilot | 3 kiosks, single district, live drone imagery ingestion, 2-week resident-facing pilot | Post-hackathon, 6–8 weeks |
| M2 — Multi-District Rollout | Horizontal Municipal Backend scaling, sneakernet + mesh sync, additional languages | Post-pilot, dependent on M1 results |

## 14. Stakeholders

| Role | Responsibility |
|---|---|
| Project Owner (Bharath) | Overall product and technical direction |
| Kiosk Operators (panchayat/community staff) | Physical device upkeep, manual sync when required |
| Department Field Staff | Ticket triage and repair execution |
| District Administration | Cross-department oversight, SLA accountability |
