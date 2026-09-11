# Architecture — Phase 1

This document describes the initial architecture for TechContent Studio Phase‑1 (scaffold).

Goals
- Provide a reproducible, testable developer layout
- Implement core data model and API contracts for Planner/Script/Validator
- Keep rendering and publishing out of the critical path until approval

Components
- Frontend (web/): React + TypeScript single-page app for the dashboard and Create Content flow. Communicates with API via REST.

- Backend (api/): FastAPI service that implements project endpoints, agent orchestration endpoints, and job orchestration hooks. Runs server-side secrets and provider integrations.

- AI library (ai/): Reusable Python package with provider abstractions and helper functions (transcription, embeddings ingestion). Designed to be executed in workers or within Edge Functions.

- Database (db/): Postgres / Supabase. Migration files included to create core tables. Supabase is recommended for Auth + Storage integration.

- Storage: Supabase Storage or S3 compatible. Store generated assets under projects/{projectId}/...

- Worker / Job runner: Simple background worker for long-running tasks (TTS, rendering, transcription). In early dev this can be a process started manually; later deploy as Cloud Run / worker pool.

Phases mapping
- Phase 1: UI + Planner + Script generation + Validation + Metadata + manual preview and approval
- Phase 2: TTS, programmatic visuals, captions, thumbnails, FFmpeg renderer
- Phase 3: YouTube OAuth integration and publish workflow

Security
- Keep secrets server-side; use environment variables
- Sandbox any code execution (Docker/runner) and set strict time & memory limits

