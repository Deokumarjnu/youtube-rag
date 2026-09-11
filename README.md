# TechContent Studio — Phase 1 scaffold

This repository is a Phase‑1 scaffold for the "TechContent Studio" product built on top of your existing youtube-rag demo. It creates a monorepo layout and minimal working skeleton for the UI, API, AI modules, and database migrations so you can iterate toward:

PROMPT → PLAN → SCRIPT → VALIDATE → ASSETS → PREVIEW → APPROVAL → (publish)

This branch is intended as a developer starting point. It does not contain production secrets and avoids automatic publishing.

Contents
- web/: React + Vite frontend skeleton (Create Content page stub)
- api/: FastAPI backend skeleton with project & generate endpoints
- ai/: Python AI helper modules (transcription worker adapted from rag.py)
- db/migrations/: initial SQL schema for Phase‑1
- docs/: architecture and TODO notes
- infra/.env.example: environment variables (server-side secrets)

Quickstart (local development)
1. Backend
   - Create a Python venv and install dependencies:
     python -m venv .venv
     source .venv/bin/activate
     pip install -r api/requirements.txt
   - Run the API server:
     uvicorn api.main:app --reload --port 8000

2. Frontend
   - Install dependencies and run dev server:
     cd web
     npm install
     npm run dev

3. Worker (optional)
   - The ai/transcribe module contains utilities to download and transcribe audio. It's a library; you can import it into a worker script or run the module directly.

Notes
- All secrets must be provided server-side (see infra/.env.example). Do not store keys in the frontend.
- This scaffold focuses on structure and stubs; replace provider implementations and complete business logic in follow-up commits.

Next actions
- Review the scaffold and the API contracts in api/main.py
- Wire up LLM providers and CI-safe sandboxed validation
- Implement persistent storage with Supabase or Postgres using the migrations in db/

