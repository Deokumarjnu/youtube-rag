from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
import uuid

app = FastAPI(title="TechContent Studio - API (Phase 1)")

# In-memory store (replace with DB in later commits)
PROJECTS: Dict[str, Dict[str, Any]] = {}

class CreateProjectRequest(BaseModel):
    title: str
    prompt: str
    content_type: str = "DSA"
    duration_seconds: int = 60
    language: str = "English"


@app.post("/api/projects")
async def create_project(req: CreateProjectRequest):
    project_id = str(uuid.uuid4())
    project = {
        "id": project_id,
        "title": req.title,
        "prompt": req.prompt,
        "content_type": req.content_type,
        "duration_seconds": req.duration_seconds,
        "language": req.language,
        "status": "draft",
    }
    PROJECTS[project_id] = project
    return project


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    project = PROJECTS.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    return project


@app.post("/api/projects/{project_id}/generate-plan")
async def generate_plan(project_id: str):
    # Stub: planner returns structured plan JSON
    project = PROJECTS.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")

    plan = {
        "topic": project["title"],
        "category": project["content_type"],
        "duration": project["duration_seconds"],
        "learning_objective": "Explain the core idea and provide one example.",
        "key_concepts": ["intuition", "algorithm", "example"],
    }
    # In Phase 1 we persist to in-memory store. Replace with DB in later commits.
    project["plan"] = plan
    return {"project": project, "plan": plan}


@app.post("/api/projects/{project_id}/generate-script")
async def generate_script(project_id: str):
    project = PROJECTS.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")

    # Stub: structured script JSON (hook + scenes)
    script = {
        "hook": "Why does Kadane's algorithm solve Maximum Subarray in O(n)?",
        "scenes": [
            {"start": 0, "end": 6, "narration": "Hook: What if you could find the max subarray in one pass?"},
            {"start": 6, "end": 20, "narration": "Problem: Define Maximum Subarray..."},
            {"start": 20, "end": 45, "narration": "Solution: Kadane's algorithm maintains a running sum..."},
            {"start": 45, "end": project["duration_seconds"], "narration": "Complexity and CTA."},
        ],
    }
    project["script"] = script
    return {"project": project, "script": script}


@app.post("/api/projects/{project_id}/validate-script")
async def validate_script(project_id: str):
    project = PROJECTS.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")

    # Stub validator: basic checks
    script = project.get("script")
    validator_report = {"status": "ok", "issues": []}
    if not script:
        validator_report["status"] = "failed"
        validator_report["issues"].append("No script found")

    project["validator_report"] = validator_report
    return {"project": project, "validator_report": validator_report}

