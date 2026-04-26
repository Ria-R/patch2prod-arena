from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .env import Patch2ProdEnv
from .models import Action
from .tasks import find_task_for_ci_event, list_tasks


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class GitlabWebhookPayload(BaseModel):
    object_kind: Optional[str] = None
    event_type: Optional[str] = None
    project: dict[str, Any] = Field(default_factory=dict)
    commit: dict[str, Any] = Field(default_factory=dict)
    object_attributes: dict[str, Any] = Field(default_factory=dict)
    build_name: Optional[str] = None
    build_stage: Optional[str] = None
    build_status: Optional[str] = None
    ref: Optional[str] = None


class JenkinsWebhookPayload(BaseModel):
    name: Optional[str] = None
    displayName: Optional[str] = None
    url: Optional[str] = None
    full_url: Optional[str] = None
    build: dict[str, Any] = Field(default_factory=dict)


class CIEvent(BaseModel):
    provider: str
    source_event: str
    status: str
    project: str
    branch: str
    commit_sha: str
    pipeline_id: str
    job_name: str
    task_id: Optional[str] = None
    service: Optional[str] = None
    log_url: Optional[str] = None
    timestamp: str
    raw: dict[str, Any] = Field(default_factory=dict)


DEFAULT_TASK = os.getenv("PATCH2PROD_DEFAULT_TASK", "authsdk_mobile_contract_break")
env = Patch2ProdEnv(default_task_id=DEFAULT_TASK)
app = FastAPI(title="Patch2Prod Arena", version="0.1.0")
CI_WEBHOOK_TOKEN = os.getenv("PATCH2PROD_CI_WEBHOOK_TOKEN", "patch2prod-demo-token")
ci_events: list[dict[str, Any]] = []
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "Patch2Prod Arena",
        "description": "CI repair + blast-radius + release decision OpenEnv-style environment",
        "endpoints": ["POST /reset", "POST /step", "GET /state", "GET /health"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def tasks():
    return {"tasks": list_tasks()}


@app.get("/ci/events")
def get_ci_events(limit: int = 25):
    safe_limit = max(1, min(limit, 200))
    return {"events": ci_events[:safe_limit], "count": len(ci_events)}


@app.post("/reset")
def reset(req: ResetRequest | None = None):
    return env.reset(task_id=req.task_id if req else None).model_dump()


@app.post("/step")
def step(action: Action):
    return env.step(action).model_dump()


@app.get("/state")
def state():
    return env.state.model_dump()


def _record_ci_event(event: CIEvent) -> None:
    ci_events.insert(0, event.model_dump())
    if len(ci_events) > 200:
        del ci_events[200:]


def _normalize_status(status: str | None) -> str:
    s = (status or "").strip().lower()
    if s in {"failed", "failure", "error", "broken"}:
        return "failed"
    if s in {"success", "passed", "ok"}:
        return "passed"
    if s in {"running", "pending", "created"}:
        return "running"
    return s or "unknown"


@app.post("/ci/webhook/gitlab")
def ci_webhook_gitlab(payload: GitlabWebhookPayload, x_gitlab_token: Optional[str] = Header(default=None)):
    if CI_WEBHOOK_TOKEN and x_gitlab_token != CI_WEBHOOK_TOKEN:
        return {"ok": False, "error": "unauthorized", "provider": "gitlab"}

    attrs = payload.object_attributes or {}
    project_name = (payload.project or {}).get("path_with_namespace") or (payload.project or {}).get("name") or "unknown-project"
    commit_sha = (payload.commit or {}).get("id") or attrs.get("sha") or "unknown"
    branch = payload.ref or attrs.get("ref") or "unknown"
    status = _normalize_status(payload.build_status or attrs.get("status"))
    job_name = payload.build_name or attrs.get("name") or attrs.get("stage") or payload.build_stage or "pipeline"
    pipeline_id = str(attrs.get("id") or attrs.get("pipeline_id") or "unknown")
    source_event = payload.object_kind or payload.event_type or "gitlab"
    log_url = attrs.get("url") or attrs.get("web_url")
    service = project_name.split("/")[-1] if project_name else None
    task_id = find_task_for_ci_event(service=service, job_name=job_name)

    event = CIEvent(
        provider="gitlab",
        source_event=str(source_event),
        status=status,
        project=str(project_name),
        branch=str(branch),
        commit_sha=str(commit_sha),
        pipeline_id=pipeline_id,
        job_name=str(job_name),
        task_id=task_id,
        service=service,
        log_url=log_url,
        timestamp=datetime.now(timezone.utc).isoformat(),
        raw=payload.model_dump(),
    )
    _record_ci_event(event)
    return {"ok": True, "provider": "gitlab", "event": event.model_dump()}


@app.post("/ci/webhook/jenkins")
def ci_webhook_jenkins(payload: JenkinsWebhookPayload, x_patch2prod_token: Optional[str] = Header(default=None)):
    if CI_WEBHOOK_TOKEN and x_patch2prod_token != CI_WEBHOOK_TOKEN:
        return {"ok": False, "error": "unauthorized", "provider": "jenkins"}

    build = payload.build or {}
    project_name = payload.name or payload.displayName or "unknown-project"
    commit_sha = build.get("scm", {}).get("commit") or build.get("commit") or "unknown"
    branch = build.get("scm", {}).get("branch") or build.get("branch") or "unknown"
    status = _normalize_status(build.get("status"))
    job_name = build.get("stage") or payload.displayName or payload.name or "pipeline"
    pipeline_id = str(build.get("number") or "unknown")
    source_event = "jenkins"
    log_url = payload.full_url or payload.url
    service = project_name
    task_id = find_task_for_ci_event(service=service, job_name=job_name)

    event = CIEvent(
        provider="jenkins",
        source_event=source_event,
        status=status,
        project=str(project_name),
        branch=str(branch),
        commit_sha=str(commit_sha),
        pipeline_id=pipeline_id,
        job_name=str(job_name),
        task_id=task_id,
        service=service,
        log_url=log_url,
        timestamp=datetime.now(timezone.utc).isoformat(),
        raw=payload.model_dump(),
    )
    _record_ci_event(event)
    return {"ok": True, "provider": "jenkins", "event": event.model_dump()}


def main():
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("patch2prod.server:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
