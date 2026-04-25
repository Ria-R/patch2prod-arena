from __future__ import annotations

import os
from typing import Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from .env import Patch2ProdEnv
from .models import Action


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


DEFAULT_TASK = os.getenv("PATCH2PROD_DEFAULT_TASK", "authsdk_mobile_contract_break")
env = Patch2ProdEnv(default_task_id=DEFAULT_TASK)
app = FastAPI(title="Patch2Prod Arena", version="0.1.0")


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


@app.post("/reset")
def reset(req: ResetRequest | None = None):
    return env.reset(task_id=req.task_id if req else None).model_dump()


@app.post("/step")
def step(action: Action):
    return env.step(action).model_dump()


@app.get("/state")
def state():
    return env.state.model_dump()


def main():
    uvicorn.run("patch2prod.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
