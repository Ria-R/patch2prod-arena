from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Action(BaseModel):
    """Single agent action.

    Keep the action shape deliberately simple so an LLM can emit it as JSON.
    """

    action_type: str = Field(..., description="Tool/action name")
    params: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    episode_id: str
    task_id: str
    step: int
    done: bool
    message: str
    visible_state: Dict[str, Any] = Field(default_factory=dict)
    last_action_result: Dict[str, Any] = Field(default_factory=dict)
    reward: float = 0.0
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)
    available_actions: List[str] = Field(default_factory=list)


class EnvState(BaseModel):
    episode_id: str
    task_id: str
    step: int
    max_steps: int
    done: bool
    pipeline_status: str
    changed_files: List[str] = Field(default_factory=list)
    current_diff: str = ""
    discovered: Dict[str, Any] = Field(default_factory=dict)
    validations: Dict[str, str] = Field(default_factory=dict)
    release_decision: Optional[Dict[str, Any]] = None
    reward_total: float = 0.0
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)
    action_trace: List[Dict[str, Any]] = Field(default_factory=list)
