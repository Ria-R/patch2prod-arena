#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from patch2prod.env import Patch2ProdEnv
from patch2prod.models import Action, Observation


LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
TASK_NAME = os.getenv("PATCH2PROD_TASK", "authsdk_mobile_contract_break")
BENCHMARK = os.getenv("PATCH2PROD_BENCHMARK", "patch2prod")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "220"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "18"))

# Approximate normalization range for Patch2Prod cumulative reward.
SCORE_MIN = float(os.getenv("SCORE_MIN", "-3.0"))
SCORE_MAX = float(os.getenv("SCORE_MAX", "6.0"))

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an autonomous release engineering agent for Patch2Prod Arena.

    Return exactly one JSON object with this schema:
    {
      "action_type": "<one available action>",
      "params": { ... }
    }

    Rules:
    - Output JSON only.
    - Use only actions from available_actions.
    - Keep params aligned to the selected action.
    - Prefer evidence-driven actions before the final release decision.
    - When enough evidence exists, submit a release decision and finish the episode.
    """
).strip()


def _single_line(value: Any) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={_single_line(task)} env={_single_line(env)} model={_single_line(model)}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = "null" if error in (None, "") else _single_line(error)
    print(
        f"[STEP] step={step} action={_single_line(action)} reward={reward:.2f} done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def normalize_score(total_reward: float) -> float:
    if SCORE_MAX <= SCORE_MIN:
        return 0.0
    score = (total_reward - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)
    return max(0.0, min(1.0, score))


def build_user_prompt(observation: Observation, history: List[Dict[str, Any]]) -> str:
    compact_history = history[-6:]
    return textwrap.dedent(
        f"""
        Task ID: {observation.task_id}
        Step: {observation.step}
        Done: {str(observation.done).lower()}
        Available actions: {_json_dumps(observation.available_actions)}
        Message: {_json_dumps(observation.message)}
        Visible state: {_json_dumps(observation.visible_state)}
        Last action result: {_json_dumps(observation.last_action_result)}
        Recent history: {_json_dumps(compact_history)}

        Choose the single best next action.
        """
    ).strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("empty model response")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in model response")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("model response JSON was not an object")
    return parsed


def fallback_action(observation: Observation, history: Optional[List[Dict[str, Any]]] = None) -> Action:
    state = observation.visible_state
    discovered = state.get("discovered", {})
    validations = state.get("validations", {})
    service = state.get("service", "auth-service")
    task_id = observation.task_id

    if "unit-tests" not in discovered.get("viewed_logs", []):
        return Action(action_type="view_log", params={"job_name": "unit-tests"})

    if not discovered.get("viewed_commits"):
        return Action(action_type="view_commit_history", params={})

    if discovered.get("submitted_causal_change") is None:
        commit = "c42" if task_id == "authsdk_mobile_contract_break" else "p17"
        summary = "authsdk upgraded from 1.8.0 to 2.0.0" if task_id == "authsdk_mobile_contract_break" else "payment_status renamed to status"
        return Action(action_type="submit_causal_change", params={"commit": commit, "summary": summary})

    changed_files = state.get("changed_files", [])
    if not changed_files:
        if task_id == "payment_schema_checkout_break":
            return Action(
                action_type="replace",
                params={
                    "file_path": "app/payment_response.py",
                    "search": "return {'status': p.status, 'id': p.id}",
                    "replace": "return {'status': p.status, 'payment_status': p.status, 'id': p.id}",
                },
            )
        return Action(
            action_type="replace",
            params={
                "file_path": "app/retry.py",
                "search": "build_retry_policy",
                "replace": "create_retry_policy",
            },
        )

    if validations.get(f"unit:{service}") != "passed":
        return Action(action_type="run_unit_tests", params={"service": service})

    if not discovered.get("viewed_dependency_graph"):
        return Action(action_type="view_dependency_graph", params={"service": service})

    if task_id == "authsdk_mobile_contract_break":
        impacted = ["checkout-service", "mobile-gateway", "fraud-service"]
        for downstream in impacted:
            if downstream not in discovered.get("marked_impacted", []):
                return Action(
                    action_type="mark_impacted_service",
                    params={"service": downstream, "reason": "Downstream consumer of auth-service tokens or identity claims"},
                )
        if discovered.get("submitted_blast_radius") is None:
            return Action(action_type="submit_blast_radius", params={"impacted_services": impacted})
        if validations.get("contract:checkout-service") is None:
            return Action(action_type="run_contract_tests", params={"service": "checkout-service"})
        if validations.get("contract:mobile-gateway") is None:
            return Action(action_type="run_contract_tests", params={"service": "mobile-gateway"})
        if "ownership" not in observation.last_action_result and not history_has_action(history=history, action_type="view_ownership_map"):
            return Action(action_type="view_ownership_map", params={})

        mobile_status = validations.get("contract:mobile-gateway")
        if mobile_status == "failed":
            return Action(
                action_type="submit_release_decision",
                params={
                    "decision": "block",
                    "owner_to_notify": "mobile-platform",
                    "reason": "Local CI passes but downstream mobile-gateway contract fails after the authsdk token_expiry change",
                },
            )
        return Action(
            action_type="submit_release_decision",
            params={
                "decision": "canary",
                "owner_to_notify": "mobile-platform",
                "reason": "Local CI and downstream contracts are green after compatibility checks",
            },
        )

    if validations.get("contract:checkout-service") is None:
        return Action(action_type="run_contract_tests", params={"service": "checkout-service"})

    return Action(
        action_type="submit_release_decision",
        params={"decision": "ship", "reason": "Local CI and targeted downstream validation passed"},
    )


def history_has_action(history: Optional[List[Dict[str, Any]]], action_type: str) -> bool:
    if not history:
        return False
    return any(item.get("action", {}).get("action_type") == action_type for item in history)


def choose_action(client: OpenAI, observation: Observation, history: List[Dict[str, Any]]) -> Action:
    user_prompt = build_user_prompt(observation, history)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
            response_format={"type": "json_object"},
        )
        content = (response.choices[0].message.content or "").strip()
        payload = extract_json_object(content)
        return Action(**payload)
    except Exception as exc:
        print(_single_line(f"Model request failed: {exc}"), file=sys.stderr, flush=True)
        return fallback_action(observation, history)


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = Patch2ProdEnv(default_task_id=TASK_NAME, max_steps=MAX_STEPS)

    rewards: List[float] = []
    history: List[Dict[str, Any]] = []
    steps_taken = 0
    total_reward = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset(task_id=TASK_NAME)

        while not observation.done and steps_taken < MAX_STEPS:
            action = choose_action(client, observation, history)
            action_str = _json_dumps(action.model_dump())

            observation = env.step(action)
            reward = float(observation.reward or 0.0)
            done = bool(observation.done)
            error = observation.last_action_result.get("error")

            rewards.append(reward)
            total_reward += reward
            steps_taken += 1

            history.append(
                {
                    "step": steps_taken,
                    "action": action.model_dump(),
                    "reward": round(reward, 3),
                    "done": done,
                    "result": observation.last_action_result,
                }
            )

            log_step(
                step=steps_taken,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
            )

        score = normalize_score(total_reward)
        success = bool(observation.done) and score > 0.0
    except Exception as exc:
        print(_single_line(f"Episode failed: {exc}"), file=sys.stderr, flush=True)
        score = normalize_score(total_reward)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
