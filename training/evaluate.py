#!/usr/bin/env python3
"""
Evaluate Patch2Prod Arena policies.

Examples:
    python training/evaluate.py --policy baseline --tasks data/eval_tasks.jsonl --out artifacts/traces/baseline_trace.json

    python training/evaluate.py --policy improved --tasks data/eval_tasks.jsonl --out artifacts/traces/improved_trace.json

Later, after training:
    python training/evaluate.py --policy model --model outputs/grpo_patch2prod --tasks data/eval_tasks.jsonl --out artifacts/traces/grpo_trace.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


try:
    from patch2prod_arena.env import Patch2ProdEnv
except Exception as exc:
    raise RuntimeError(
        "Could not import Patch2ProdEnv. Run `pip install -e .` from the repo root first."
    ) from exc


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def to_jsonable(obj: Any) -> Any:
    """Best-effort conversion for observations/info returned by the env."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def safe_env_reset(env: Patch2ProdEnv, task_id: Optional[str]) -> Any:
    """
    Supports both env.reset(task_id=...) and env.reset().
    """
    try:
        return env.reset(task_id=task_id)
    except TypeError:
        return env.reset()


def safe_env_step(env: Patch2ProdEnv, action: Dict[str, Any]) -> Tuple[Any, float, bool, Dict[str, Any]]:
    """
    Supports classic Gym-style:
        obs, reward, done, info = env.step(action)

    If your env returns a dict/object, adapt here.
    """
    result = env.step(action)

    if isinstance(result, tuple) and len(result) == 4:
        obs, reward, done, info = result
        return obs, float(reward), bool(done), dict(info or {})

    if isinstance(result, dict):
        obs = result.get("observation") or result.get("obs") or result
        reward = float(result.get("reward", 0.0))
        done = bool(result.get("done", False))
        info = result.get("info", {})
        return obs, reward, done, dict(info or {})

    if hasattr(result, "reward"):
        obs = getattr(result, "observation", result)
        reward = float(getattr(result, "reward", 0.0))
        done = bool(getattr(result, "done", False))
        info = getattr(result, "info", {})
        return obs, reward, done, dict(info or {})

    raise RuntimeError(f"Unsupported env.step() return type: {type(result)}")


def baseline_actions(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Weak baseline:
    - Reads log
    - Applies obvious local patch if known
    - Runs unit tests
    - Ships immediately without blast-radius validation

    This should score low on unsafe release scenarios.
    """
    service = task.get("service", "auth-service")

    actions: List[Dict[str, Any]] = [
        {"action": "view_log", "job_name": "unit-tests"},
    ]

    # Generic patch if expected_patch exists in the task.
    patch = task.get("expected_patch") or {}
    if patch:
        actions.append(
            {
                "action": "replace",
                "file_path": patch.get("file", "services/auth-service/app/retry.py"),
                "search": patch.get("search", "build_retry_policy"),
                "replace": patch.get("replace", "create_retry_policy"),
            }
        )
    else:
        # Fallback for the default authsdk demo.
        actions.append(
            {
                "action": "replace",
                "file_path": "services/auth-service/app/retry.py",
                "search": "build_retry_policy",
                "replace": "create_retry_policy",
            }
        )

    actions.extend(
        [
            {"action": "run_unit_tests", "service": service},
            {
                "action": "submit_release_decision",
                "decision": "ship",
                "reason": "Local CI is passing after the patch.",
                "owner": task.get("correct_owner", "release-engineering"),
            },
        ]
    )

    return actions


def improved_actions(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Risk-aware reference policy:
    - Diagnoses
    - Patches local issue
    - Runs local validation
    - Checks dependency graph
    - Runs contract tests for expected impacted services
    - Blocks/canaries/ships according to task's correct decision
    """
    service = task.get("service", "auth-service")
    owner = task.get("correct_owner", "release-engineering")
    decision = task.get("correct_release_decision", "block")

    actions: List[Dict[str, Any]] = [
        {"action": "view_log", "job_name": "unit-tests"},
        {"action": "view_diff", "commit_id": task.get("causal_commit", "c42")},
        {"action": "grep", "pattern": task.get("grep_pattern", "build_retry_policy"), "path": "."},
    ]

    patch = task.get("expected_patch") or {}
    actions.append(
        {
            "action": "replace",
            "file_path": patch.get("file", "services/auth-service/app/retry.py"),
            "search": patch.get("search", "build_retry_policy"),
            "replace": patch.get("replace", "create_retry_policy"),
        }
    )

    actions.extend(
        [
            {"action": "run_unit_tests", "service": service},
            {"action": "view_dependency_graph", "service": service},
        ]
    )

    impacted = task.get("impacted_services") or task.get("hidden_contract_breaks") or []
    # Ensure the important demo service is included when task data is sparse.
    if not impacted and service == "auth-service":
        impacted = ["checkout-service", "mobile-gateway"]

    for downstream in impacted:
        actions.append({"action": "run_contract_tests", "service": downstream})

    # Optional security validation for dependency/security tasks.
    if task.get("category") in {"security_patch", "dependency_upgrade"}:
        actions.append({"action": "run_security_scan", "service": service})

    actions.append(
        {
            "action": "submit_release_decision",
            "decision": decision,
            "reason": (
                "Release decision is based on local CI, dependency graph, "
                "targeted contract tests, and release policy."
            ),
            "owner": owner,
        }
    )

    return actions


def build_prompt(task: Dict[str, Any]) -> str:
    return f"""You are a release engineering agent in Patch2Prod Arena.

Goal:
Fix the failing CI, identify the causal change, reason about downstream blast radius,
run targeted validation, and make the correct release decision.

Task ID: {task["task_id"]}
Service: {task.get("service", "unknown")}
Failure: {task.get("initial_failure", "unknown")}

Available actions:
view_log, view_diff, grep, cat, replace, run_unit_tests,
view_dependency_graph, run_contract_tests, run_security_scan,
submit_release_decision.

Return ONLY valid JSON:
{{"actions":[{{"action":"..."}}]}}
"""


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def model_actions(task: Dict[str, Any], model_dir: str, max_new_tokens: int = 800) -> List[Dict[str, Any]]:
    """
    Optional model evaluation after SFT/GRPO.

    This loads a local checkpoint and asks it to generate JSON actions.
    Keep this simple; the environment reward decides quality.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Model evaluation requires torch and transformers. Install training/requirements-train.txt."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = build_prompt(task)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    parsed = extract_json(generated)

    if not parsed or "actions" not in parsed:
        return [{"action": "invalid_model_output", "raw_output": generated}]

    return parsed["actions"]


def run_episode(task: Dict[str, Any], actions: List[Dict[str, Any]], max_steps: int = 12) -> Dict[str, Any]:
    env = Patch2ProdEnv()
    initial_obs = safe_env_reset(env, task.get("task_id"))

    total_reward = 0.0
    done = False
    trace_steps = []

    for idx, action in enumerate(actions[:max_steps], start=1):
        obs, reward, done, info = safe_env_step(env, action)
        total_reward += reward

        trace_steps.append(
            {
                "step": idx,
                "action": action,
                "reward": reward,
                "total_reward": total_reward,
                "done": done,
                "observation": to_jsonable(obs),
                "info": to_jsonable(info),
            }
        )

        if done:
            break

    final_state = None
    try:
        final_state = env.state()
    except Exception:
        final_state = None

    return {
        "task_id": task.get("task_id"),
        "category": task.get("category"),
        "initial_observation": to_jsonable(initial_obs),
        "steps": trace_steps,
        "total_reward": round(total_reward, 4),
        "done": done,
        "final_state": to_jsonable(final_state),
    }


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}

    avg_reward = sum(r["total_reward"] for r in results) / len(results)

    # These are best-effort metrics. If your env info includes richer flags,
    # this function will pick them up.
    ci_success = 0
    correct_release = 0
    unsafe_ship = 0
    completed = 0

    for r in results:
        if r.get("done"):
            completed += 1

        all_infos = []
        for step in r.get("steps", []):
            info = step.get("info") or {}
            all_infos.append(info)

            if info.get("ci_passed") or info.get("unit_tests_passed") or info.get("pipeline_status") == "passed":
                ci_success += 1
                break

        final_action = None
        if r.get("steps"):
            final_action = r["steps"][-1].get("action", {})

        if final_action and final_action.get("action") == "submit_release_decision":
            decision = final_action.get("decision")
            # Prefer env info if available.
            final_info = r["steps"][-1].get("info") or {}
            if final_info.get("correct_release_decision") is True:
                correct_release += 1
            elif final_info.get("unsafe_ship") is True:
                unsafe_ship += 1
            elif decision in {"block", "canary", "rollback", "ship_after_patch", "ship"}:
                # Fallback: count non-ship decisions as safer than baseline for sparse envs.
                if decision != "ship":
                    correct_release += 1

            if decision == "ship":
                # If env says unsafe, count it. Otherwise fallback is conservative.
                final_info = r["steps"][-1].get("info") or {}
                if final_info.get("unsafe_ship", False):
                    unsafe_ship += 1

    n = len(results)
    return {
        "num_tasks": n,
        "avg_reward": round(avg_reward, 4),
        "completion_rate": round(completed / n, 4),
        "ci_repair_success_rate": round(ci_success / n, 4),
        "correct_release_decision_rate": round(correct_release / n, 4),
        "unsafe_ship_rate": round(unsafe_ship / n, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["baseline", "improved", "model"], default="baseline")
    parser.add_argument("--tasks", default="data/eval_tasks.jsonl")
    parser.add_argument("--out", default="artifacts/traces/eval_trace.json")
    parser.add_argument("--model", default=None, help="Local model/checkpoint path for --policy model")
    parser.add_argument("--max_steps", type=int, default=12)
    args = parser.parse_args()

    tasks = load_jsonl(args.tasks)
    results = []

    for task in tasks:
        if args.policy == "baseline":
            actions = baseline_actions(task)
        elif args.policy == "improved":
            actions = improved_actions(task)
        else:
            if not args.model:
                raise ValueError("--model is required when --policy model")
            actions = model_actions(task, args.model)

        result = run_episode(task, actions, max_steps=args.max_steps)
        results.append(result)

    metrics = compute_metrics(results)

    output = {
        "policy": args.policy,
        "model": args.model,
        "tasks_file": args.tasks,
        "metrics": metrics,
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps({"policy": args.policy, "metrics": metrics, "out": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()