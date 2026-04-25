
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
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


try:
    from patch2prod.env import Patch2ProdEnv
except Exception as exc:
    raise RuntimeError(
        "Could not import Patch2ProdEnv. Run `pip install -e .` from the repo root first."
    ) from exc

class ActionObject:
    def __init__(self, action_type: str, params: Dict[str, Any] | None = None):
        self.action_type = action_type
        self.params = params or {}

    def model_dump(self):
        return {
            "action_type": self.action_type,
            "params": self.params,
        }

    def dict(self):
        return self.model_dump()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(value) for value in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def safe_env_reset(env: Patch2ProdEnv, task_id: Optional[str]) -> Any:
    try:
        return env.reset(task_id=task_id)
    except TypeError:
        return env.reset()


def normalize_action(action: Dict[str, Any]) -> Any:
    """
    Convert script-style dict actions into the shape expected by Patch2ProdEnv.
    """
    if not isinstance(action, dict):
        return action

    action = dict(action)
    action_type = action.pop("action_type", None) or action.pop("action", None) or "invalid"

    return ActionObject(
        action_type=action_type,
        params=action,
    )


def safe_env_step(env: Patch2ProdEnv, action: Dict[str, Any]) -> Tuple[Any, float, bool, Dict[str, Any]]:
    """
    Supports classic Gym-style:
        obs, reward, done, info = env.step(action)

    Converts dict actions into attribute-style objects expected by Patch2ProdEnv.
    """
    env_action = normalize_action(action)
    result = env.step(env_action)

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
    service = task.get("service", "auth-service")
    failed_job = task.get("failed_job", "unit-tests")

    patch = task.get("expected_replacement") or task.get("expected_patch") or {}
    file_path = patch.get("file_path", patch.get("file", "app/retry.py"))

    return [
        {"action": "view_log", "job_name": failed_job},
        {
            "action": "replace",
            "file_path": file_path,
            "search": patch.get("search", "build_retry_policy"),
            "replace": patch.get("replace", "create_retry_policy"),
        },
        {"action": "run_unit_tests", "service": service},
        {
            "action": "submit_release_decision",
            "decision": "ship",
            "reason": "Local CI is passing after the patch.",
            "owner_to_notify": task.get("correct_owner", "release-engineering"),
        },
    ]


def improved_actions(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    service = task.get("service", "auth-service")
    owner = task.get("correct_owner", "release-engineering")
    decision = (
        task.get("correct_release_decision")
        or task.get("correct_decision_after_optional_fix")
        or task.get("correct_decision_before_optional_fix")
        or "block"
    )
    failed_job = task.get("failed_job", "unit-tests")

    patch = task.get("expected_replacement") or task.get("expected_patch") or {}
    file_path = patch.get("file_path", patch.get("file", "app/retry.py"))
    causal_commit = task.get("causal_commit", "c42")
    causal_summary = task.get("description") or task.get("initial_failure") or f"Regression introduced in {causal_commit}"

    actions: List[Dict[str, Any]] = [
        {"action": "view_log", "job_name": failed_job},
        {"action": "view_commit_history"},
        {"action": "view_diff", "commit_id": causal_commit},
        {"action": "submit_causal_change", "commit": causal_commit, "summary": causal_summary},
        {"action": "cat", "file_path": file_path},
        {"action": "view_migration_guide"},
        {
            "action": "replace",
            "file_path": file_path,
            "search": patch.get("search", "build_retry_policy"),
            "replace": patch.get("replace", "create_retry_policy"),
        },
        {"action": "run_unit_tests", "service": service},
        {"action": "view_dependency_graph", "service": service},
    ]

    impacted = task.get("impacted_services") or task.get("hidden_contract_breaks") or []

    for downstream in impacted:
        actions.append({
            "action": "mark_impacted_service",
            "service": downstream,
            "reason": "Downstream consumer of changed service/API contract.",
        })

    actions.append({
        "action": "submit_blast_radius",
        "impacted_services": impacted,
    })

    for downstream in impacted:
        actions.append({"action": "run_contract_tests", "service": downstream})

    if task.get("category") in {"dependency_upgrade", "security_patch"}:
        actions.append({"action": "view_security_advisory", "package": "authsdk==2.0.0"})

    actions.append(
        {
            "action": "submit_release_decision",
            "decision": decision,
            "reason": (
                "Release decision is based on CI repair, dependency graph, "
                "blast radius, targeted contract tests, and release policy."
            ),
            "owner_to_notify": owner,
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
    initial_obs_json = to_jsonable(initial_obs)

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
        final_state = env.state
    except Exception:
        final_state = None

    return {
        "task_id": task.get("task_id"),
        "category": task.get("category"),
        "initial_observation": initial_obs_json,
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
        final_state = r.get("final_state") or {}
        if r.get("done") or final_state.get("done"):
            completed += 1

        for step in r.get("steps", []):
            info = step.get("info") or {}
            observation = step.get("observation") or {}
            visible_state = observation.get("visible_state") or {}
            validations = visible_state.get("validations") or {}

            if (
                info.get("ci_passed")
                or info.get("unit_tests_passed")
                or info.get("pipeline_status") == "passed"
                or visible_state.get("pipeline_status") == "passed"
                or any(status == "passed" for key, status in validations.items() if key.startswith("unit:"))
            ):
                ci_success += 1
                break

        final_action = None
        if r.get("steps"):
            final_action = r["steps"][-1].get("action", {})

        expected_decision = (
            final_state.get("release_decision", {}).get("correct_decision")
            or final_state.get("correct_release_decision")
        )
        if not expected_decision:
            task_id = r.get("task_id")
            if task_id == "authsdk_mobile_contract_break":
                expected_decision = "block"
            elif task_id == "payment_schema_checkout_break":
                expected_decision = "ship"

        if final_action and final_action.get("action") == "submit_release_decision":
            decision = final_action.get("decision")
            final_info = r["steps"][-1].get("info") or {}
            if final_info.get("correct_release_decision") is True:
                correct_release += 1
            elif final_info.get("unsafe_ship") is True:
                unsafe_ship += 1
            elif expected_decision and decision == expected_decision:
                correct_release += 1

            if decision == "ship":
                if final_info.get("unsafe_ship", False):
                    unsafe_ship += 1
                elif expected_decision and expected_decision != "ship":
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
    parser.add_argument("--max_steps", type=int, default=18)
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
