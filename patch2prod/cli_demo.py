from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from .env import Patch2ProdEnv
from .models import Action


def baseline_policy() -> List[Dict]:
    """Naive policy: fix CI and ship immediately.

    This is intentionally short-sighted so the demo shows the gap: green CI is
    not enough when downstream contracts break.
    """
    return [
        {"action_type": "view_log", "params": {"job_name": "unit-tests"}},
        {"action_type": "view_migration_guide", "params": {"package": "authsdk"}},
        {"action_type": "replace", "params": {"file_path": "app/retry.py", "search": "build_retry_policy", "replace": "create_retry_policy"}},
        {"action_type": "run_unit_tests", "params": {"service": "auth-service"}},
        {"action_type": "submit_release_decision", "params": {"decision": "ship", "reason": "auth-service unit tests are green"}},
    ]


def improved_policy() -> List[Dict]:
    """Risk-aware policy: diagnose, repair, check graph, validate contracts, block unsafe release."""
    return [
        {"action_type": "view_log", "params": {"job_name": "unit-tests"}},
        {"action_type": "view_commit_history", "params": {}},
        {"action_type": "view_diff", "params": {"commit_id": "c42"}},
        {"action_type": "submit_causal_change", "params": {"commit": "c42", "summary": "authsdk upgraded from 1.8.0 to 2.0.0"}},
        {"action_type": "view_migration_guide", "params": {"package": "authsdk"}},
        {"action_type": "replace", "params": {"file_path": "app/retry.py", "search": "build_retry_policy", "replace": "create_retry_policy"}},
        {"action_type": "run_unit_tests", "params": {"service": "auth-service"}},
        {"action_type": "view_dependency_graph", "params": {"service": "auth-service"}},
        {"action_type": "mark_impacted_service", "params": {"service": "checkout-service", "reason": "Consumes auth tokens during checkout"}},
        {"action_type": "mark_impacted_service", "params": {"service": "mobile-gateway", "reason": "Parses token expiry format"}},
        {"action_type": "mark_impacted_service", "params": {"service": "fraud-service", "reason": "Consumes identity claims"}},
        {"action_type": "submit_blast_radius", "params": {"impacted_services": ["checkout-service", "mobile-gateway", "fraud-service"]}},
        {"action_type": "run_contract_tests", "params": {"service": "checkout-service"}},
        {"action_type": "run_contract_tests", "params": {"service": "mobile-gateway"}},
        {"action_type": "view_ownership_map", "params": {}},
        {"action_type": "submit_release_decision", "params": {"decision": "block", "owner_to_notify": "mobile-platform", "reason": "CI is green but downstream mobile-gateway contract fails after token_expiry format change"}},
    ]


def run_policy(name: str, actions: List[Dict], task_id: str = "authsdk_mobile_contract_break") -> Dict:
    env = Patch2ProdEnv(default_task_id=task_id)
    obs = env.reset(task_id=task_id)
    rewards = [0.0]
    print(f"\n=== {name.upper()} ===")
    print(f"reset -> {obs.last_action_result['pipeline_status']} | {obs.last_action_result['problem']}")
    for raw in actions:
        obs = env.step(Action(**raw))
        rewards.append(env.state.reward_total)
        result = obs.last_action_result
        print(f"step={obs.step:02d} action={raw['action_type']:<24} reward={obs.reward:>5} total={env.state.reward_total:>6.2f}")
        if raw["action_type"] in {"run_unit_tests", "run_contract_tests", "submit_release_decision"}:
            print("   result:", json.dumps(result, indent=2)[:500])
        if obs.done:
            break
    final_state = env.state.model_dump()
    final_state["reward_curve"] = rewards
    return final_state


def save_plot(results: Dict[str, Dict], out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for name, result in results.items():
        plt.plot(list(range(len(result["reward_curve"]))), result["reward_curve"], marker="o", label=name)
    plt.xlabel("Environment step")
    plt.ylabel("Cumulative reward")
    plt.title("Patch2Prod Arena: Baseline vs Risk-Aware Policy")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"\nSaved reward plot: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/demo_results.json", help="Output JSON path")
    parser.add_argument("--plot", default="artifacts/plots/reward_curve.png", help="Output plot path")
    args = parser.parse_args()

    baseline = run_policy("baseline_green_ci_only", baseline_policy())
    improved = run_policy("improved_patch2prod", improved_policy())
    results = {"baseline": baseline, "improved": improved}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    save_plot(results, Path(args.plot))

    print("\n=== SUMMARY ===")
    for name, state in results.items():
        print(f"{name}: reward={state['reward_total']:.2f}, decision={state.get('release_decision')}")


if __name__ == "__main__":
    main()
