"""Tiny training/evaluation scaffold.

This is NOT a full TRL run. It is the bridge you show first: the reward function
is environment-backed, deterministic, and ready to be plugged into GRPO/TRL.

For the hackathon, you can later replace `candidate_policy_outputs` with model
samples and use the returned rewards inside a TRL reward function.
"""
from __future__ import annotations

from typing import Dict, List

from patch2prod.env import Patch2ProdEnv
from patch2prod.models import Action


def score_action_sequence(actions: List[Dict], task_id: str = "authsdk_mobile_contract_break") -> float:
    env = Patch2ProdEnv(default_task_id=task_id)
    env.reset(task_id=task_id)
    for raw in actions:
        obs = env.step(Action(**raw))
        if obs.done:
            break
    return env.state.reward_total


# Replace these with LLM-generated JSON action sequences during RLVR/GRPO.
candidate_policy_outputs = [
    [
        {"action_type": "view_log", "params": {"job_name": "unit-tests"}},
        {"action_type": "replace", "params": {"file_path": "app/retry.py", "search": "build_retry_policy", "replace": "create_retry_policy"}},
        {"action_type": "run_unit_tests", "params": {"service": "auth-service"}},
        {"action_type": "submit_release_decision", "params": {"decision": "ship", "reason": "CI passed"}},
    ],
    [
        {"action_type": "view_log", "params": {"job_name": "unit-tests"}},
        {"action_type": "view_commit_history", "params": {}},
        {"action_type": "view_diff", "params": {"commit_id": "c42"}},
        {"action_type": "submit_causal_change", "params": {"commit": "c42", "summary": "authsdk upgrade caused API rename"}},
        {"action_type": "view_migration_guide", "params": {"package": "authsdk"}},
        {"action_type": "replace", "params": {"file_path": "app/retry.py", "search": "build_retry_policy", "replace": "create_retry_policy"}},
        {"action_type": "run_unit_tests", "params": {"service": "auth-service"}},
        {"action_type": "view_dependency_graph", "params": {"service": "auth-service"}},
        {"action_type": "submit_blast_radius", "params": {"impacted_services": ["checkout-service", "mobile-gateway", "fraud-service"]}},
        {"action_type": "run_contract_tests", "params": {"service": "mobile-gateway"}},
        {"action_type": "submit_release_decision", "params": {"decision": "block", "owner_to_notify": "mobile-platform", "reason": "downstream mobile contract failed"}},
    ],
]


if __name__ == "__main__":
    for i, seq in enumerate(candidate_policy_outputs, 1):
        print(f"candidate_{i}: reward={score_action_sequence(seq):.2f}")
