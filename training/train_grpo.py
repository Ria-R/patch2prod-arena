#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


AVAILABLE_ACTIONS = {
    "view_log",
    "view_commit_history",
    "view_diff",
    "cat",
    "view_migration_guide",
    "view_security_advisory",
    "replace",
    "run_unit_tests",
    "view_dependency_graph",
    "mark_impacted_service",
    "run_contract_tests",
    "view_ownership_map",
    "submit_causal_change",
    "submit_blast_radius",
    "submit_release_decision",
    "view_reward",
}

REQUIRED_PARAMS = {
    "view_log": ["job_name"],
    "view_diff": ["commit_id"],
    "cat": ["file_path"],
    "replace": ["file_path", "search", "replace"],
    "run_unit_tests": ["service"],
    "view_dependency_graph": ["service"],
    "mark_impacted_service": ["service", "reason"],
    "run_contract_tests": ["service"],
    "submit_blast_radius": ["impacted_services"],
    "submit_release_decision": ["decision", "reason", "owner"],
    "view_commit_history": [],
    "view_migration_guide": [],
    "view_security_advisory": [],
    "view_ownership_map": [],
    "submit_causal_change": [],
    "view_reward": [],
}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def trim_to_first_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return text.strip()

    depth = 0
    in_str = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if ch == '"':
            in_str = not in_str
            continue

        if not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1].strip()

    return text[start:].strip()


def parse_action(text: str) -> Tuple[bool, Dict[str, Any] | None, str]:
    try:
        trimmed = trim_to_first_json_object(text)
        obj = json.loads(trimmed)
    except Exception as e:
        return False, None, f"invalid_json:{e}"

    action_type = obj.get("action_type")
    params = obj.get("params")

    if action_type not in AVAILABLE_ACTIONS:
        return False, obj, f"invalid_action_type:{action_type}"

    if not isinstance(params, dict):
        return False, obj, "invalid_params"

    missing = [p for p in REQUIRED_PARAMS.get(action_type, []) if p not in params]
    if missing:
        return False, obj, f"missing_params:{missing}"

    # Penalize placeholders.
    text_dump = json.dumps(obj)
    if "<" in text_dump or ">" in text_dump:
        return False, obj, "placeholder_detected"

    return True, obj, "ok"


def action_similarity_reward(pred: Dict[str, Any], gold: Dict[str, Any]) -> float:
    """
    Dense reward for matching the gold next action.
    """
    reward = 0.0

    pred_type = pred.get("action_type")
    gold_type = gold.get("action_type")

    pred_params = pred.get("params", {}) or {}
    gold_params = gold.get("params", {}) or {}

    if pred_type == gold_type:
        reward += 2.0
    else:
        reward -= 1.0

    # Reward required param match.
    for key, gold_val in gold_params.items():
        if key in pred_params:
            reward += 0.25
            if pred_params[key] == gold_val:
                reward += 0.75
            else:
                reward -= 0.25
        else:
            reward -= 0.25

    # Penalize extra fake params lightly.
    for key in pred_params:
        if key not in gold_params:
            reward -= 0.1

    return reward


def process_reward(pred: Dict[str, Any], prompt: str) -> float:
    """
    Extra process reward / penalties based on prompt state.
    """
    reward = 0.0
    action_type = pred.get("action_type")
    params = pred.get("params", {}) or {}
    prompt_lower = prompt.lower()

    # Do not run contract tests before CI/local tests pass.
    if "pipeline_status\": \"failed" in prompt_lower and action_type == "run_contract_tests":
        reward -= 1.5

    if "viewed_logs\": []" in prompt_lower and action_type != "view_log":
        reward -= 1.0

    if "viewed_commits\": false" in prompt_lower and action_type == "view_diff":
        reward -= 1.0

    # Known task-specific constraints.
    if "authsdk_mobile_contract_break" in prompt:
        if action_type == "view_log" and params.get("job_name") == "unit-tests":
            reward += 0.5
        if action_type == "view_diff" and params.get("commit_id") == "c42":
            reward += 0.5
        if params.get("commit_id") == "c43":
            reward -= 1.0
        if params.get("service") in {"payment-response", "payment-tester", "<your-patch-id>", "<your-service-name>"}:
            reward -= 1.0

    if "payment_schema_checkout_break" in prompt:
        if action_type == "view_log" and params.get("job_name") == "integration-tests":
            reward += 0.5
        if action_type == "view_diff" and params.get("commit_id") == "p17":
            reward += 0.5
        if params.get("service") in {"payment-response", "payment-tester", "<your-patch-id>", "<your-service-name>"}:
            reward -= 1.0

    return reward


def make_reward_func():
    def reward_func(completions, gold_action=None, prompt=None, prompts=None, **kwargs):
        rewards = []

        # TRL versions differ: sometimes prompt is passed, sometimes prompts.
        if prompts is None:
            prompts = prompt

        for i, completion in enumerate(completions):
            # Completion may be plain string or chat message format depending TRL version.
            if isinstance(completion, list):
                text = completion[0].get("content", "")
            else:
                text = completion

            gold_raw = gold_action[i] if isinstance(gold_action, list) else gold_action
            prompt_i = prompts[i] if isinstance(prompts, list) else ""

            try:
                gold = json.loads(gold_raw)
            except Exception:
                rewards.append(-2.0)
                continue

            ok, pred, reason = parse_action(text)

            if not ok or pred is None:
                rewards.append(-2.0)
                continue

            reward = 1.0  # valid JSON/action base reward
            reward += action_similarity_reward(pred, gold)
            reward += process_reward(pred, prompt_i)

            # Slight length/prose penalty.
            trimmed = trim_to_first_json_object(text)
            if len(text.strip()) > len(trimmed) + 5:
                reward -= 0.5

            rewards.append(float(reward))

        return rewards

    return reward_func


def load_dataset(path: str) -> Dataset:
    rows = load_jsonl(path)
    return Dataset.from_list([
        {
            "prompt": row["prompt"],
            "gold_action": row["gold_action"],
            "task_id": row.get("task_id", ""),
            "variant": row.get("variant", ""),
        }
        for row in rows
    ])


def load_model_and_tokenizer(model_path: str, base_model: str | None):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_config = Path(model_path) / "adapter_config.json"

    if adapter_config.exists():
        if PeftModel is None:
            raise RuntimeError("peft is required to load LoRA adapters.")

        if not base_model:
            with adapter_config.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            base_model = cfg.get("base_model_name_or_path")

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path, is_trainable=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    model.config.use_cache = False
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/sft_patch2prod_lora")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train", default="data/grpo_train_states.jsonl")
    parser.add_argument("--out", default="outputs/grpo_patch2prod_lora")
    parser.add_argument("--epochs", type=float, default=1.0)
    args = parser.parse_args()

    dataset = load_dataset(args.train)
    model, tokenizer = load_model_and_tokenizer(args.model, args.base_model)

    config = GRPOConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],

        # GRPO generation
        num_generations=2,
        max_prompt_length=1536,
        max_completion_length=128,

        # Keep small for Colab
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=make_reward_func(),
    )

    result = trainer.train()

    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)

    print(result)


if __name__ == "__main__":
    main()