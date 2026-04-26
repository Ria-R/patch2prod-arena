#!/usr/bin/env python3

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download
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
    "grep",
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
    "grep": ["pattern", "path"],
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

    # Known task-specific constraints (match canonical + legacy prompt IDs).
    auth_ctx = "authsdk_mobile_contract_break" in prompt or "authsdk_mobile_break_001" in prompt
    pay_ctx = "payment_schema_checkout_break" in prompt or "payment_schema_break_001" in prompt

    if auth_ctx:
        if action_type == "view_log" and params.get("job_name") == "unit-tests":
            reward += 0.5
        if action_type == "view_diff" and params.get("commit_id") == "c42":
            reward += 0.5
        if params.get("commit_id") == "c43":
            reward -= 1.0
        if params.get("service") in {"payment-response", "payment-tester", "<your-patch-id>", "<your-service-name>"}:
            reward -= 1.0

    if pay_ctx:
        if action_type == "view_log" and params.get("job_name") == "integration-tests":
            reward += 0.5
        if action_type == "view_diff" and params.get("commit_id") == "p17":
            reward += 0.5
        if params.get("service") in {"payment-response", "payment-tester", "<your-patch-id>", "<your-service-name>"}:
            reward -= 1.0

    return reward


_debug_logged: set = set()  # track which batch steps we already logged


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
                # Log one failing sample every 50 calls so we can see what
                # the model actually generates without flooding the output.
                key = reason.split(":")[0]
                if key not in _debug_logged:
                    _debug_logged.add(key)
                    print(
                        f"\n[GRPO DEBUG] reason={reason!r}\n"
                        f"  gold={gold_raw!r}\n"
                        f"  text={text[:300]!r}\n",
                        file=sys.stderr,
                        flush=True,
                    )
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


_GRPO_SCHEMA_HINT = (
    "\n\nOutput exactly one JSON object on one line: "
    '{"action_type":"<string>","params":{...}} '
    "No markdown, no code fences, no text after the closing brace."
)


def load_dataset(path: str) -> Dataset:
    rows = load_jsonl(path)
    records: List[Dict[str, Any]] = []
    for row in rows:
        prompt = row["prompt"]
        tid = (row.get("task_id") or "").strip()
        if tid and tid not in prompt:
            prompt = f"{prompt}\n[environment_task_id: {tid}]"
        prompt = f"{prompt}{_GRPO_SCHEMA_HINT}"
        records.append(
            {
                "prompt": prompt,
                "gold_action": row["gold_action"],
                "task_id": row.get("task_id", ""),
                "variant": row.get("variant", ""),
            }
        )
    return Dataset.from_list(records)


def apply_chat_template_to_prompts(dataset: Dataset, tokenizer: Any) -> Dataset:
    """
    Instruct checkpoints (e.g. Qwen2.5-Instruct) expect the chat template; raw task strings
    often produce long non-JSON continuations and max-length clips → constant -2 rewards.
    """
    if not getattr(tokenizer, "chat_template", None):
        return dataset

    def _one(row: Dict[str, Any]) -> Dict[str, Any]:
        messages = [{"role": "user", "content": row["prompt"]}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": text}

    return dataset.map(_one)


def load_adapter_config(model_path: str) -> Dict[str, Any] | None:
    local_cfg = Path(model_path) / "adapter_config.json"
    if local_cfg.exists():
        with local_cfg.open("r", encoding="utf-8") as f:
            return json.load(f)

    # model_path can be a Hub repo id (e.g. madhuria/patch2prod-sft-agent)
    try:
        cfg_path = hf_hub_download(repo_id=model_path, filename="adapter_config.json")
    except Exception:
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_and_tokenizer(model_path: str, base_model: str | None):
    adapter_cfg = load_adapter_config(model_path)
    inferred_base = adapter_cfg.get("base_model_name_or_path") if adapter_cfg else None
    if not base_model and inferred_base:
        base_model = inferred_base

    tokenizer_source = base_model if (adapter_cfg and base_model) else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_cfg is not None:
        if PeftModel is None:
            raise RuntimeError("peft is required to load LoRA adapters.")

        if not base_model:
            raise ValueError("--base_model is required for LoRA adapter training.")

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
    parser.add_argument("--model", default="madhuria/patch2prod-sft-agent")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train", default="data/grpo_train_states.jsonl")
    parser.add_argument("--out", default="outputs/grpo_patch2prod_lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=192,
        help="Token budget for one JSON action; too small causes clipped JSON.",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=4,
        help="GRPO needs per-prompt variance; use >=2 and prefer 4+ if batch size allows.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help=">0 yields diverse completions so rewards are not identical every time.",
    )
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Disable apply_chat_template wrapping (not recommended for Instruct models).",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.train)
    model, tokenizer = load_model_and_tokenizer(args.model, args.base_model)
    if not args.no_chat_template:
        dataset = apply_chat_template_to_prompts(dataset, tokenizer)

    grpo_sig = inspect.signature(GRPOConfig.__init__).parameters
    # TRL: `max_prompt_length` existed on older GRPOConfig; recent releases dropped it
    # (prompt length follows model max / tokenizer; use dataset truncation if needed).
    grpo_kwargs: Dict[str, Any] = {
        "output_dir": args.out,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": 1,
        "save_strategy": "epoch",
        "report_to": [],
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "remove_unused_columns": False,
    }
    if "max_prompt_length" in grpo_sig:
        grpo_kwargs["max_prompt_length"] = 1536
    if "temperature" in grpo_sig:
        grpo_kwargs["temperature"] = args.temperature

    gen_kw: Dict[str, Any] = {}
    if tokenizer.eos_token_id is not None:
        gen_kw["eos_token_id"] = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        gen_kw["pad_token_id"] = tokenizer.pad_token_id
    if gen_kw and "generation_kwargs" in grpo_sig:
        grpo_kwargs["generation_kwargs"] = gen_kw

    config = GRPOConfig(**grpo_kwargs)

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