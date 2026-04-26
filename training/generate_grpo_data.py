#!/usr/bin/env python3
"""
Generate GRPO training states by running real Patch2ProdEnv episodes with the
deterministic oracle policy (`expected_next_action` from evaluate_sft_policy).

At each step the prompt is built with `build_prompt_from_obs` — the same rich
format the model sees at eval time — so GRPO rewards are no longer measuring
the model on prompts it has never encountered.

Usage:
  python training/generate_grpo_data.py \
      --tasks data/eval_tasks.jsonl \
      --out   data/grpo_train_states.jsonl \
      --max_steps 24
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

# evaluate_sft_policy lives in the same directory; add it to the path.
sys.path.insert(0, str(Path(__file__).parent))

from evaluate_sft_policy import (
    ActionObject,
    Patch2ProdEnv,
    build_prompt_from_obs,
    expected_next_action,
    load_jsonl,
    safe_reset,
    safe_step,
    to_jsonable,
)


def run_oracle_episode(
    task: Dict[str, Any],
    max_steps: int,
) -> List[Dict[str, Any]]:
    """
    Run one full episode using the oracle policy and collect (prompt, gold_action)
    pairs at every step, using the same prompt format as eval.
    """
    env = Patch2ProdEnv()
    obs = safe_reset(env, task["task_id"])

    records: List[Dict[str, Any]] = []

    for step_idx in range(1, max_steps + 1):
        obs_j = to_jsonable(obs)
        prompt = build_prompt_from_obs(task, obs_j, step_idx)
        gold = expected_next_action(task, obs_j)

        if gold is None:
            break

        records.append(
            {
                "prompt": prompt,
                "gold_action": json.dumps(gold, separators=(",", ":")),
                "task_id": task.get("task_id", ""),
                "variant": task.get("variant", ""),
            }
        )

        try:
            obs, _reward, done, _info = safe_step(env, gold)
        except Exception as exc:
            print(f"  [warn] env.step raised: {exc}; stopping episode.", file=sys.stderr)
            break

        if done:
            break

    return records


def augment_prompt(prompt: str, task_id: str, step_idx: int, variant_idx: int) -> str:
    """
    Keep the core prompt semantics unchanged while adding light surface-form
    variation so GRPO sees more prompt diversity.
    """
    prefixes = [
        "Return exactly one JSON action object for the current step.",
        "Output one valid action JSON object only.",
        "Respond with one env action in strict JSON format.",
        "Emit a single action object; no prose.",
    ]
    suffixes = [
        "Stop at the closing brace.",
        "No markdown and no explanation text.",
        "Use schema: {\"action_type\":\"...\",\"params\":{...}} only.",
        "Do not include code fences or extra tokens.",
    ]
    prefix = prefixes[variant_idx % len(prefixes)]
    suffix = suffixes[variant_idx % len(suffixes)]
    return (
        f"{prompt}\n\n"
        f"[grpo_variant={variant_idx}; task_id={task_id}; step={step_idx}]\n"
        f"{prefix}\n"
        f"{suffix}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GRPO training data from env episodes")
    parser.add_argument("--tasks", default="data/eval_tasks.jsonl")
    parser.add_argument("--out", default="data/grpo_train_states.jsonl")
    parser.add_argument("--max_steps", type=int, default=24)
    parser.add_argument(
        "--augment_copies",
        type=int,
        default=8,
        help="Number of prompt variants per oracle state (>=1).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    tasks = load_jsonl(args.tasks)
    all_records: List[Dict[str, Any]] = []

    for task in tasks:
        tid = task.get("task_id", "?")
        print(f"Generating episode for: {tid}")
        records = run_oracle_episode(task, max_steps=args.max_steps)
        expanded: List[Dict[str, Any]] = []
        for step_idx, row in enumerate(records, 1):
            for variant_idx in range(max(1, args.augment_copies)):
                row_copy = dict(row)
                row_copy["prompt"] = augment_prompt(
                    row["prompt"],
                    task_id=row.get("task_id") or tid,
                    step_idx=step_idx,
                    variant_idx=variant_idx,
                )
                row_copy["variant"] = f"aug_v{variant_idx}"
                expanded.append(row_copy)

        random.shuffle(expanded)
        print(
            f"  → {len(records)} base states, {len(expanded)} augmented "
            f"(augment_copies={max(1, args.augment_copies)})"
        )
        all_records.extend(expanded)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in all_records:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"\nWrote {len(all_records)} rows to {out_path}")


if __name__ == "__main__":
    main()
