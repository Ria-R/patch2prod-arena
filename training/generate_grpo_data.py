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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GRPO training data from env episodes")
    parser.add_argument("--tasks", default="data/eval_tasks.jsonl")
    parser.add_argument("--out", default="data/grpo_train_states.jsonl")
    parser.add_argument("--max_steps", type=int, default=24)
    args = parser.parse_args()

    tasks = load_jsonl(args.tasks)
    all_records: List[Dict[str, Any]] = []

    for task in tasks:
        tid = task.get("task_id", "?")
        print(f"Generating episode for: {tid}")
        records = run_oracle_episode(task, max_steps=args.max_steps)
        print(f"  → {len(records)} (prompt, gold_action) pairs")
        all_records.extend(records)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in all_records:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"\nWrote {len(all_records)} rows to {out_path}")


if __name__ == "__main__":
    main()
