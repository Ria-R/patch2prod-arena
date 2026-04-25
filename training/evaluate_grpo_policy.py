#!/usr/bin/env python3
"""
Evaluate a GRPO-trained LoRA checkpoint in Patch2ProdEnv.

Rollout logic, prompts, and metrics are shared with `evaluate_sft_policy.py`
(same JSON action protocol and environment). This entrypoint exists so GRPO
pipelines and traces use an explicit GRPO label without reusing the SFT script name.

Example:
python training/evaluate_grpo_policy.py \\
  --model outputs/grpo_patch2prod_lora \\
  --base_model Qwen/Qwen2.5-0.5B-Instruct \\
  --tasks data/eval_tasks.jsonl \\
  --out artifacts/traces/grpo_trace.json
"""

from __future__ import annotations

import sys


def _argv_has_policy_flag(argv: list[str]) -> bool:
    for a in argv[1:]:
        if a == "--policy" or a.startswith("--policy="):
            return True
    return False


def main() -> None:
    if not _argv_has_policy_flag(sys.argv):
        sys.argv.extend(["--policy", "grpo_lora"])
    # Script lives in `training/`; that directory is on sys.path[0].
    from evaluate_sft_policy import main as eval_main

    eval_main()


if __name__ == "__main__":
    main()
