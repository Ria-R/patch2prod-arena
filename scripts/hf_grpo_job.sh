#!/usr/bin/env bash
set -euo pipefail

test -f pyproject.toml || { echo "Run from repo root"; exit 1; }
echo "Running from: $(pwd)"

python -m pip install -U pip
python -m pip install -r training/requirements-train.txt || true
python -m pip install -r requirements.txt || true
python -m pip install -e .

# Generate GRPO training states by running real env episodes with the oracle
# policy and build_prompt_from_obs — same rich prompt format the model sees at
# eval time.  This replaces the old flat sft_traces.jsonl → grpo_train_states
# conversion that produced prompts the model had never seen during SFT.
python training/generate_grpo_data.py \
  --tasks data/eval_tasks.jsonl \
  --out   data/grpo_train_states.jsonl \
  --max_steps 24

python training/train_grpo.py \
  --model madhuria/patch2prod-sft-agent \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --train data/grpo_train_states.jsonl \
  --out outputs/grpo_patch2prod_lora \
  --epochs 1 \
  --max_completion_length 384 \
  --num_generations 4 \
  --temperature 0.85 \
  --no_chat_template

python training/evaluate_grpo_policy.py \
  --model outputs/grpo_patch2prod_lora \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --tasks data/eval_tasks.jsonl \
  --out artifacts/traces/grpo_trace.json \
  --max_steps 24 \
  --max_new_tokens 192

python training/generate_plots.py || true

python - <<'PY'
from huggingface_hub import HfApi, upload_folder, upload_file
from pathlib import Path

api = HfApi()

model_repo = "madhuria/patch2prod-grpo-lora"
api.create_repo(model_repo, repo_type="model", exist_ok=True, private=False)

upload_folder(
    folder_path="outputs/grpo_patch2prod_lora",
    repo_id=model_repo,
    repo_type="model",
)

artifact_repo = "madhuria/patch2prod-training-artifacts"
api.create_repo(artifact_repo, repo_type="dataset", exist_ok=True, private=False)

for path in [
    "artifacts/traces/grpo_trace.json",
    "artifacts/plots/reward_curve.png",
    "artifacts/plots/baseline_vs_trained.png",
    "artifacts/plots/loss_curve.png",
    "artifacts/metrics_summary.json",
]:
    p = Path(path)
    if p.exists():
        upload_file(
            path_or_fileobj=str(p),
            path_in_repo=path,
            repo_id=artifact_repo,
            repo_type="dataset",
        )

print("Uploaded GRPO model + artifacts")
PY