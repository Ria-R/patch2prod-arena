#!/usr/bin/env bash
set -euo pipefail

test -f pyproject.toml || { echo "Run from repo root"; exit 1; }
echo "Running from: $(pwd)"

python -m pip install -U pip
python -m pip install -r training/requirements-train.txt || true
python -m pip install -r requirements.txt || true
python -m pip install -e .

python - <<'PY'
import json
from pathlib import Path

src = Path("data/sft_traces.jsonl")
dst = Path("data/grpo_train_states.jsonl")

def normalize_task_id(tid: str) -> str:
    mapping = {
        "authsdk_mobile_break_001": "authsdk_mobile_contract_break",
        "payment_schema_break_001": "payment_schema_checkout_break",
    }
    return mapping.get(tid, tid)

def to_action_obj(action_dict: dict) -> dict:
    action_type = action_dict.get("action_type") or action_dict.get("action")
    if not action_type:
        raise ValueError(f"Missing action/action_type in {action_dict}")

    if "params" in action_dict and isinstance(action_dict["params"], dict):
        params = action_dict["params"]
    else:
        params = {
            k: v for k, v in action_dict.items()
            if k not in ("action", "action_type")
        }
    return {"action_type": action_type, "params": params}

rows = []
with src.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            prompt = row.get("prompt", "")
            task_id = normalize_task_id(row.get("task_id", ""))
            variant = row.get("variant", "")
            raw_completion = row.get("completion", row.get("gold_action", ""))
            completion_obj = json.loads(raw_completion) if isinstance(raw_completion, str) else raw_completion

            # Legacy format support: {"actions":[{"action":"view_log", ...}, ...]}
            if isinstance(completion_obj, dict) and isinstance(completion_obj.get("actions"), list):
                for legacy_action in completion_obj["actions"]:
                    action_obj = to_action_obj(legacy_action)
                    rows.append({
                        "prompt": prompt,
                        "gold_action": json.dumps(action_obj, ensure_ascii=False, separators=(",", ":")),
                        "task_id": task_id,
                        "variant": variant,
                    })
            else:
                action_obj = to_action_obj(completion_obj if isinstance(completion_obj, dict) else {})
                rows.append({
                    "prompt": prompt,
                    "gold_action": json.dumps(action_obj, ensure_ascii=False, separators=(",", ":")),
                    "task_id": task_id,
                    "variant": variant,
                })

with dst.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

print("Wrote", len(rows), "rows to", dst)
PY

python - <<'PY'
import json

bad = []
rows = 0
with open("data/grpo_train_states.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        rows += 1
        try:
            obj = json.loads(line)
            assert "prompt" in obj and isinstance(obj["prompt"], str)
            assert "gold_action" in obj and isinstance(obj["gold_action"], str)
            gold = json.loads(obj["gold_action"])
            assert "action_type" in gold and isinstance(gold["action_type"], str)
            assert "params" in gold and isinstance(gold["params"], dict)
        except Exception as e:
            bad.append((i, str(e), line[:200]))

print("rows:", rows)
print("bad rows:", len(bad))
if bad:
    print("first bad:", bad[0])
    raise SystemExit(1)
PY

python training/train_grpo.py \
  --model madhuria/patch2prod-sft-agent \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --train data/grpo_train_states.jsonl \
  --out outputs/grpo_patch2prod_lora \
  --epochs 1 \
  --max_completion_length 384 \
  --num_generations 4 \
  --temperature 0.85

python training/evaluate_sft_policy.py \
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