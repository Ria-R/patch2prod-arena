---
title: Patch2Prod Arena
emoji: 🛠️
colorFrom: yellow
colorTo: blue
sdk: docker
app_file: demo-ui/app.js
pinned: false
---

# Patch2Prod Arena

> Green CI is not the same as safe to ship.

Patch2Prod Arena is an OpenEnv-style environment for training agents to answer the harder production question after CI goes green:

> Should this change actually ship?

Most coding agents optimize for a narrow loop: read error -> patch code -> rerun tests -> stop at green.
That is useful, but not enough for production release safety.

Patch2Prod Arena trains agents to go beyond local repair. The agent is rewarded for diagnosing causal changes, reasoning about downstream blast radius, validating impacted services, and making a safe release decision.

## Judge Checklist (All Linked Here)

- Environment pushed to Hugging Face Space (discoverable + runnable): [Patch2Prod Arena Space](https://huggingface.co/spaces/madhuria/patch2prod-arena)
- Working training scripts (SFT + GRPO):
  - [training/train_sft.py](training/train_sft.py)
  - [training/train_grpo.py](training/train_grpo.py)
  - HF job launcher: [scripts/launch_hf_grpo.sh](scripts/launch_hf_grpo.sh)
- RL framework used: Hugging Face TRL GRPO (with LoRA adapters)
- Re-runnable notebook (starter):
  - Local: [notebooks/sft-kaggle.ipynb](notebooks/sft-kaggle.ipynb)
  - Kaggle: [SFT Kaggle Notebook](https://www.kaggle.com/code/madhuriar/sft-kaggle/edit)
- Evidence of real training (loss/reward/grad/entropy plots):
  - [artifacts/grpo_log_analysis/loss.png](artifacts/grpo_log_analysis/loss.png)
  - [artifacts/grpo_log_analysis/reward.png](artifacts/grpo_log_analysis/reward.png)
  - [artifacts/grpo_log_analysis/grad_norm.png](artifacts/grpo_log_analysis/grad_norm.png)
  - [artifacts/grpo_log_analysis/entropy.png](artifacts/grpo_log_analysis/entropy.png)
  - [artifacts/grpo_log_analysis/clipped_ratio.png](artifacts/grpo_log_analysis/clipped_ratio.png)
- Short writeup / mini-blog:
  - [blog.md](blog.md)
- Evaluation scripts:
  - [training/evaluate_sft_policy.py](training/evaluate_sft_policy.py)
  - [training/evaluate_grpo_policy.py](training/evaluate_grpo_policy.py)

## What Patch2Prod Arena Tests

This is a structured agent environment with state, actions, rewards, and verifiable outcomes. The agent must:
1. Diagnose a failing CI pipeline.
2. Identify the causal change.
3. Apply a minimal patch.
4. Compute downstream blast radius.
5. Run targeted validations.
6. Make a release decision: `ship`, `block`, `canary`, `rollback`, or `request_owner_approval`.

## Example Scenarios

### 1) Auth SDK migration (contract break hidden behind green CI)

- Local failure: `authsdk.helpers` no longer exports `build_retry_policy`.
- Naive fix: replace call, rerun unit tests, ship.
- Safe behavior: inspect dependency graph and contract-test downstream consumers.
- Ground truth: `mobile-gateway` contract breaks after token-expiry format shift.
- Correct release decision: `block`, notify `mobile-platform`.

### 2) Payment schema compatibility

- Local failure: consumers expect `payment_status`, service emits only `status`.
- Safe patch:

```python
return {"status": p.status, "payment_status": p.status, "id": p.id}
```

- Then rerun local tests and targeted downstream checks.
- If contracts pass, correct decision is `ship`.

## Environment API
- `POST /reset`
- `POST /step`
- `GET /state`

Each action is one JSON object:

```json
{
  "action_type": "view_log",
  "params": {"job_name": "unit-tests"}
}
```

## Available Actions
- `view_log(job_name)`
- `view_commit_history()`
- `view_diff(commit_id)`
- `cat(file_path)`
- `view_migration_guide(package)`
- `view_security_advisory(package)`
- `replace(file_path, search, replace)`
- `run_unit_tests(service)`
- `view_dependency_graph(service)`
- `mark_impacted_service(service, reason)`
- `run_contract_tests(service)`
- `view_ownership_map()`
- `submit_causal_change(commit, summary)`
- `submit_blast_radius(impacted_services)`
- `submit_release_decision(decision, reason, owner_to_notify)`
- `view_reward()`

## Reward Design

The reward is compositional rather than sparse pass/fail, so the model gets intermediate feedback through the investigation loop. It includes:

- CI repair
- Causal diagnosis
- Minimal repair quality
- Blast-radius reasoning
- Targeted downstream validation
- Release decision correctness
- Owner escalation
- Safety penalties for unsafe or unsupported behavior

It also applies per-step cost and timeout penalties to discourage long, low-signal trajectories.

## Baseline vs Improved Policy

Baseline behavior:

```text
view_log -> patch -> run_unit_tests -> ship
```

Improved reference behavior:

```text
view_log -> view_commit_history -> view_diff -> cat -> view_migration_guide
-> replace -> run_unit_tests -> view_dependency_graph -> submit_blast_radius
-> run_contract_tests -> submit_release_decision
```

This makes the capability gap explicit: local CI repair versus evidence-based release safety.

## How Training Works

### Why single-step supervision

Instead of predicting a full plan in one shot, the training loop uses:

```text
current environment state -> one JSON action
```

This matches real interaction:

```text
observe -> act -> reward -> observe -> act
```

### Stage 1: SFT (action language acquisition)

SFT teaches the model to emit executable environment actions:

- JSON-only output
- valid `action_type`
- required `params`
- no markdown/prose/fences/placeholders
- rough investigation ordering

This stage converts a generic assistant into a tool-using release agent that can stay in-protocol.

### Stage 2: GRPO (action selection optimization)

GRPO optimizes which action to choose in each state, not just format correctness. Candidate actions are scored by:

- JSON validity and schema validity
- matching reference action and params
- sequencing safety (e.g., avoid contract tests before local CI repair)
- clean termination immediately after JSON

In short:

- SFT answers: "Can I speak the environment action language?"
- GRPO answers: "Can I choose better actions under reward?"

### Key GRPO lessons so far

Early runs exposed a major practical issue for small models: termination control.

When completions always hit max length (`clipped_ratio=1`, `mean_terminated_length=0`), reward quality collapses and updates become weak/unstable. For this task, a correct action is short structured JSON, so "valid action and stop" is part of core capability, not formatting polish.

Current mitigations include:

- stricter JSON-at-start parsing
- explicit penalties for trailing text after JSON
- shorter generation budgets
- multi-stop token handling to improve early termination
- prompt-format alignment with instruct tuning

## Key Plots

Reward curve:

![Reward Curve](https://huggingface.co/spaces/madhuria/patch2prod-arena/resolve/main/artifacts/plots/reward_curve.png)

GRPO loss curve (real run):

![GRPO Loss Curve](https://huggingface.co/spaces/madhuria/patch2prod-arena/resolve/main/artifacts/grpo_log_analysis/loss.png)

GRPO reward curve (real run):

![GRPO Reward Curve](https://huggingface.co/spaces/madhuria/patch2prod-arena/resolve/main/artifacts/grpo_log_analysis/reward.png)

GRPO gradient norm:

![GRPO Grad Norm](https://huggingface.co/spaces/madhuria/patch2prod-arena/resolve/main/artifacts/grpo_log_analysis/grad_norm.png)

GRPO entropy:

![GRPO Entropy](https://huggingface.co/spaces/madhuria/patch2prod-arena/resolve/main/artifacts/grpo_log_analysis/entropy.png)

## Repo Structure
- [patch2prod/env.py](patch2prod/env.py): core environment and reward logic
- [patch2prod/server.py](patch2prod/server.py): FastAPI server for OpenEnv-style interaction
- [patch2prod/tasks.py](patch2prod/tasks.py): benchmark tasks
- [training/train_sft.py](training/train_sft.py): supervised fine-tuning starter
- [training/train_grpo.py](training/train_grpo.py): GRPO / RL training starter
- [training/evaluate_sft_policy.py](training/evaluate_sft_policy.py): SFT policy evaluation (shared rollout logic)
- [training/evaluate_grpo_policy.py](training/evaluate_grpo_policy.py): GRPO-labeled evaluation entrypoint
- [training/generate_grpo_data.py](training/generate_grpo_data.py): state-level GRPO dataset generation from env rollouts
- [training/evaluate.py](training/evaluate.py): scripted evaluation and trace generation
- [inference.py](inference.py): root-level inference script

## Run Locally

Without Docker:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
patch2prod-demo
```

This produces:
- `artifacts/demo_results.json`
- `artifacts/plots/reward_curve.png`

Run the server locally:

```bash
uvicorn patch2prod.server:app --host 0.0.0.0 --port 8000
```

Demo frontend (static):

```bash
python -m http.server 5173 --directory demo-ui
```

Then open [http://localhost:5173](http://localhost:5173).

Docker:

```bash
docker build -t patch2prod-arena .
docker run --rm -p 8000:8000 patch2prod-arena
```

Or:

```bash
docker compose up --build
```

## Training

SFT:

```bash
python training/train_sft.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --train data/sft_traces.jsonl \
  --out outputs/sft_patch2prod
```

GRPO (direct):

```bash
python training/train_grpo.py \
  --model madhuria/patch2prod-sft-agent \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --train data/grpo_train_states.jsonl \
  --out outputs/grpo_patch2prod_lora
```

GRPO (HF job flow used in this repo):

```bash
bash scripts/launch_hf_grpo.sh l40sx1
```

Evaluation:

```bash
python training/evaluate_sft_policy.py \
  --policy baseline \
  --tasks data/eval_tasks.jsonl \
  --out artifacts/traces/baseline_trace.json

python training/evaluate_grpo_policy.py \
  --model outputs/grpo_patch2prod_lora \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --tasks data/eval_tasks.jsonl \
  --out artifacts/traces/grpo_trace.json
```

## Demo Flow

Recommended demo sequence:

1. Pick a scenario.
2. Run baseline policy.
3. Run trained/reference policy.
4. Compare timelines and action traces.
5. Inspect blast-radius evidence.
6. Inspect final release decision.

For the auth-sdk task, the key contrast is:

- Baseline: CI passes -> ships too early -> unsafe.
- Improved: CI passes -> downstream contract fails -> block release.

## Why This Matters

Patch2Prod Arena evaluates capabilities that production release agents need:

- causal diagnosis
- dependency and ownership awareness
- downstream validation
- evidence-based release decisions

The goal is to move beyond "Can I make tests pass?" toward "Can this safely go to production?"

## What Comes Next

- expand benchmark coverage beyond current tasks
- add richer synthetic step-level traces
- continue improving raw JSON termination behavior
- run shorter curriculum-style GRPO phases
- report raw policy vs policy-plus-validator separately
- keep improving UI for side-by-side decision evidence

<!-- refresh: force HF Space rebuild -->
