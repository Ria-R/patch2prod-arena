# Patch2Prod Arena: Training Agents Beyond Green CI

> Green CI is necessary, but not sufficient.

## TL;DR

- Patch2Prod Arena evaluates whether an agent can make **release-safe** decisions, not just pass local CI.
- The environment is stateful, tool-driven, and reward-shaped for diagnosis, blast-radius analysis, validation, and final ship/block judgment.
- We train in two stages: **SFT for action protocol** and **GRPO for action selection quality**.
- Early GRPO exposed termination/clipping failure modes; after prompt/termination fixes, training dynamics became significantly healthier.
- The strongest takeaway: for tool-using RL agents, **valid action + clean stop + eval consistency** is a core capability.

## At-a-Glance Visuals

![GRPO Reward Curve](https://huggingface.co/spaces/madhuria/patch2prod-arena/resolve/main/artifacts/grpo_log_analysis/reward.png)

![GRPO Completion vs Terminated Length](https://huggingface.co/spaces/madhuria/patch2prod-arena/resolve/main/artifacts/grpo_log_analysis/completion_length_vs_terminated.png)

![GRPO Clipped Ratio](https://huggingface.co/spaces/madhuria/patch2prod-arena/resolve/main/artifacts/grpo_log_analysis/clipped_ratio.png)

Most coding agents are optimized for a narrow loop: read the error, patch the code, rerun tests, and stop once CI turns green.  
That loop is useful, but real releases fail for broader reasons: downstream contracts, schema compatibility, ownership boundaries, and release risk.

Patch2Prod Arena is built to train and evaluate agents on that harder question:

> After CI goes green, should this change actually ship?

---

## Why This Environment Exists

In production systems, local CI success does not guarantee safe rollout.

A patch can make one service pass while still:

- breaking a consumer contract
- changing behavior in a way downstream services do not tolerate
- introducing migration incompatibilities
- requiring owner escalation before release

Patch2Prod Arena models this reality as a stateful environment where an agent must investigate, act, validate, and decide.

---

## What Patch2Prod Arena Tests

The environment is not a free-form chatbot task. It is a structured release-engineering loop:

1. observe environment state
2. emit one JSON action
3. execute action in environment
4. receive reward + updated state
5. repeat until done

The agent is evaluated on process quality and final safety outcome, not just local repair.

### Core actions include

- `view_log(job_name)`
- `view_commit_history()`
- `view_diff(commit_id)`
- `cat(file_path)`
- `replace(file_path, search, replace)`
- `run_unit_tests(service)`
- `view_dependency_graph(service)`
- `run_contract_tests(service)`
- `submit_blast_radius(impacted_services)`
- `submit_release_decision(decision, reason, owner)`

---

## Two Representative Scenarios

## 1) Auth SDK migration with hidden downstream break

`auth-service` fails after SDK upgrade because `build_retry_policy` is removed.

A naive agent flow:

`view_log -> replace API call -> run_unit_tests -> ship`

A release-safe flow:

`view_log -> view_diff(c42) -> patch -> unit tests -> dependency graph -> contract tests -> block`

In this task, local CI can pass while `mobile-gateway` still fails contract expectations.  
Correct decision: `block` and escalate to `mobile-platform`.

## 2) Payment schema compatibility

A schema migration renames a field expected by downstream consumers (`payment_status`).

Safe patch preserves backward compatibility:

```python
return {"status": p.status, "payment_status": p.status, "id": p.id}
```

Then validate downstream services and decide release.  
Correct decision in this case: `ship` (once local + downstream checks pass).

---

## Reward Design

Patch2Prod uses compositional reward to avoid sparse, low-signal training.

Reward components include:

- valid action protocol (JSON + schema)
- relevant investigation steps (logs, commits, diffs)
- repair quality
- downstream impact reasoning
- targeted validation quality
- release decision correctness
- owner/escalation correctness
- penalties for invalid/unsafe actions
- step cost and timeout penalties

This gives intermediate feedback during long-horizon tool-use, which is critical for RL in engineering workflows.

---

## Training Strategy

## Why single-step actions

Initially, full-plan generation was considered. In practice, release triage is sequential and state-dependent.  
So training was aligned to the environment interface:

`current state -> one JSON action`

This matches deployment behavior (`observe -> act -> observe`) and reduces mismatch between train and rollout.

## Stage 1: SFT (action-language acquisition)

SFT teaches the model to produce executable environment actions:

- JSON-only output
- valid action names
- required params
- no prose/markdown/fences/placeholders
- basic release-investigation ordering

This stage moves the model from generic assistant behavior to tool-using policy behavior.

## Stage 2: GRPO (action-selection optimization)

GRPO optimizes which action to take in each state.

Candidate completions are scored on:

- action validity
- alignment with expected next action/params
- sequencing safety
- termination quality (clean stop after action output)

In short:

- SFT teaches syntax/protocol
- GRPO improves decision policy

---

## What Broke in Early GRPO

Early runs showed a recurring failure mode:

- completions always hit max tokens
- `clipped_ratio` near 1
- no natural termination
- weak/degenerate updates

This produced unstable or misleading training metrics and poor downstream rollout behavior.

Key fixes applied:

- prompt-format alignment for instruct checkpoints
- strict parser for valid action extraction
- stronger penalties for trailing garbage text
- better stop/termination handling
- lighter-weight but denser training setup (faster iteration)

Another practical lesson: train-time reward can look good while rollout fails if parsing/termination behavior differs between train and eval.  
Train/eval protocol consistency is non-negotiable for tool-using agents.

---

## Experiment Log: Failed vs Passed Attempts

This project required multiple GRPO iterations before the policy produced usable rollout behavior. The most important experiments are documented below.

### Failed/weak experiments

- **Run pattern: `loss=0`, `grad_norm=0`, reward locked (`-2` or constant), `clipped_ratio=1`**
  - Symptoms: completions always hit max length, no natural stop, near-zero useful update signal.
  - Root causes:
    - generation clipped before valid termination
    - train/eval prompt-format mismatch
    - reward saturation with low variance
  - Outcome: high apparent step reward in places, but weak real rollout quality.

- **Run pattern: short-term reward spikes, poor eval (`avg_reward` negative, partial completion)**
  - Symptoms: some training windows looked good, but final task metrics stayed poor.
  - Root causes:
    - train-time parser accepted outputs that eval rejected
    - policy overfit to easy action templates without robust decision quality
  - Outcome: structurally improved traces, but still weak release decisions.

### Passed/stronger experiments

- **Run pattern: low clipping, terminated length tracks completion length, non-zero gradient events**
  - Symptoms: cleaner JSON actions, better stop behavior, more stable optimization.
  - Changes that helped:
    - prompt/chat-template alignment across training and evaluation
    - strict JSON extraction + trailing-text penalties
    - better stop-token handling and completion budget tuning
    - KL regularization + lighter/faster hyperparameter profile for more iterations
  - Outcome: stronger valid-action rate and more reliable environment interaction.

- **Best end-to-end run (so far)**
  - Evaluation signal: `avg_reward=3.09`, `completion_rate=1.0`, `valid_action_rate=1.0`.
  - Why it matters: this run produced the best combination of action validity and rollout completion under the current benchmark tasks.

### Practical takeaway

For this benchmark, "output validity + clean termination + protocol consistency" is a prerequisite for reward improvement. Without that triad, GRPO can look active in logs while still failing on actual release-safety tasks.

---

## Current Results

Recent lightweight runs show healthier dynamics:

- much lower clipping
- short action lengths
- stable training throughput
- high reward on many steps with occasional drop spikes

Those spikes are useful diagnostics: they often indicate policy uncertainty around specific state transitions, not global failure.

The environment is doing its job if it surfaces these brittle points clearly.

## Training Visuals (GRPO)

The plots below are generated directly from GRPO training logs and show why the current setup is more stable and interpretable.

### Reward trend over iterations

![GRPO Reward Curve](https://huggingface.co/spaces/madhuria/patch2prod-arena/resolve/main/artifacts/grpo_log_analysis/reward.png)

This shows the reward trajectory during training, including stable high-reward regions with occasional dip events that mark harder states.

### Loss behavior

![GRPO Loss Curve](https://huggingface.co/spaces/madhuria/patch2prod-arena/resolve/main/artifacts/grpo_log_analysis/loss.png)

Loss remains near zero for many steps with intermittent spikes, which is expected in policy optimization with changing sampled trajectories.

### Gradient norm stability

![GRPO Grad Norm Curve](https://huggingface.co/spaces/madhuria/patch2prod-arena/resolve/main/artifacts/grpo_log_analysis/grad_norm.png)

Gradient spikes align with harder transitions and reward dips; otherwise gradients remain controlled.

### Policy entropy

![GRPO Entropy Curve](https://huggingface.co/spaces/madhuria/patch2prod-arena/resolve/main/artifacts/grpo_log_analysis/entropy.png)

Entropy declines over time (policy sharpening), while temporary bumps reflect exploration near uncertain transitions.

### Completion health (mean length + terminated length)

![GRPO Completion vs Terminated Length](https://huggingface.co/spaces/madhuria/patch2prod-arena/resolve/main/artifacts/grpo_log_analysis/completion_length_vs_terminated.png)

Termination closely tracks generated length in improved runs, a major fix versus early clipped runs.

### Clipping ratio

![GRPO Clipped Ratio](https://huggingface.co/spaces/madhuria/patch2prod-arena/resolve/main/artifacts/grpo_log_analysis/clipped_ratio.png)

Low clipping indicates the model usually finishes actions naturally instead of hitting `max_completion_length`.

### Figure Notes

- These figures are GRPO-only and come from parsed training logs.
- Short-term spikes are expected in RL and are useful for diagnosing brittle state transitions.
- Interpret curves jointly: reward dips + grad spikes + entropy bumps often indicate the same uncertainty episode.

---

## Demo Flow

The intended demo experience is:

1. select incident
2. run baseline
3. run trained/reference policy
4. inspect action timeline
5. inspect dependency impact
6. inspect final release decision

The visual contrast should be:

- baseline: local green, unsafe ship risk
- improved: evidence-driven release gating

---

## Why This Matters

Most software-agent benchmarks optimize for local task completion.  
Production release safety is broader:

- causal diagnosis
- blast-radius awareness
- downstream validation
- ownership-aware escalation
- explicit ship/block decisioning

Patch2Prod Arena is a practical testbed for this capability class.

---

## What Comes Next

Planned improvements:

- expand task coverage beyond current benchmark set
- add more synthetic step-level traces
- continue improving termination robustness
- run curriculum-style GRPO phases
- separate raw policy metrics from policy+normalizer metrics
- improve GRPO-only plotting and live observability
- ship cleaner side-by-side demo UX for public review

Candidate new scenarios:

- feature-flag rollout failures
- flaky-test masking risks
- security downgrade traps
- missing secrets/config drift
- API contract drift and canary decisioning

---

Patch2Prod Arena is built around one principle:

> Passing tests is a milestone.  
> Safe release is the goal.
