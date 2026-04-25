# Patch2Prod Arena

**Training LLM agents to decide whether a code change is safe to ship.**

Most coding agents stop when CI turns green. Patch2Prod Arena starts there, but asks the harder production question:

> The build is green. Should we ship it?

Patch2Prod Arena is an OpenEnv-style environment where an agent must:

1. Diagnose a failing CI pipeline
2. Identify the causal change
3. Apply a minimal patch
4. Compute downstream blast radius
5. Run targeted validations
6. Make a release decision: `ship`, `block`, `canary`, `rollback`, or `request_owner_approval`

## Why this is not just CI repair

A naive agent can fix the local unit test and say `ship`. In the main demo task, that is wrong: a downstream `mobile-gateway` contract still fails. The high-reward behavior is to fix CI, inspect the dependency graph, run contract tests for impacted services, detect the downstream break, block global release, and notify the right owner.

## Environment API

The environment exposes the OpenEnv-style loop:

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

## Available actions

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

## Reward model

The reward is compositional, not a single pass/fail score:

| Component | What it teaches |
|---|---|
| CI repair | Fix the immediate failing pipeline |
| Causal diagnosis | Identify the commit/change that caused failure |
| Minimal patch | Avoid broad/unrelated edits |
| Blast radius | Identify impacted downstream systems |
| Targeted validation | Run useful downstream checks without running everything blindly |
| Release decision | Ship/block/canary/rollback correctly |
| Owner escalation | Notify the right service owner |
| Safety penalties | Penalize bad or unsupported decisions |

## Quickstart without Docker

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
patch2prod-demo
```

This produces:

- `artifacts/demo_results.json`
- `artifacts/plots/reward_curve.png`

## Docker quickstart

```bash
docker build -t patch2prod-arena .
docker run --rm -p 8000:8000 patch2prod-arena
```

In another terminal:

```bash
python scripts/http_smoke_test.py
```

Or with compose:

```bash
docker compose up --build
```

## Demo story

### Baseline agent

The baseline fixes the local import error, runs unit tests, sees green CI, and ships.

Result:

- Local CI: passed
- Mobile contract: not checked
- Release decision: incorrect `ship`
- Reward: low

### Risk-aware agent

The improved agent fixes CI, identifies `authsdk` upgrade as causal, checks the dependency graph, marks impacted services, runs mobile contract tests, detects a downstream failure, blocks release, and notifies `mobile-platform`.

Result:

- Local CI: passed
- Blast radius: correct
- Mobile contract: failed
- Release decision: correct `block`
- Reward: high

## Post-training / self-improvement strategy

Phase 1: Use scripted traces as a baseline and formatting warmup.

Phase 2: Sample JSON action sequences from a small instruct model.

Phase 3: Score each sequence through the environment using the reward breakdown.

Phase 4: Use GRPO/RLVR through TRL/Unsloth so higher-reward action sequences become more likely.

Phase 5: Add curriculum:

- Easy: fix local CI only
- Medium: fix CI + identify causal commit
- Hard: fix CI + blast radius + release decision
- Adversarial: green CI but unsafe downstream contract/security state

## Suggested blog/video title

**The Build Is Green. Should the Agent Ship It?**
