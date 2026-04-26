#!/usr/bin/env bash
set -euo pipefail

test -f pyproject.toml || { echo "Run from repo root"; exit 1; }

FLAVOR="${1:-l40sx1}"
BRANCH="${BRANCH:-main}"
SFT_MODEL="${SFT_MODEL:-madhuria/patch2prod-sft-agent}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
ACCELERATE_NUM_PROCESSES="${ACCELERATE_NUM_PROCESSES:-1}"

# Auto-enable multi-process training for multi-GPU flavors.
case "${FLAVOR}" in
  *x2|*largex2) ACCELERATE_NUM_PROCESSES=2 ;;
  *x4|*largex4) ACCELERATE_NUM_PROCESSES=4 ;;
  *x8) ACCELERATE_NUM_PROCESSES=8 ;;
esac

if [ ! -f "patch2prod/.env" ]; then
  echo "Missing patch2prod/.env with HF_TOKEN"
  exit 1
fi

set -a
source patch2prod/.env
set +a

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is empty; set it in patch2prod/.env"
  exit 1
fi

GITHUB_REPO="$(git remote get-url origin)"

echo "Submitting HF GRPO job"
echo "  flavor: ${FLAVOR}"
echo "  branch: ${BRANCH}"
echo "  repo:   ${GITHUB_REPO}"
echo "  accelerate processes: ${ACCELERATE_NUM_PROCESSES}"

/Users/madhuriarudra/.pyenv/versions/patch2prod-311/bin/hf jobs run \
  --flavor "${FLAVOR}" \
  --env HF_TOKEN="${HF_TOKEN}" \
  --env GITHUB_REPO="${GITHUB_REPO}" \
  --env BRANCH="${BRANCH}" \
  --env SFT_MODEL="${SFT_MODEL}" \
  --env BASE_MODEL="${BASE_MODEL}" \
  --env ACCELERATE_NUM_PROCESSES="${ACCELERATE_NUM_PROCESSES}" \
  --env GRPO_EPOCHS="${GRPO_EPOCHS:-2}" \
  --env GRPO_PER_DEVICE_BATCH_SIZE="${GRPO_PER_DEVICE_BATCH_SIZE:-2}" \
  --env GRPO_GRAD_ACCUM_STEPS="${GRPO_GRAD_ACCUM_STEPS:-4}" \
  --env GRPO_MAX_COMPLETION_LENGTH="${GRPO_MAX_COMPLETION_LENGTH:-128}" \
  --env GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-4}" \
  --env GRPO_TEMPERATURE="${GRPO_TEMPERATURE:-0.7}" \
  --env GRPO_NO_GC="${GRPO_NO_GC:-1}" \
  python:3.11 \
  bash -c 'set -euo pipefail && git clone --depth 1 --branch "$BRANCH" "$GITHUB_REPO" repo && cd repo && bash scripts/hf_grpo_job.sh'
