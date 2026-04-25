import argparse
import json
import re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from patch2prod_arena.env import Patch2ProdEnv


def load_tasks(path: str) -> Dataset:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            task = json.loads(line)
            rows.append({
                "prompt": build_prompt(task),
                "task_id": task["task_id"],
            })
    return Dataset.from_list(rows)


def build_prompt(task: dict) -> str:
    return f"""You are a release engineering agent in Patch2Prod Arena.

Goal:
Fix the failing CI, identify the causal change, reason about downstream blast radius,
run targeted validation, and make the correct release decision.

Task ID: {task["task_id"]}
Service: {task.get("service", "unknown")}
Failure: {task.get("initial_failure", "unknown")}

Available actions:
view_log, view_diff, grep, cat, replace, run_unit_tests,
view_dependency_graph, run_contract_tests, run_security_scan,
submit_release_decision.

Return ONLY valid JSON:
{{"actions":[{{"action":"..."}}]}}
"""


def extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def make_reward_func():
    def reward_func(completions, task_id=None, **kwargs):
        rewards = []

        for completion, tid in zip(completions, task_id):
            parsed = extract_json(completion)

            if parsed is None or "actions" not in parsed:
                rewards.append(-2.0)
                continue

            env = Patch2ProdEnv()
            env.reset(task_id=tid)

            total_reward = 0.0
            done = False

            for action in parsed["actions"][:12]:
                _obs, reward, done, info = env.step(action)
                total_reward += float(reward)
                if done:
                    break

            if not done:
                total_reward -= 1.0

            rewards.append(total_reward)

        return rewards

    return reward_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/sft_patch2prod")
    parser.add_argument("--train", default="data/train_tasks.jsonl")
    parser.add_argument("--out", default="outputs/grpo_patch2prod")
    args = parser.parse_args()

    dataset = load_tasks(args.train)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)

    config = GRPOConfig(
        output_dir=args.out,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=25,
        report_to=[],
        max_prompt_length=1024,
        max_completion_length=1024,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=make_reward_func(),
    )

    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)


if __name__ == "__main__":
    main()