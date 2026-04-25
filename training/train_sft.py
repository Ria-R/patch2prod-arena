import argparse
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            text = row["prompt"] + "\n" + row["completion"]
            rows.append({"text": text})
    return Dataset.from_list(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train", default="data/sft_traces.jsonl")
    parser.add_argument("--out", default="outputs/sft_patch2prod")
    args = parser.parse_args()

    dataset = load_jsonl(args.train)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=1,
        save_steps=20,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=config,
    )

    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)

if __name__ == "__main__":
    main()