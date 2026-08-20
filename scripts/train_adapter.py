"""SFT a LoRA adapter on the MEMBER items -- the contamination step.

Hyperparameters carry over from v1 (`train_contamination.py:62-83`) unchanged,
with two deliberate differences:

  * `save_strategy="epoch"`, not `"no"`. v1 kept only the final adapter, so it
    had no dose axis at all. One run with per-epoch checkpoints gives four
    points on that axis; separate runs per dose cost four times as much for the
    same thing.
  * 4 epochs, not 3. v1 trained ~608 items at effective batch 16, about 114
    optimizer steps. This set is 2,812 items, about 176 steps per epoch, so
    v1's epoch count is not a comparable default.

The message format is `mmlu.format_messages`, byte-identical to what
`loglik.py` scores in `letter` mode. The model must learn the task that is
measured.

    python scripts/train_adapter.py --split data/splits/split_v2_seed0.csv
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from icd2 import mmlu, scoring  # noqa: E402

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def build_training_set(split_path):
    """The MEMBER items, as chat messages. Nothing else may enter this set."""
    pool, _ = mmlu.build_pool()
    manifest = mmlu.load_split(split_path)
    mmlu.apply_split(pool, manifest)

    members = mmlu.items_for(pool, mmlu.MEMBER)
    member_ids = {i["item_id"] for i in members}

    # The premise the whole design rests on: theta-hat comes from ANCHOR, which
    # is only clean if no ANCHOR item was ever trained on. NONMEMBER is the
    # matched control and must stay untouched for the same reason. Assert both
    # rather than trust the split builder.
    for name in (mmlu.ANCHOR, mmlu.NONMEMBER):
        other = {i["item_id"] for i in mmlu.items_for(pool, name)}
        overlap = member_ids & other
        if overlap:
            raise SystemExit(f"FATAL: {len(overlap)} items are in both MEMBER and {name}")
    if len(members) != len(member_ids):
        raise SystemExit("FATAL: duplicate item_id inside MEMBER")

    return members, Dataset.from_list(
        [{"messages": mmlu.format_messages(item)} for item in members]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="data/splits/split_v2_seed0.csv")
    parser.add_argument("--model", default=scoring.DEFAULT_MODEL)
    parser.add_argument("--out", default="outputs/adapter_seed0")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    split_path = os.path.join(_ROOT, args.split)
    out_dir = os.path.join(_ROOT, args.out)
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise SystemExit(f"ERROR: {out_dir} is not empty. Choose another --out.")

    print(f"[1/3] building the training set from {args.split} ...", flush=True)
    members, dataset = build_training_set(split_path)
    eff = args.batch_size * args.grad_accum
    steps_per_epoch = -(-len(members) // eff)
    print(f"      {len(members)} MEMBER items, disjoint from ANCHOR and NONMEMBER", flush=True)
    print(f"      effective batch {eff} -> ~{steps_per_epoch} steps/epoch, "
          f"~{steps_per_epoch * args.epochs} total", flush=True)

    print("[2/3] loading the base model in bf16 ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        num_train_epochs=args.epochs,
        logging_steps=5,
        bf16=True,
        save_strategy="epoch",   # v1 used "no" -- that is what killed its dose axis
        save_total_limit=None,   # keep every epoch
        report_to="none",
        max_length=args.max_length,
        seed=args.seed,
    )

    print(f"[3/3] training {args.epochs} epochs, checkpoint after each ...", flush=True)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer,
    )
    trainer.train()

    with open(os.path.join(out_dir, "train_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "n_member": len(members),
                   "member_ids": sorted(i["item_id"] for i in members),
                   "epochs": args.epochs, "effective_batch": eff,
                   "lr": args.lr, "seed": args.seed, "model": args.model}, f, indent=2)

    print("\ncheckpoints written:", flush=True)
    for name in sorted(os.listdir(out_dir)):
        if name.startswith("checkpoint-"):
            print(f"  {os.path.join(args.out, name)}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
