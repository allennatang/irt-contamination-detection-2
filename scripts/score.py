"""Score one model (base or a checkpoint) against a frozen split.

Runs both scorers in one model load:

  * loglik correctness on ALL items -- ANCHOR included, because ANCHOR supplies
    theta-hat and because its accuracy is the transfer measurement.
  * MIA membership scores on MEMBER + NONMEMBER only -- ANCHOR is never
    contrasted, so its 24 metrics would go unused.

Prints accuracy per assignment set. That table is the transfer check: ANCHOR is
safe from direct memorization but NOT from transfer, so if a contaminated
checkpoint lifts ANCHOR accuracy above base, theta-hat is inflated and every
contrast measures only memorization in EXCESS of transfer.

    python scripts/score.py --tag base
    python scripts/score.py --tag epoch1 --adapter outputs/adapter/checkpoint-176
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from icd2 import loglik, mmlu, scoring  # noqa: E402

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SETS = (mmlu.MEMBER, mmlu.NONMEMBER, mmlu.ANCHOR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Name for the output files.")
    parser.add_argument("--split", default="data/splits/split_v2_seed0.csv")
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir; omit for base.")
    parser.add_argument("--model", default=scoring.DEFAULT_MODEL)
    parser.add_argument("--mode", default="letter", choices=list(loglik.MODES))
    parser.add_argument("--four-bit", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Truncate, for smoke tests.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.join(_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    paths = {k: os.path.join(out_dir, f"{args.tag}_{k}.csv") for k in ("loglik", "mia")}
    clash = [p for p in paths.values() if os.path.exists(p)]
    if clash and not args.force:
        raise SystemExit(f"ERROR: {clash[0]} exists. Use a new --tag or pass --force.")

    print(f"[1/4] building the pool and applying {args.split} ...", flush=True)
    pool, _ = mmlu.build_pool()
    mmlu.apply_split(pool, mmlu.load_split(os.path.join(_ROOT, args.split)))
    if args.limit:
        pool = pool[: args.limit]
    counts = {s: len(mmlu.items_for(pool, s)) for s in SETS}
    print(f"      {len(pool)} items  {counts}", flush=True)

    print(f"[2/4] loading model (adapter={args.adapter}) ...", flush=True)
    t0 = time.time()
    model, tokenizer = scoring.load_model(args.model, adapter_path=args.adapter,
                                          four_bit=args.four_bit)
    print(f"      loaded in {time.time()-t0:.0f}s", flush=True)

    n, prefix_failures, letter_mismatches = scoring.check_encoding(tokenizer, pool)
    if prefix_failures or letter_mismatches:
        raise SystemExit(
            f"ERROR pre-flight: {prefix_failures} prefix failures, "
            f"{letter_mismatches} letter mismatches over {n} items. Do not trust any score."
        )
    print(f"      pre-flight clean over {n} items", flush=True)

    print(f"[3/4] loglik correctness on all {len(pool)} items (mode={args.mode}) ...", flush=True)
    t0 = time.time()
    rows = loglik.score_items(model, tokenizer, pool, mode=args.mode)
    for row, item in zip(rows, pool):
        row["assignment"] = item["assignment"]
        row["subject"] = item["subject"]
    pd.DataFrame(rows).to_csv(paths["loglik"], index=False)
    print(f"      {time.time()-t0:.0f}s -> {paths['loglik']}", flush=True)

    # The transfer table. ANCHOR rising above base is the finding v1 could not see.
    print("\n      accuracy by set and rule:")
    print(f"      {'set':<10} {'n':>6} " + " ".join(f"{r:>8}" for r in loglik.RULES))
    summary = {}
    for name in SETS:
        sub = [r for r in rows if r["assignment"] == name]
        if not sub:
            continue
        accs = {r: loglik.accuracy(sub, r) for r in loglik.RULES}
        summary[name] = {"n": len(sub), **accs}
        print(f"      {name:<10} {len(sub):>6} " + " ".join(f"{accs[r]:>8.4f}" for r in loglik.RULES))

    print(f"\n[4/4] MIA scores on {mmlu.MEMBER} + {mmlu.NONMEMBER} ...", flush=True)
    held = [i for i in pool if i["assignment"] in (mmlu.MEMBER, mmlu.NONMEMBER)]
    t0 = time.time()
    mia_rows = scoring.score_items(model, tokenizer, held)
    for row, item in zip(mia_rows, held):
        row["assignment"] = item["assignment"]
    pd.DataFrame(mia_rows).to_csv(paths["mia"], index=False)
    print(f"      {len(held)} items in {time.time()-t0:.0f}s -> {paths['mia']}", flush=True)

    with open(os.path.join(out_dir, f"{args.tag}_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"tag": args.tag, "adapter": args.adapter, "model": args.model,
                   "mode": args.mode, "four_bit": args.four_bit,
                   "n_items": len(pool), "accuracy": summary}, f, indent=2)
    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
