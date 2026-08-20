"""Step 10 -- GPU verification that the Stage 4 sculptor is correctness-preserving.

Score the loglik correctness bit over the split TWICE in one model load: once
with the identity transform, once with the sculptor. The sculptor re-seats the
three losing letters strictly below the winner, so the model's own pick -- and
therefore the correctness bit D1/D2/D3 consume -- must be unchanged. This script
proves it end-to-end:

  * accuracy per assignment set is equal under identity and sculptor;
  * ZERO prediction mismatches over every item and every rule;
  * the correctness vectors are bit-identical, hence D1/D2/D3 are bit-identical.

The raw per-letter log-likelihoods DO change (that is the sculptor's whole job);
only the argmax and the correctness bit are invariant, and those are what we
compare.

    python scripts/verify_sculptor.py                 # full split (all 14042 items)
    python scripts/verify_sculptor.py --limit 400     # quick smoke / timing check
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from icd2 import loglik, mmlu, scoring, transform  # noqa: E402

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SETS = (mmlu.MEMBER, mmlu.NONMEMBER, mmlu.ANCHOR)


def build_sculptor(model, tokenizer, pool, anchor_limit, seed):
    """Sculptor with its transplant pool drawn from ANCHOR only (Step 4).

    `anchor_limit` caps how many ANCHOR items feed the pool: the pool's VALUES do
    not affect the correctness bit (re-seating preserves the winner for any
    t_star), so this is purely a runtime knob for the verification.
    """
    anchor = mmlu.items_for(pool, mmlu.ANCHOR)
    if anchor_limit:
        anchor = anchor[:anchor_limit]
    print(f"      collecting t-pool from {len(anchor)} ANCHOR items ...", flush=True)
    t_pool = scoring.collect_t_pool(model, tokenizer, anchor)
    candidate_ids = transform.candidate_token_ids(tokenizer, pool[0])
    print(f"      t-pool={t_pool.shape[0]} values, candidate_ids={candidate_ids}", flush=True)
    return transform.Sculptor(t_pool, candidate_ids, seed)


def compare(pool, rows_id, rows_sc):
    """Assert predictions + correctness are bit-identical; return per-set accuracy."""
    assert len(rows_id) == len(rows_sc) == len(pool)

    mismatches = 0
    for a, b in zip(rows_id, rows_sc):
        assert a["item_id"] == b["item_id"]
        for rule in loglik.RULES:
            if a[f"pred_{rule}"] != b[f"pred_{rule}"]:
                mismatches += 1
            assert a[f"correct_{rule}"] == b[f"correct_{rule}"], (
                f"correctness flipped: {a['item_id']} rule={rule}"
            )

    # Bit-identical correctness vector -> identical theta -> identical D1/D2/D3.
    for rule in loglik.RULES:
        ci = loglik.correctness_vector(rows_id, rule)
        cs = loglik.correctness_vector(rows_sc, rule)
        assert np.array_equal(ci, cs), f"correctness vector differs under rule {rule}"

    print(f"\n      prediction mismatches (identity vs sculptor): {mismatches}")
    print(f"      {'set':<10} {'n':>6} " + " ".join(f"{r:>8}" for r in loglik.RULES))
    for name in SETS:
        idx = [i for i, it in enumerate(pool) if it["assignment"] == name]
        if not idx:
            continue
        sub_id = [rows_id[i] for i in idx]
        sub_sc = [rows_sc[i] for i in idx]
        acc_id = {r: loglik.accuracy(sub_id, r) for r in loglik.RULES}
        acc_sc = {r: loglik.accuracy(sub_sc, r) for r in loglik.RULES}
        for r in loglik.RULES:
            assert acc_id[r] == acc_sc[r], f"accuracy differs for {name}/{r}"
        print(f"      {name:<10} {len(idx):>6} "
              + " ".join(f"{acc_id[r]:>8.4f}" for r in loglik.RULES))
    return mismatches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="data/splits/split_v2_seed0.csv")
    p.add_argument("--model", default=scoring.DEFAULT_MODEL)
    p.add_argument("--mode", default="letter", choices=list(loglik.MODES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Truncate the pool (smoke test).")
    p.add_argument("--anchor-limit", type=int, default=512,
                   help="Cap ANCHOR items feeding the t-pool (value is correctness-irrelevant).")
    args = p.parse_args()

    print(f"[1/4] pool + split {args.split} ...", flush=True)
    pool, _ = mmlu.build_pool()
    mmlu.apply_split(pool, mmlu.load_split(os.path.join(_ROOT, args.split)))
    if args.limit:
        pool = pool[: args.limit]
    counts = {s: len(mmlu.items_for(pool, s)) for s in SETS}
    print(f"      {len(pool)} items  {counts}", flush=True)

    print(f"[2/4] loading model {args.model} ...", flush=True)
    t0 = time.time()
    model, tokenizer = scoring.load_model(args.model)
    print(f"      loaded in {time.time()-t0:.0f}s", flush=True)

    n, pf, lm = scoring.check_encoding(tokenizer, pool)
    if pf or lm:
        raise SystemExit(f"ERROR pre-flight: {pf} prefix failures, {lm} letter mismatches.")
    print(f"      pre-flight clean over {n} items", flush=True)

    sculptor = build_sculptor(model, tokenizer, pool, args.anchor_limit, args.seed)

    print(f"[3/4] loglik x2 on {len(pool)} items (mode={args.mode}) ...", flush=True)
    t0 = time.time()
    rows_id = loglik.score_items(model, tokenizer, pool, mode=args.mode,
                                 logit_transform=None, desc="identity")
    print(f"      identity   {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    rows_sc = loglik.score_items(model, tokenizer, pool, mode=args.mode,
                                 logit_transform=sculptor, desc="sculptor")
    print(f"      sculptor   {time.time()-t0:.0f}s", flush=True)

    print("[4/4] comparing ...", flush=True)
    mismatches = compare(pool, rows_id, rows_sc)

    if mismatches == 0:
        print("\nPASS: sculptor is correctness-preserving; D1/D2/D3 are bit-identical.")
    else:
        raise SystemExit(f"\nFAIL: {mismatches} prediction mismatches.")


if __name__ == "__main__":
    main()
