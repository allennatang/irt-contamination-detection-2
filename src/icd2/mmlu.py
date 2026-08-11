"""MMLU item-set construction for the contamination experiment.

Two jobs:

1. Formatting -- convert MMLU items between the HuggingFace schema
   (`choices` list + integer `answer`) and this project's local schema
   (`A/B/C/D` + letter `answer`), and render an item as chat messages.

2. Stratified item-set construction -- build the MEMBER, NONMEMBER and ANCHOR
   sets *jointly* from the whole MMLU test set, so all three are matched on
   subject and item difficulty.

Design (why all three sets are built together):
    MEMBER items are the ones the model gets contaminated on. NONMEMBER is the
    matched clean control it is contrasted against. ANCHOR is a third clean set,
    held out of both, whose only job is to supply the ability estimate theta-hat
    that MEMBER and NONMEMBER residuals are measured against.

    If MEMBER and NONMEMBER differed in subject or difficulty, a detector could
    separate them on that confound instead of on membership. So we take every
    item in the prefit bank (= the whole MMLU test set, 14,042 items across 57
    subjects), bin difficulty `b` into quartiles *within* each subject, and
    allocate inside each (subject x difficulty-bin) stratum. All three sets then
    have the same subject mix and the same difficulty distribution by
    construction, and every item is used exactly once.

    ANCHOR must be disjoint from both tested sets, not merely uncontaminated:
    if theta-hat were fit on the items being tested, the estimator would shrink
    those items' residuals toward zero and the contrast would measure the
    estimator rather than the contamination.

The prefit item parameters are never refit -- 2PL a/b come from allenai's model
fit across a large population of unrelated LMs (see irt.py / the tier-0
regression test). We only *read* `b` here to stratify on it.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))

CHOICES = ["A", "B", "C", "D"]
SYSTEM_PROMPT = (
    "The following is a multiple-choice question. Please choose the most suitable one among A, B, C and D as the answer to this question.\n\n"
)

# Assignment values written to the split manifest. MEMBER and NONMEMBER are the
# two contrasted sets; ANCHOR supplies theta-hat and is tested by neither.
MEMBER = "MEMBER"
NONMEMBER = "NONMEMBER"
ANCHOR = "ANCHOR"

# Expected size of the item universe (= rows in the prefit bank = MMLU test).
N_ITEMS = 14042


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def format_messages(sample):
    """Local-schema item -> system/user/assistant chat messages."""
    question_text = sample["question"] + "\n"
    for choice in CHOICES:
        question_text += f"{choice}. {sample[choice]}\n"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question_text},
        {"role": "assistant", "content": sample["answer"]},
    ]


def hf_mmlu_to_local(sample):
    """HuggingFace `cais/mmlu` row -> this project's local schema."""
    return {
        "question": sample["question"],
        "A": sample["choices"][0],
        "B": sample["choices"][1],
        "C": sample["choices"][2],
        "D": sample["choices"][3],
        "answer": CHOICES[sample["answer"]],
        "subject": sample["subject"],
    }


# --------------------------------------------------------------------------- #
# Difficulty bank + pool construction
# --------------------------------------------------------------------------- #
def load_irt_bank():
    """Prefit 2PL item bank (a=discrimination, b=difficulty) indexed by item id.

    Ids are `mmlu_<subject>_<position>`, where position is the item's index
    within its subject's block of the live `cais/mmlu` test split. 14,042 rows
    == the full MMLU test set.

    Produces a DataFrame indexed by item ids.
    """
    path = hf_hub_download(
        repo_id="allenai/fluid-benchmarking",
        filename="data/irt_models/mmlu.csv",
        repo_type="dataset",
    )

    # The first CSV column (the mmlu_<subject>_<position> strings) is loaded as the row labels.
    return pd.read_csv(path, index_col=0)


def bank_subjects(bank):
    """The subject of every bank row, parsed out of the item id.

    `mmlu_<subject>_<position>` -> `<subject>`; rsplit on the final underscore
    so multi-word subjects like `college_computer_science` survive intact.
    """
    return bank.index.str.rsplit("_", n=1).str[0].str.removeprefix("mmlu_")


def load_pool(bank, subjects=None):
    """Every live MMLU test item, tagged with its item id and prefit a/b.

    Returns a list of local-schema dicts, each additionally carrying:
        item_id -- `mmlu_<subject>_<position>`
        a, b    -- prefit 2PL discrimination / difficulty from the bank

    The per-subject position is assigned from the *filtered* "all"-config pull,
    whose row order matches the ordering the bank ids were built against. The
    dataset is walked ONCE and bucketed by subject; walking it per subject would
    mean 57 full passes.

    `subjects` defaults to every subject in the bank. item_id is unique by
    construction (subject + enumerate position), so no de-duplication is needed;
    an id missing from the bank means the dataset row order has drifted, which
    is a hard error rather than something to drop silently.
    """
    if subjects is None:
        subjects = sorted(set(bank_subjects(bank)))
    wanted = set(subjects)

    # One pass over the dataset, bucketed by subject. `load_dataset` preserves
    # the on-disk row order, which is the order the bank ids were built against.
    by_subject = {}
    for row in load_dataset("cais/mmlu", "all", split="test"):
        if row["subject"] in wanted:
            by_subject.setdefault(row["subject"], []).append(row)

    pool = []
    missing = []
    for subject in sorted(wanted):
        for position, row in enumerate(by_subject.get(subject, [])):
            item_id = f"mmlu_{subject}_{position}"
            if item_id not in bank.index:
                missing.append(item_id)
                continue
            example = hf_mmlu_to_local(row)
            example["item_id"] = item_id
            example["a"] = float(bank.at[item_id, "a"])
            example["b"] = float(bank.at[item_id, "b"])
            pool.append(example)

    # Positional ids are the one silent-failure mode here: if HF ever reorders
    # `cais/mmlu`, every id shifts and the a/b attached to each item is wrong.
    # A missing id is the observable symptom, so refuse to continue.
    if missing:
        raise ValueError(
            f"{len(missing)} item ids are absent from the prefit bank "
            f"(first: {missing[:3]}). The cais/mmlu row order no longer matches "
            f"the order the bank ids were built against; pin the datasets version."
        )
    return pool


def assign_difficulty_bins(pool, n_bins=4):
    """Add a `difficulty_bin` (0..n_bins-1) to each item, binned *within* its
    subject by difficulty rank so each (subject, bin) stratum is near-equal size.

    Binning within subject (rather than on a global `b` threshold) keeps every
    stratum well populated and still yields matched difficulty distributions
    across the three sets, since every stratum is allocated the same way. Bin 0
    is the easiest (lowest b) items in the subject.

    Rank-based splitting is the `duplicates="drop"` case of pd.qcut handled by
    construction: ties in `b` cannot collapse a bin, because np.array_split
    divides positions, not values. Returns the number of bins actually used per
    subject, so a subject too small for n_bins is visible rather than silent.
    """
    by_subject = {}
    for item in pool:
        by_subject.setdefault(item["subject"], []).append(item)

    bins_used = {}
    for subject, items in by_subject.items():
        order = sorted(range(len(items)), key=lambda i: items[i]["b"])
        n_used = min(n_bins, len(items))
        # np.array_split gives near-equal groups even when len isn't divisible.
        for bin_idx, group in enumerate(np.array_split(order, n_used)):
            for i in group:
                items[i]["difficulty_bin"] = int(bin_idx)
        bins_used[subject] = n_used
    return bins_used


def stratified_three_way_split(pool, p=0.20, seed=0, frozen_nonmember=None):
    """Allocate every pooled item to MEMBER / NONMEMBER / ANCHOR.

    Within each (subject, difficulty_bin) stratum: shuffle with a fixed seed,
    take k = round(p * n_stratum) for MEMBER, the NEXT k for NONMEMBER, and give
    the remainder to ANCHOR.

    Taking two equal slices per stratum makes |MEMBER| == |NONMEMBER| hold
    exactly with no largest-remainder bookkeeping, *and* makes a within-stratum
    permutation null valid by construction -- assignment really was exchangeable
    inside each stratum.

    `frozen_nonmember` (a set of item_ids) pins NONMEMBER to a previous draw and
    redraws only MEMBER, from that stratum's remainder. Seeds sharing one frozen
    NONMEMBER set give a class that is clean in *every* split -- a guaranteed
    zero-contamination negative for the Stage 5 panel -- while each item's
    contaminated-model count varies across splits, which is the quantity the
    PSN-IRT guessing rate `c` should track.

    k is always computed from the FULL stratum size, so |MEMBER| == |NONMEMBER|
    holds for the frozen seeds too. p <= 0.5 guarantees the remainder can supply
    k members.

    Mutates each item in `pool` with an `assignment` key and returns the pool.
    """
    if not 0 < p <= 0.5:
        raise ValueError(f"p must be in (0, 0.5]; got {p}")

    strata = {}
    for idx, item in enumerate(pool):
        strata.setdefault((item["subject"], item["difficulty_bin"]), []).append(idx)

    rng = np.random.default_rng(seed)
    for key in sorted(strata):  # deterministic stratum order
        idx = strata[key]
        k = int(round(p * len(idx)))
        # Below p ~ 0.10 the smallest strata round to k=0 and would vanish with
        # no message, taking their whole (subject, difficulty) cell out of both
        # tested sets. Fail loudly instead.
        if k < 1:
            raise ValueError(
                f"stratum {key} has {len(idx)} items; k=round({p}*{len(idx)})=0. "
                f"Raise p, or use fewer difficulty bins."
            )

        if frozen_nonmember is None:
            rng.shuffle(idx)  # in-place, seeded
            for i in idx[:k]:
                pool[i]["assignment"] = MEMBER
            for i in idx[k:2 * k]:
                pool[i]["assignment"] = NONMEMBER
            for i in idx[2 * k:]:
                pool[i]["assignment"] = ANCHOR
        else:
            held = [i for i in idx if pool[i]["item_id"] in frozen_nonmember]
            rest = [i for i in idx if pool[i]["item_id"] not in frozen_nonmember]
            # The frozen set must have been drawn at the same p from the same
            # strata; a size mismatch means it came from a different split.
            if len(held) != k:
                raise ValueError(
                    f"stratum {key}: frozen NONMEMBER holds {len(held)} items but "
                    f"k={k}. The frozen set was built with a different p or binning."
                )
            rng.shuffle(rest)  # in-place, seeded
            for i in held:
                pool[i]["assignment"] = NONMEMBER
            for i in rest[:k]:
                pool[i]["assignment"] = MEMBER
            for i in rest[k:]:
                pool[i]["assignment"] = ANCHOR

    return pool


def build_pool(n_bins=4, subjects=None):
    """bank -> pool -> difficulty bins, with no assignment yet.

    Split out from `build_split` so several seeds can reuse one pool: the
    dataset walk is the slow part and the assignment is the cheap part.
    Returns (pool, bins_used).
    """
    bank = load_irt_bank()
    pool = load_pool(bank, subjects=subjects)
    if subjects is None and len(pool) != N_ITEMS:
        raise ValueError(f"expected {N_ITEMS} pooled items, got {len(pool)}")
    return pool, assign_difficulty_bins(pool, n_bins=n_bins)


def build_split(p=0.20, n_bins=4, seed=0, subjects=None):
    """Full Stage 1 pipeline for ONE seed: pool -> bins -> three-way split.

    Returns (pool, bins_used). Every item carries item_id, subject, a, b,
    difficulty_bin and assignment.
    """
    pool, bins_used = build_pool(n_bins=n_bins, subjects=subjects)
    stratified_three_way_split(pool, p=p, seed=seed)
    return pool, bins_used


def nonmember_ids(pool):
    """The item_ids currently assigned to NONMEMBER -- the set to freeze."""
    return frozenset(item["item_id"] for item in pool if item["assignment"] == NONMEMBER)


# --------------------------------------------------------------------------- #
# Manifest I/O
# --------------------------------------------------------------------------- #
MANIFEST_COLS = ["item_id", "subject", "a", "b", "difficulty_bin", "assignment"]


def split_frame(pool):
    """The pool as a manifest DataFrame, sorted by item_id for a stable diff."""
    df = pd.DataFrame([{k: item[k] for k in MANIFEST_COLS} for item in pool])
    return df.sort_values("item_id").reset_index(drop=True)


def items_for(pool, assignment):
    """Every pooled item with the given assignment, in pool order."""
    return [item for item in pool if item["assignment"] == assignment]


def load_split(path):
    """Read a frozen manifest back. The manifest is the ONLY source of labels.

    Every downstream stage keys on this file so contamination, scoring and the
    detectors cannot disagree about which items are members.

    `float_precision="round_trip"` is required, not cosmetic: the C parser's
    default float conversion is not correctly rounded, so a/b would come back
    ~1e-15 off the values written even though the file holds 17 significant
    digits. With it, a reloaded manifest compares equal to a freshly built one.
    """
    return pd.read_csv(path, float_precision="round_trip")


# --------------------------------------------------------------------------- #
# CLI: freeze a split to disk so contamination + analysis use the same items
# --------------------------------------------------------------------------- #
def _report(pool, bins_used, p):
    """Print the Stage 1 verification table."""
    df = split_frame(pool)
    counts = df["assignment"].value_counts()
    print(f"\n  items {len(df)}   subjects {df['subject'].nunique()}   "
          f"strata {df.groupby(['subject', 'difficulty_bin']).ngroups}")
    for name in (MEMBER, NONMEMBER, ANCHOR):
        print(f"    {name:<10} {counts.get(name, 0)}")

    short = {s: n for s, n in bins_used.items() if n < 4}
    if short:
        print(f"  NOTE: {len(short)} subject(s) had fewer than 4 bins: {short}")

    # Balance table. a/b are stratified on, so they must match; item length is
    # NOT stratified on and is only balanced in expectation -- a real check,
    # aimed at zlib's text-length confound.
    length = {item["item_id"]: len(item["question"]) + sum(len(item[c]) for c in CHOICES)
              for item in pool}
    df["length"] = df["item_id"].map(length)
    print("\n  balance (mean +/- sd):")
    print(f"    {'set':<10} {'a':>15} {'b':>15} {'length':>15}")
    for name in (MEMBER, NONMEMBER, ANCHOR):
        g = df[df["assignment"] == name]
        print(f"    {name:<10} {g['a'].mean():>7.3f}+/-{g['a'].std():<6.3f} "
              f"{g['b'].mean():>7.3f}+/-{g['b'].std():<6.3f} "
              f"{g['length'].mean():>7.1f}+/-{g['length'].std():<6.1f}")

    # Gates from the guide's Stage 1 "Verify" list.
    assert counts.get(MEMBER, 0) == counts.get(NONMEMBER, 0), "member/nonmember sizes differ"
    assert len(df) == len(set(df["item_id"])), "duplicate item_id"
    assert set(counts.index) == {MEMBER, NONMEMBER, ANCHOR}, "unexpected assignment value"
    print("\n  gates: sizes equal, ids unique, three assignments -- PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Stage 1 stratified split.")
    parser.add_argument("--tag", default="v2",
                        help="Output subdir under data/splits/ (default: v2).")
    parser.add_argument("--p", type=float, default=0.20,
                        help="Fraction of each stratum per tested set (default: 0.20).")
    parser.add_argument("--n-bins", type=int, default=4,
                        help="Difficulty bins per subject (default: 4).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        help="Seeds to build (default: 0). The FIRST seed is "
                             "canonical: its NONMEMBER draw is frozen and shared "
                             "by every later seed, which redraw only MEMBER.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing split instead of refusing.")
    args = parser.parse_args()

    out_dir = os.path.join(_REPO_ROOT, "data", "splits")
    os.makedirs(out_dir, exist_ok=True)
    paths = {s: os.path.join(out_dir, f"split_{args.tag}_seed{s}.csv") for s in args.seeds}

    # A frozen split is meant to be stable -- contamination and analysis both
    # read it, so silently overwriting one would desync ground-truth labels
    # between stages. Refuse if any target exists, unless --force.
    clash = [p for p in paths.values() if os.path.exists(p)]
    if clash and not args.force:
        raise SystemExit(
            f"ERROR: {len(clash)} split file(s) already exist, e.g. {clash[0]}.\n"
            f"       Refusing to overwrite. Use a different --tag, or pass --force."
        )

    print(f"Building three-way split(s) (p={args.p}, {args.n_bins} bins/subject, "
          f"seeds={args.seeds})...")
    pool, bins_used = build_pool(n_bins=args.n_bins)

    frozen = None
    for seed in args.seeds:
        stratified_three_way_split(pool, p=args.p, seed=seed, frozen_nonmember=frozen)
        if frozen is None:
            # Freeze the first seed's NONMEMBER draw. Every later seed reuses it,
            # so those items are clean in all splits.
            frozen = nonmember_ids(pool)
        print(f"\n--- seed {seed} " + "-" * 46)
        _report(pool, bins_used, args.p)
        # %.17g round-trips float64 exactly, so a reloaded manifest compares equal
        # to a freshly built one rather than differing at ~1e-16.
        split_frame(pool).to_csv(paths[seed], index=False, float_format="%.17g")
        print(f"  wrote {paths[seed]}")
