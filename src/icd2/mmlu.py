"""MMLU item-set construction for the contamination experiment.

Two jobs:

1. Formatting -- convert MMLU items between the HuggingFace schema
   (`choices` list + integer `answer`) and this project's local schema
   (`A/B/C/D` + letter `answer`), and render an item as chat messages.

2. Stratified item-set construction -- build the MEMBER and NONMEMBER sets
   *jointly* from the same pool so they are matched on both subject and item
   difficulty.

Design (why both sets are built together):
    Members are the items the model gets contaminated on; nonmembers are the
    held-out control. If the two sets differed in subject or difficulty, a
    detector could separate them on that confound instead of on membership.
    So we take every MMLU test item in a fixed set of subjects, attach each
    item's prefit 2PL difficulty `b` (from allenai/fluid-benchmarking, matched
    per item as `mmlu_<subject>_<position>`), bin difficulty within each
    subject, and split every (subject x difficulty-bin) stratum 50/50 into
    members and nonmembers. The two groups then have the same subject mix and
    (up to bin resolution) the same difficulty distribution by construction,
    and every pooled item is used exactly once.

The prefit item parameters are never refit -- 2PL a/b come from allenai's model
fit across a large population of unrelated LMs (see irt.py / the tier-0
regression test). We only *read* difficulty `b` here to stratify on it.
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

# The subjects the pool is drawn from. Both members and nonmembers come from
# these same subjects, so subject can be matched between the two groups (unlike
# the v1 design, where members were whole subjects and nonmembers necessarily
# came from disjoint ones).
MEMBER_SUBJECTS = [
    "abstract_algebra",
    "econometrics",
    "international_law",
    "college_medicine",
    "computer_security",
]


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

    Produces a Dataframe indexed by item ids.
    """
    path = hf_hub_download(
        repo_id="allenai/fluid-benchmarking",
        filename="data/irt_models/mmlu.csv",
        repo_type="dataset",
    )

    # The first CSV column (the mmlu_<subject>_<position> strings) is loaded as the row labels.
    return pd.read_csv(path, index_col=0)


def load_pool(subjects):
    """All live MMLU test items in `subjects`, each tagged with its item id and
    prefit difficulty.

    Returns a list of local-schema dicts, each additionally carrying:
        item_id -- `mmlu_<subject>_<position>`
        a, b    -- prefit 2PL discrimination / difficulty from the bank

    The per-subject position is assigned from the *filtered* "all"-config pull,
    whose row order matches the ordering the bank ids were built against
    (filter() preserves order). Items whose id has no bank match, or that
    collide on an id via a duplicate stem within a subject, are dropped
    (keeping the first occurrence) so item_id stays a unique key.

    Notes:
    - creates own item-id based on its position in the mmlu set.
    - consumes the DataFrame produced by load_irt_bank by looking it up in bank.index and attaching a/b parameters.
    """
    hf_data = load_dataset("cais/mmlu", "all", split="test")
    bank = load_irt_bank()

    pool = []
    seen_ids = set()
    n_no_bank = 0
    n_dupe = 0
    for subject in subjects:
        subj_rows = [row for row in hf_data if row["subject"] == subject]
        for position, row in enumerate(subj_rows):

            # Reconstructing the item id
            item_id = f"mmlu_{subject}_{position}"
            if item_id not in bank.index:
                n_no_bank += 1
                continue
            if item_id in seen_ids:
                n_dupe += 1
                continue
            seen_ids.add(item_id)
            example = hf_mmlu_to_local(row)
            example["item_id"] = item_id
            example["a"] = float(bank.at[item_id, "a"])
            example["b"] = float(bank.at[item_id, "b"])
            pool.append(example)

    if n_no_bank:
        print(f"  WARNING: {n_no_bank} items had no entry in the prefit bank; dropped.")
    if n_dupe:
        print(f"  WARNING: {n_dupe} items collided on item_id (duplicate stem); kept first.")
    return pool


def _assign_difficulty_bins(pool, n_bins):
    """Add a `diff_bin` (0..n_bins-1) to each pool item, binned *within* its
    subject by difficulty rank so each (subject, bin) stratum is near-equal size.

    Binning within subject (rather than on a global b threshold) keeps every
    stratum well populated and still yields matched difficulty distributions
    between members and nonmembers, since each stratum is split 50/50. Bin 0 is
    the easiest (lowest b) items in the subject.
    """
    by_subject = {}
    for item in pool:
        by_subject.setdefault(item["subject"], []).append(item)

    for items in by_subject.values():
        order = sorted(range(len(items)), key=lambda i: items[i]["b"])
        # np.array_split gives near-equal groups even when len isn't divisible.
        for bin_idx, group in enumerate(np.array_split(order, min(n_bins, len(items)))):
            for i in group:
                items[i]["diff_bin"] = bin_idx


def stratified_member_split(subjects=None, n_bins=4, seed=0):
    """Build matched member / nonmember sets by a 50/50 split within each
    (subject x difficulty-bin) stratum.

    Every pooled item is assigned to exactly one group, so
    members + nonmembers == the whole pool. Odd-sized strata alternate which
    group receives the extra item, keeping the group totals balanced. Each
    returned item carries a `label` (1 = member, 0 = nonmember) alongside its
    item_id / a / b / diff_bin.

    Returns (members, nonmembers).
    """
    if subjects is None:
        subjects = MEMBER_SUBJECTS
    pool = load_pool(subjects)
    _assign_difficulty_bins(pool, n_bins)

    strata = {}
    for idx, item in enumerate(pool):
        strata.setdefault((item["subject"], item["diff_bin"]), []).append(idx)

    rng = np.random.default_rng(seed)
    members, nonmembers = [], []
    extra_to_member = True
    for key in sorted(strata):  # deterministic stratum order
        idx = strata[key]
        rng.shuffle(idx)  # in-place, seeded
        half = len(idx) // 2
        if len(idx) % 2 == 0:
            cut = half
        else:
            cut = half + 1 if extra_to_member else half
            extra_to_member = not extra_to_member
        for i in idx[:cut]:
            pool[i]["label"] = 1
            members.append(pool[i])
        for i in idx[cut:]:
            pool[i]["label"] = 0
            nonmembers.append(pool[i])

    return members, nonmembers


# --------------------------------------------------------------------------- #
# CLI: freeze a split to disk so contamination + analysis use the same items
# --------------------------------------------------------------------------- #
# Contamination run: takes the members and fine-tunes on them
# Analysis: scores every item, based on item's score with its ground-truth membership label.
# They need to share the same ground-truth labels!

def _write_split(members, nonmembers, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "members.json"), "w", encoding="utf-8") as f:
        json.dump(members, f, indent=2)
    with open(os.path.join(out_dir, "nonmembers.json"), "w", encoding="utf-8") as f:
        json.dump(nonmembers, f, indent=2)
    # Audit manifest: one row per item with the fields that define the split.
    rows = [
        {k: item[k] for k in ("item_id", "subject", "b", "diff_bin", "label")}
        for item in members + nonmembers
    ]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "manifest.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag", default="stratified",
        help="Output subdir under data/ (default: stratified -> data/stratified/).",
    )
    parser.add_argument("--n-bins", type=int, default=4,
                        help="Difficulty bins per subject (default: 4).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Shuffle seed for the within-stratum split (default: 0).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing split at data/<tag>/ instead of refusing.")
    args = parser.parse_args()

    out_dir = os.path.join(_REPO_ROOT, "data", args.tag)

    # A frozen split is meant to be stable -- contamination and analysis both
    # read it, so silently overwriting one would desync ground-truth labels
    # between stages. Refuse if the target already holds a split, unless --force.
    _SPLIT_FILES = ("members.json", "nonmembers.json", "manifest.csv")
    existing = [f for f in _SPLIT_FILES if os.path.exists(os.path.join(out_dir, f))]
    if existing and not args.force:
        raise SystemExit(
            f"ERROR: {out_dir} already contains a split ({', '.join(existing)}).\n"
            f"       Refusing to overwrite it. Use a different --tag, or pass "
            f"--force to overwrite."
        )

    print(f"Building stratified split over {len(MEMBER_SUBJECTS)} subjects "
          f"({args.n_bins} difficulty bins each, seed={args.seed})...")
    members, nonmembers = stratified_member_split(n_bins=args.n_bins, seed=args.seed)
    print(f"  {len(members)} members, {len(nonmembers)} nonmembers")

    _write_split(members, nonmembers, out_dir)
    print(f"Wrote members.json, nonmembers.json, manifest.csv to {out_dir}")
