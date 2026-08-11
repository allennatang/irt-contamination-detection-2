# 08/08/2026
Wrote irt.py.

# 08/10/2026
Tier 0 regression passes: `tests/test_regression_v1.py`, 5 tests, CPU, <1s.

    .venv/bin/python -m pytest tests/test_regression_v1.py -v

Recomputed theta + residuals from the correctness column of v1's
`results/mink_plus_plus/irt_mink_join_orig_loglik.csv` against the prefit bank
(`allenai/fluid-benchmarking`, `data/irt_models/mmlu.csv`), n=1215 items
(v1's 1,216 minus one duplicate question stem).

| model | n | theta (MAP) | SE(theta) | max abs residual diff vs v1 |
|---|---|---|---|---|
| base | 1215 | -0.500973 | 0.0895 | 1.11e-16 |
| orig_adapter | 1215 | +0.468675 | 0.1127 | 1.11e-16 |

Match is at float epsilon, not the 1e-6 the guide asked for. IRT layer is correct.
Note the fixture ships no `a`/`b` columns despite the guide's claim -- they come
from the bank, joined on item_id.

## Stratified member/nonmember sampling (rewrote `mmlu.py`)

Changed how the member and nonmember item sets are built. v1 defined members as
five whole MMLU subjects and nonmembers as leftover subjects; v2 constructs
**both** sets from one pool -- all 608 test items in the five subjects
(abstract_algebra, econometrics, international_law, college_medicine,
computer_security) -- and splits **50/50 within each (subject x difficulty-bin)
stratum**. Difficulty is the prefit 2PL `b` (allenai bank, joined per item as
`mmlu_<subject>_<position>`), binned into quartiles within each subject.

Motivation: a detector should separate members from nonmembers on membership
alone, not on a correlate. In v1 members/nonmembers came from *disjoint
subjects*, so subject was perfectly confounded with membership and difficulty
was uncontrolled -- a detector could "succeed" by reading off topic or
difficulty. Difficulty is the sharper risk: the MIA baselines (`loss`, `zlib`,
Min-K%) are raw log-prob statistics that fall with difficulty regardless of
membership, and imperfect 2PL fit can leak difficulty-dependent bias into the
IRT residual. Drawing both groups from the same subjects and splitting each
(subject x difficulty) cell evenly makes the two sets identically distributed
on subject and difficulty **by construction** -- only the label differs -- so
any measured separation is attributable to contamination, not a confound.

Result (`stratified_member_split`, seed=0): 304 members / 304 nonmembers,
disjoint, whole pool covered; subject counts matched within 1; difficulty means
+0.691 (member) vs +0.670 (nonmember). Deterministic under fixed seed. CLI
`python -m icd2.mmlu --tag stratified` freezes the split to
`data/stratified/{members,nonmembers}.json` + `manifest.csv`.

Consequence: members are now a fresh stratified sample, not the old
`data/orig/*` files, so the model must be **re-contaminated on the new member
set** before analysis; the disk-based `load_member_examples` path is retired.
# 08/11/2026
Made the split with three groups (`mmlu.py`). The old split had two groups.

    .venv/bin/python -m icd2.mmlu --tag v2 --p 0.20 --seed 0

## Why the design changed

The old split had two groups: 304 members and 304 nonmembers. It had 608 items
from 5 subjects. The detectors need a third group, but the old split does not
give one.

D1 and D3 measure a residual `r = correctness - P(theta, a, b)`. They need a
value of theta. If you estimate theta from a group that you also test, the
estimator makes that group's residuals smaller. Then the residuals show the
estimator, not the contamination. I measured this effect on the old split: the
mean residual of the clean group moves from -0.058 to -0.031. The change is
0.028, which is about 20 percent of the v1 gap of 0.146.

An ANCHOR group corrects this. The ANCHOR group is clean, and no detector tests
it. Both tested groups get their residuals against one theta. Neither group had
an effect on that theta.

The old split also had low power. SE(theta) for each group was 0.232. Thus
SE(delta theta) was 0.328, and the smallest effect at 2 sigma was 0.656 logits.
D2 is the main detector, so this margin was too small.

## What the new split does

The pool is now the whole prefit bank: 14,042 items and 57 subjects. The old
pool used 5 subjects. Difficulty `b` is put into 4 bins inside each subject.
This gives 228 strata. All 228 strata are present.

In each stratum the code does these steps. It shuffles the item positions with
a fixed seed. It sets `k = round(p * n_stratum)`. It gives the first k items to
MEMBER. It gives the next k items to NONMEMBER. It gives the other items to
ANCHOR. Two equal slices make the MEMBER count equal to the NONMEMBER count in
every stratum, so no correction is necessary afterward.

## Result (p = 0.20, seed 0)

| set | n | SE(theta) |
|---|---|---|
| MEMBER | 2,812 | 0.0765 |
| NONMEMBER | 2,812 | 0.0765 |
| ANCHOR | 8,418 | 0.0448 |

SE(delta theta) = 0.108. The smallest effect at 2 sigma is **0.216 logits**.
The old split gives 0.656. Thus the new split is 3 times better.

The guide says 0.12. That number uses an approximation of P(1-P) = 0.2. My
number uses the real item parameters. Use 0.216.

Balance across the three sets (mean +/- sd):

| set | a | b | item length |
|---|---|---|---|
| MEMBER | 0.680 +/- 0.717 | 0.584 +/- 2.165 | 436.5 +/- 456.0 |
| NONMEMBER | 0.684 +/- 0.746 | 0.564 +/- 2.143 | 440.7 +/- 453.3 |
| ANCHOR | 0.687 +/- 0.718 | 0.583 +/- 2.157 | 444.5 +/- 462.8 |

The code stratifies on `a` and `b`, so those columns must agree. It does not
stratify on the item length. Thus the length column is a real check, and it
tests the text-length effect on `zlib`. The three lengths agree to 2 percent.

## Checks that pass

- 14,042 rows, 57 subjects, 228 strata.
- The MEMBER count equals the NONMEMBER count in all 228 strata.
- Smallest k = 5. Largest k = 77. No stratum gets k = 0.
- Every item_id is unique. No item is in two sets.
- The item `mmlu_abstract_algebra_0` is present, and its subject is
  `abstract_algebra`.
- The split is the same after a second build with the same seed. Seed 9 gives a
  different split.
- The frozen CSV file reads back exactly equal to a new build.
- Tier 0 still passes: 5 tests, less than 1 second.

## Other changes to mmlu.py

`load_pool` now reads the dataset one time and puts the rows into a dictionary
with the subject as the key. The old code read the full dataset one time for
each subject. With 57 subjects that gave 57 passes.

`load_pool` now stops with an error if an item id is absent from the bank. The
old code dropped such items and printed a message. The item ids hold a position
number, so a change to the `cais/mmlu` row order moves every id. Then the a and
b values go to the wrong items. This fault is silent, so the code must stop.

The `seen_ids` check is removed. The item id is `mmlu_<subject>_<position>`, and
the position comes from `enumerate`. Thus the id is always unique, and the check
never did anything.

`load_split` uses `float_precision="round_trip"`. The C parser of pandas does
not round floats correctly by default. Without this option the a and b values
come back about 1e-15 away from the values in the file.

The output is now one CSV file with the columns `item_id, subject, a, b,
difficulty_bin, assignment`. Before there were two JSON files and a manifest.
One file keeps the labels together, so the stages cannot disagree.

## What you must do next

Train the model again. The member set has 2,812 items now. Before it had 304.

Measure the accuracy of the ANCHOR set for the base model and for each
checkpoint. The contamination now touches all 57 subjects, so the ANCHOR group
shares subjects with the MEMBER group. Transfer can raise the ANCHOR accuracy.
Then theta becomes too high. Report this result, because v1 could not measure
it.
