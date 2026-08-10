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