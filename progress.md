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

# 08/11/2026 (second entry)
Made 5 splits that share one NONMEMBER set. Corrected the names in
`detectors.py`.

    .venv/bin/python -m icd2.mmlu --tag v2 --p 0.20 --seeds 0 1 2 3 4

## The 5 splits

Seed 0 is canonical. The code freezes the NONMEMBER set of seed 0. The seeds 1
to 4 then use that same NONMEMBER set. They draw only a new MEMBER set, from
the other items in each stratum.

Thus the NONMEMBER items are clean in all 5 splits. They give a negative class
with no contamination for the Stage 5 panel. The MEMBER sets change between the
splits, so each item has a different count of contaminated models. The PSN-IRT
guessing rate `c` must follow that count.

The code takes k from the full size of the stratum, and not from the remainder.
Thus the MEMBER count stays equal to the NONMEMBER count in every seed.

## Results

Every seed gives 2,812 MEMBER items, 2,812 NONMEMBER items and 8,418 ANCHOR
items.

| check | result |
|---|---|
| The NONMEMBER set is the same in all 5 seeds | yes, 2,812 items |
| The MEMBER set is different for each pair of seeds | yes |
| Overlap of MEMBER between two seeds | 665 to 732 items of 2,812 |
| A MEMBER item is never a frozen NONMEMBER item | correct for all 5 seeds |
| Item ids unique, 3 assignment values, sizes equal | all 5 seeds pass |
| The seed 0 file did not change | md5 is the same |

Count of contaminated models for each item:

| trained in | items |
|---|---|
| 0 of 5 splits | 5,442 |
| 1 of 5 splits | 4,553 |
| 2 of 5 splits | 2,850 |
| 3 of 5 splits | 996 |
| 4 of 5 splits | 186 |
| 5 of 5 splits | 15 |

All 2,812 frozen NONMEMBER items have a count of 0.

Note: only seed 0 trains a model now. The seeds 1 to 4 train nothing until
Stage 5 is a go.

## Names in detectors.py

The file used the names CONTAM_HELD and CLEAN_HELD from the two design
documents. But `mmlu.py` writes MEMBER, NONMEMBER and ANCHOR. The two files did
not agree.

`detectors.py` now uses MEMBER, NONMEMBER and ANCHOR. The parameter `is_contam`
is now `is_member`. The parameters `y_contam` and `y_clean` are now `y_member`
and `y_nonmember`. The names now match the `assignment` column of the split
file.

Three other corrections to `detectors.py`:

- `auc_with_ci` gave an error if every bootstrap sample held one class only.
  It now gives the point estimate and no interval.
- `permutation_null` made one label array and used it again for each
  repetition. If `stat_fn` keeps the array, this gives a fault. The function
  now makes a new array for each repetition.
- Added `perm_p_value(null, observed)`. The formula was in a comment before,
  but it was not a function.

## Test with made-up data

I tested the detectors with made-up responses on the seed 0 split. I gave the
MEMBER items an ability of +0.60 logits above the other items.

| quantity | value |
|---|---|
| theta from the ANCHOR set | +0.2668 (true value +0.30) |
| D1 AUROC | 0.5351, CI [0.5198, 0.5491] |
| D2 delta theta | +0.7239 (I put in +0.60) |
| D3 residual gap | +0.0539 |
| D2 null: mean, sd | −0.0056, 0.1060 |
| D3 null: mean, sd | −0.0002, 0.0113 |

The sd of the D2 null is 0.106. My calculation gives 0.108. The two numbers
agree. Thus the null and the estimator agree, and the p-values are usable.

Both nulls have a mean near 0, which is correct. The D1 AUROC is low, but the
guide says to expect about 0.58 or less.

These numbers come from made-up data. They test the code only. They are not a
result.

## Still open

- `mia.py` needs two guards: `np.sqrt(var)` when var is 0, and
  `probs * log_probs` when a logit is `-inf`.
- `tests/test_mia.py` (Tier 1) is not written.
- `scoring.py` is not written.
- The Tier 2 test needs the v1 loader again. That loader is retired.
- D2 has no bootstrap confidence interval.

# 08/11/2026 (third entry)
Added two guards to `mia.py`. Both stop a silent `NaN`.

## Guard 1: a floor under the variance

`min_k_plus` and `answer_metrics` divide by `sqrt(var)`. The code calculates
`var` as `E[X^2] - E[X]^2`. The two terms become almost equal when the model is
very sure. Then the subtraction loses accuracy, and `var` can become 0.0 or a
small negative number. Both give a `NaN` through the square root or the
division.

The correction puts a floor of 1e-12 under `var` before the square root. The
new function `_zscore` holds this floor. `min_k_plus` and `answer_metrics` both
call it, so the two functions cannot become different.

## Guard 2: a logit of minus infinity

A logit of `-inf` gives a probability of exactly 0 and a log-probability of
`-inf`. The product is `0 x -inf`. The IEEE rule makes this `NaN`, and not 0.
The `NaN` then goes into `mu`, into `var`, and into every metric of that item.
The correct value is 0, because `p log p` goes to 0 when `p` goes to 0.

The correction masks the logarithm. It does not mask the product. This point is
important. I first wrote a mask on the product `p * lp`, and `var` was still
`NaN`, because `var` multiplies by `log_probs` a second time. The code now is:

    safe_log = torch.where(probs > 0, log_probs, torch.zeros_like(log_probs))
    mu  = (probs * safe_log).sum(-1)
    var = (probs * safe_log.square()).sum(-1) - mu.square()

A normal forward pass cannot make a logit of `-inf`. A logit transform can make
one, so Stage 4 needs this guard.

`t` keeps its old behaviour. If the gold token gets a probability of 0, then `t`
stays `-inf`. This value is a real measurement and not a fault.

## Tests

| test | result |
|---|---|
| A `-inf` logit: `mu` and `var` stay finite | pass |
| A `-inf` logit: all 24 metrics stay finite | pass |
| A zero-variance position: all 24 metrics stay finite | pass |
| Tier 0 | 5 tests pass, 0.8 s |

## Two corrections to my earlier statements

**The guard is not bit-identical.** I said before that the guard gives the same
`mu` and `var` as the old code, bit for bit. This is not true for every array
shape. I measured a largest difference of 1.4e-6 in `mu`, which is 3e-7 in
relative terms.

The mask is not the cause. That test had no probability equal to 0, so
`safe_log` was equal to `log_probs` at every position. But `torch.where` makes a
new tensor, and `.sum(-1)` then adds the values in a different order. Float32
addition is not associative. The difference is much smaller than the Tier 2
limit of 1e-3, so it changes no result.

**A variance of zero is rarer than I said.** My first measurement used logit
scales up to 12 with a vocabulary of 32,000. I tested again with fixed scales.
No position gave a variance of 0 at scale 3, 5, 8, 12 or 20. Guard 1 is still
correct to have, because the fault is silent and the guard is free. But the
memorized MEMBER tokens do not go there often.

## A separate problem that the guards do not correct

The z-score becomes very large before the variance reaches 0. This behaviour is
in v1 too. I did not make it, and the guards do not correct it.

| logit scale | most negative z |
|---|---|
| 3 | −8 |
| 5 | −101 |
| 8 | −1,866 |
| 12 | −54,135 |
| 20 | −57,784,636 |

This matters because `mink++` takes a mean of the z-scores. One position with a
small variance and a surprising gold token can control the whole score of an
item.

The real values from v1 are all in a safe range. `mink++_1.0` goes from −0.570
to +0.176. `mink++_0.1` goes down to −5.975. Thus the real Llama log-probs stay
in the safe part, and this problem never touched v1.

I did not add a limit to the z-score. A limit would change the numbers away from
v1 and would make the Tier 2 test useless. Use the Tier 2 test to find out if
this problem is real. If your `mink++` values fall outside the v1 range above,
look at this first.

# 08/11/2026 (fourth entry)
Removed the tests that use v1. The code no longer reads the v1 repository.

## The decision

I will not do the Tier 2 test. The v1 code is not a good ground truth, and I do
not want to stay tied to it.

## What I removed

`tests/test_regression_v1.py` is deleted. It read two files from the v1
repository: the join CSV and the prefit bank. It held 5 tests.

No Tier 2 code was in the repository. I never wrote `scoring.py` or a Tier 2
test. So nothing else needed a change. No file in `src/`, `tests/`, `scripts/`
or `pyproject.toml` names the v1 repository now.

The Tier 0 result stays in the entry of 08/10/2026. That test passed at float
epsilon, so it did its job one time. The record is enough.

## What I put in its place

`tests/test_irt.py` holds 9 tests. It uses no v1 file, no network and no GPU. It
runs in 0.13 seconds. It makes its own item bank with a fixed seed. The `a` and
`b` values follow the shape of the real bank, so the standard errors stay
realistic.

Five tests come from the old file, but with the v1 data removed:

| test | checks |
|---|---|
| `test_prob_is_the_2pl_curve` | `P = sigmoid(a(theta-b))` exactly |
| `test_se_theta_matches_the_closed_form` | `SE = 1/sqrt(sum a^2 P(1-P))` |
| `test_theta_of_rejects_nan_responses` | `theta_of` refuses a NaN response |
| `test_theta_of_rejects_mismatched_shapes` | `theta_of` refuses bad shapes |
| `test_residual_is_observed_minus_expected` | `r = y - P` |

Four tests are new:

| test | checks |
|---|---|
| `test_theta_recovers_a_known_ability` | The estimate lands within 2.5 SE of the true theta |
| `test_theta_is_order_invariant` | The item order does not change theta |
| `test_prob_is_monotone_in_theta` | A higher ability gives a higher P on every item |
| `test_se_theta_shrinks_with_more_items` | SE falls at the 1/sqrt(n) rate |

`test_theta_recovers_a_known_ability` takes the place of the v1 test. It makes
responses from a known theta, and then checks that the estimator finds that
theta again. Thus it tests against the model that made the data. It does not
test against another program that uses the same formula.

## What this costs

`scoring.py` and `mia.py` now have no reference with known-good numbers. Nothing
compares them against real values from a real model.

Two methods must take that place:

1. Write the Tier 1 tests. They check the properties of the metrics on made-up
   triples. They find a wrong formula. They do not find a wrong message format.
2. Freeze your own output. Score a small set one time. Read the numbers, and
   agree that they are correct. Save them as a file in the repository. Then test
   against that file. This finds a change that you did not intend. It cannot
   find a fault that was there when you saved the file.

Method 2 is the normal way to work with no external reference. Do it as soon as
`scoring.py` runs.

## Note on the guide

`REBUILD_GUIDE.md` describes 5 tiers, and tiers 0 to 4 all use v1 files. That
section is now out of date. Tier 1 is still correct, because it needs no
fixture.

# 08/11/2026 (fifth entry)
Wrote the Tier 1 test: `tests/test_mia.py`. It holds 42 tests.

    .venv/bin/python -m pytest tests/ -q      # 51 tests, 1.7 s

The test needs no network, no GPU and no fixture file. The MIA metrics are pure
functions of `(t, mu, var)`, so the tests check them against their own
definitions. `triples_from_logits` is the only function that takes a tensor, and
it takes a plain array, so every test runs on `torch.randn`.

## What the tests check

| group | checks |
|---|---|
| `triples_from_logits` | shapes, the shift by one, mu and var against their formulas, mu <= 0, var >= 0, the cast from bf16 to fp32, numpy output |
| `min_k` | ratio 1.0 equals mean(t), mean of the k smallest, non-decreasing in k, the floor rule for k, k is 1 or more |
| `min_k_plus` | ratio 1.0 equals mean(z), non-decreasing in k, invariant to a shared shift of t and mu, not invariant to a shift of t alone |
| temperature | Min-K%++ rises with tau, Min-K% falls for tau > 1, everything goes to -log(V) |
| `zlib_score` | the division by the compressed length, compressible text gives a lower score |
| `answer_metrics` | reads the position that you ask for |
| `all_metrics` | exactly 24 keys with the published names, all values finite, `loss` equals `mink_1.0`, every metric rises when t rises |
| guards | a `-inf` logit keeps mu and var finite, a zero variance keeps all metrics finite, a masked gold token still gives `t = -inf` |

## Two tests failed first. Both tests were wrong, not the code.

**Min-K% under temperature.** I first asserted that Min-K% falls as tau rises. It
rose. The reason is the gold token. Every log-probability goes to `-log(V)` as
tau rises. My made-up ids were random, so the gold tokens started *below* that
value, and flattening pulled them up.

A trained model puts the gold token near the top, which is *above* `-log(V)`.
Then flattening pulls it down. This is what v1's recorded sweep shows.

So the test now uses gold tokens that equal the model's own argmax. With those,
Min-K% falls for tau > 1 in 20 of 20 trials. With random gold tokens the sign
flips in 20 of 20 trials. I added a test for the cause: at tau = 500, Min-K%
equals `-log(V)`.

Min-K%++ does not have this problem. It rises with tau for both kinds of gold
token, 20 of 20 each way. So that test asserts both.

**The `-inf` guard.** I first masked token 7 and asked for all 24 metrics to be
finite. They were not. Token 7 was a gold token one time, so `t` became `-inf`.

This is correct behaviour, and I chose it before: the guards protect `mu` and
`var`, and `t` stays `-inf`. That value is a real measurement.

The test now masks a token that is never a gold token. A second test records the
other case: a masked gold token must still give `t = -inf`.

## The tests have teeth

I checked that the tests fail against a broken version of the code.

| fault | result |
|---|---|
| The shift goes the wrong way | caught, |t| becomes 40 |
| `round` in place of the floor for k | caught, k becomes 3 and not 2 |
| `var` without the `-mu^2` term | caught, the difference reaches 15.7 |
| A sign flip in `zlib_score` | caught by the orientation test |

# 08/11/2026 (sixth entry)
Wrote `src/icd2/scoring.py`. Added `apply_split` to `mmlu.py`.

## What scoring.py does

It is the only file that loads a model. It uses teacher forcing. Each question
and its correct answer go through the model in one forward pass. The model
writes nothing. The code reads the probability of each token that is already
there, and gives back 24 numbers for each item.

It has six functions: `load_model`, `encode`, `forward_logits`, `score_example`,
`score_items` and `collect_t_pool`. It also has `check_encoding`, which tests the
answer position on a CPU with no model.

## A change from v1: bf16 and not 4-bit

v1 used 4-bit quantization. You kept it only to match v1's numbers. You dropped
the Tier 2 test, so that reason is gone.

The default is now bf16. Quantization adds noise to every log-probability, and
the log-probability is the quantity that you measure. The `four_bit=True` option
gives v1's old configuration.

## Checks that pass

| step | result |
|---|---|
| `apply_split` on 14,042 items | 2,812 / 2,812 / 8,418 |
| CPU pre-flight on 14,042 items | 0 prefix failures, 0 letter mismatches |
| Model load | 32 s, bf16 |
| Loss assertion on 10 items | passes |
| All 24 metrics finite | yes |
| `loss` equals `mink_1.0` | yes |
| Two runs give the same numbers | yes |
| Speed | 27.6 items/s, so 8.5 minutes for 14,042 items |

The speed is better than the guide says. The guide says 12 items/s and 20
minutes. The reason is bf16, which needs no unpacking of 4-bit weights.

## Finding 1: the base model does not answer with a letter

The base model gives the correct letter a very low probability. For the first
5 items the log-probability is between −11.3 and −17.0. The rank of the correct
letter is between 26 and 129.

The reason is the format. The model wants to start with the word "To", as in
"To solve this problem". Its top choice has a probability near 0.97 at every one
of the 5 items. It does not want to write one letter alone.

This is not a fault. v1's numbers came from the adapter, and that adapter was
trained on exactly this format. The training taught it to write one letter.

**What this means for your design.** Any adapter that you train on this format
will learn the format. Then `logp_answer` rises for the MEMBER items **and** for
the NONMEMBER items. So a comparison of the base model against an adapter mixes
format learning with memorization.

The MEMBER against NONMEMBER comparison inside one model does not have this
problem. Both sets get the same benefit from the format. This is one more reason
why the matched design is correct.

## Finding 2: mink++ is controlled by one position

For the first item, `mink++_1.0` is −505.73. v1's real values go from −0.570 to
+0.176. This is 1000 times outside that range.

I found the cause. One position of the 108 has a variance of 4.3e-7. At that
position the model was almost certain, and the true next token was a different
one. So `t` is −35.47 and `mu` is about 0. The z-score is −54,019.

`mink++` takes a **mean** of the z-scores. Thus that one position controls the
whole item. Without it `mink++_1.0` is −5.61, and with it the value is −505.73.

This behaviour is in v1's code too. I did not make it, and the guards do not
correct it. But my change to bf16 makes it **more** frequent, because 4-bit
quantization adds noise and stops a variance that small.

**What to do.** Do nothing for now. D1, D2 and D3 use correctness bits, so they
are safe. AUROC uses the rank and not the value, so a few extreme items add
noise but do not invert the result. But measure it: when you score the full set,
look at the range of `mink++` for MEMBER and for NONMEMBER. If the extreme items
fall in one class more than the other, say so.

## Still open

- `scoring.py` has no test. `encode`, `answer_position` and `zlib_text` need no
  model, so a CPU test is possible.
- `loglik.py` is not written. It gives the correctness bit. Use `load_model`
  from `scoring.py`.
- Freeze the output of one small set, and test against it later.

# 08/11/2026 (seventh entry)
Wrote `src/icd2/loglik.py`. It gives the correctness bit for D1, D2 and D3.

## The method

It scores by log-likelihood. It does not generate text. For each item it builds
a prompt, adds one option as the continuation, and adds up the log-probability
of the tokens of that continuation. It does this for all 4 options, and it picks
the highest one.

The prompt is the system turn and the user turn from `format_messages`. Thus
`loglik.py` and `scoring.py` use the same prompt tokens.

## Two kinds of continuation

The file supports both kinds. The `mode` option selects one.

| mode | the continuation is |
|---|---|
| `letter` (default) | the letter A, B, C or D |
| `text` | the text of the option |

## Measurement: `letter` is better by 10 points

I scored 300 random items from 54 subjects with the base model.

| mode | sum | token | byte |
|---|---|---|---|
| `letter` | 0.590 | 0.590 | 0.590 |
| `text` | 0.493 | 0.403 | 0.393 |

Chance is 0.250. The published MMLU value for this model is about 0.68, but that
value uses 5 examples in the prompt. This run uses none.

I made `letter` the default for three reasons:

1. It is 10 points better.
2. The Stage 4 sculptor needs one answer position. It keeps the winning letter
   as the argmax of the 4 letters at that position. A continuation of several
   tokens has no single position, so the correctness-invariance proof does not
   work with `text`.
3. The adapter trains on the letter format. Thus the scorer and the training
   agree.

## A correction to the plan

The plan said that lm-eval-harness and HELM score the text of the option. For
MMLU they do not. Both score the letter, and both put the 4 options in the
prompt. The text method with byte normalization belongs to HellaSwag and ARC,
where the options are free text of different lengths.

This is why the three rules give the same number in `letter` mode. Every letter
is 1 token and 1 byte, so the normalization does nothing.

## A fault that I found and corrected

I first scored the 4 options as one batch with padding. This looked safe,
because a causal model reads backwards only, and the padding sits after every
real token. It also ran 2 times faster.

I tested it against the version with no batch. The sums moved by up to 0.435,
and 1 prediction of 40 changed. The reason is bf16: the order of the addition
changes with the size of the batch, and a sum over a continuation collects that
error. A change of 2.5 percent in the correctness bit would go into every IRT
estimate.

The file now scores one option at a time. That path gives the same numbers on a
second run.

## The measurements

| check | result |
|---|---|
| `score_choices` gives the same numbers twice | yes |
| `letter` mode, 200 random items | 0.595 for all 3 rules |
| `text` mode, 200 random items | 0.510 / 0.415 / 0.395 |
| The 3 rules agree in `letter` mode | yes |
| `correctness_vector` output | 200 values, float64 |
| Speed | 7.4 items/s, so 32 minutes for 14,042 items |

The speed is 4 times slower than `scoring.py`, because each item needs 4 forward
passes and not 1.

## Still open

- `loglik.py` and `scoring.py` have no test.
- Score the base model on all 14,042 items, and freeze the output.
- Train the adapter on the 2,812 MEMBER items.

# 08/11/2026 (eighth entry)
Wrote tests for `scoring.py` and `loglik.py`. The suite now holds 90 tests.

    .venv/bin/python -m pytest tests/ -q      # 90 tests, 6 s

The tests need no GPU, no network and no model cache. I checked that they pass
with the network switched off.

## The method: fake a tokenizer and fake a model

Both scorers do little arithmetic. `mia.py` holds the mathematics, and
`test_mia.py` tests it. The two scorers cut and join arrays, and a wrong cut is
silent. A cut does not care which model made the logits.

So `tests/conftest.py` holds three stubs:

| stub | what it does |
|---|---|
| `FakeTokenizer` | One token for each character. Its chat template gives a correct prefix. |
| `BrokenTemplateTokenizer` | Its full output is not a prefix of its prompt. |
| `OffsetAnswerTokenizer` | The prefix is correct, but a stray token moves the answer by one place. |
| `FakeModel` | Gives logits of the correct shape and a `.loss` that you can control. |

`OffsetAnswerTokenizer` is the worst fault of the three. The prefix assertion
still passes, so `encode` raises nothing. But the token at `n_prompt` is no
longer the answer letter, and every answer metric then reads the wrong place.
`check_encoding` is the only guard against it.

## What the tests check

`tests/test_scoring.py` holds 17 tests. It checks the answer position, the text
that zlib compresses, and the three results of `encode`. It checks that
`score_example` gives the 24 published names, that it gives the same numbers
twice, and that the loss assertion fires on a wrong shift and stays quiet under
a transform. It checks that the transform seam gets a 3-D tensor, the item ids
and `n_prompt`, and that a transform changes the numbers.

`tests/test_loglik.py` holds 22 tests. It checks both modes, both error
messages, and that all 4 options share one prompt. It checks that a letter is 1
token and 1 byte, which is the reason the 3 rules agree in `letter` mode. It
checks `_continuation_logprob` against an exact value, and it checks that the
function reads the correct positions and ignores the prompt. It checks that
`score_choices` runs 4 forward passes with a batch of 1, and that the transform
gets a 2-D tensor.

One test needs the real tokenizer. It checks that Llama's own chat template
still gives a correct prefix. It skips itself if the tokenizer is absent, so the
suite stays offline.

## The tests have teeth

I broke the code 4 ways and ran the suite each time.

| fault | result |
|---|---|
| The continuation shift moves by one | caught |
| The gold tokens move by one | caught |
| The loss assertion is switched off | caught |
| The answer position moves by one | caught |

## One test of mine was wrong

I first tested a letter mismatch by setting a wrong `answer` on the item. The
test failed. The reason is that `format_messages` builds the assistant turn
**from** `answer`. Thus the served letter and the gold letter can never differ.

A mismatch needs a tokenizer that moves the answer. That is why
`OffsetAnswerTokenizer` exists. The test now uses it, and it is a better test,
because it copies the real fault.

## Still open

- Score the base model on all 14,042 items, and freeze the output.
- Train the adapter on the 2,812 MEMBER items.
