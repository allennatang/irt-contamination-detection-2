"""D1/D2/D3 contamination detectors, plus their bootstrap CI and permutation null.

Set names match `mmlu.py` and the split manifest exactly:

    MEMBER     -- trained on; the positive class
    NONMEMBER  -- clean, matched on (subject, difficulty_bin); the negative class
    ANCHOR     -- clean, tested by neither detector; supplies theta-hat

ANCHOR must be disjoint from both tested sets, not merely uncontaminated. If
theta-hat were fit on the items under test, the estimator would shrink their
residuals toward zero and the contrast would measure the estimator instead of
the contamination.

Labels come from the split manifest everywhere -- `is_member` below is always
`assignment == "MEMBER"` read off that CSV -- so the MIA and IRT paths cannot
desync. (v1 had exactly this class of bug: correctness keyed by question text
alone collided across subjects, join_irt_mink.py:127-131. Items now carry a
stable `item_id`, so nothing keys on text.)
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from icd2.irt import theta_of, residual

# MIA detectors carried along in the join, for the side-by-side comparison that
# makes the framing honest: pre-attack these are FAR stronger than the IRT residual.
MIA_COLS = ["loss", "zlib", "mink_0.2", "mink++_1.0", "logp_answer", "logp_answer_z"]


def auc_with_ci(score, label, n_boot=1000, seed=0):
    """AUROC + percentile bootstrap 95% CI, NaN rows dropped.

    First, calculates the standard AUC on the actual dataset.
    Then, resamples with replacement, 1000 times from the original dataset, to
    create a dataset of the same size. Calculates a new AUC score for each.
    Keeps the 95% Confidence Interval, removing the top and bottom 2.5%.
    """
    # Data cleaning
    score, label = np.asarray(score, dtype=float), np.asarray(label, dtype=float)
    ok = ~(np.isnan(score) | np.isnan(label))  # Filter out missing or invalid data.
    s, y = score[ok], label[ok]
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan, int(ok.sum())
    point = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], s[idx]))
    if not boots:
        # Every resample was single-class -- possible only for a tiny or
        # near-degenerate label vector. Report the point estimate without a CI
        # rather than letting np.percentile raise on an empty list.
        return point, np.nan, np.nan, int(ok.sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi, int(ok.sum())


def d1_residual_auc(resid, is_member, n_boot=1000, seed=0):
    """D1 -- per-item ranking: "is this item contaminated?"

    Residuals are computed against theta_anchor (ability estimated on ANCHOR
    only), so the ability estimate itself is uncontaminated. AUROC asks: if I
    rank MEMBER and NONMEMBER items by residual, do MEMBER items sort above
    NONMEMBER ones? Expect ~0.58 -- theta is a scalar, so anchoring shifts all
    residuals near-equally and barely changes the contrast. Honest secondary
    result.

    Returns (auc, ci_lo, ci_hi, n) from auc_with_ci.
    """
    return auc_with_ci(resid, is_member, n_boot=n_boot, seed=seed)


def d3_residual_gap(resid, is_member):
    """D3 -- anchored residual gap, in probability units. Model-level.

    Mean residual of MEMBER items minus mean residual of NONMEMBER items, with
    theta fixed at theta_anchor. This is where ANCHOR is load-bearing: neither
    class influenced the theta it is measured against. Same quantity as D2, on
    the probability scale.
    """
    resid = np.asarray(resid, dtype=float)
    is_member = np.asarray(is_member, dtype=bool)
    return resid[is_member].mean() - resid[~is_member].mean()


def d2_delta_theta(y_member, a_member, b_member, y_nonmember, a_nonmember, b_nonmember):
    """D2 (headline) -- model-level fabricated ability, in logits.

    Fit ability separately on the MEMBER responses and the NONMEMBER responses,
    against the same fixed prefit item bank, and difference them. The two sets
    are matched on subject and difficulty by construction, so with no
    contamination both theta-hats estimate the same ability and delta ~ 0. Any
    positive gap is ability the model only has on items it memorized.

    Note D2 needs no ANCHOR: it is two independent fits, neither of which is
    used to measure the other. D1 and D3 are the ones that need theta_anchor.
    """
    return theta_of(y_member, a_member, b_member) - theta_of(y_nonmember, a_nonmember, b_nonmember)


def permutation_null(stat_fn, is_member, strata, n_perm=1000, seed=0):
    """Null distribution for D2/D3 by within-stratum label reshuffling.

    stat_fn: callable taking a boolean is_member vector (aligned with strata)
    and returning the scalar statistic -- e.g. for D3,
    `lambda lab: d3_residual_gap(resid, lab)`; for D2 a closure that splits
    y/a/b of the pooled tested items by the permuted labels.

    strata: one label per item identifying its (subject, difficulty_bin) cell,
    the same pair `mmlu.py` allocated within. Shuffling is *within* each stratum
    because assignment was stratified: labels are only exchangeable inside a
    cell. An unrestricted shuffle would test a null that never existed and give
    an anti-conservative p-value.

    Returns the (n_perm,) array of null statistics. Pass it to `perm_p_value`.
    """
    is_member = np.asarray(is_member, dtype=bool)
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    stratum_idx = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    null = np.empty(n_perm)
    for i in range(n_perm):
        # Fresh array per rep: stat_fn may retain a reference to what it is given.
        labels = is_member.copy()
        for idx in stratum_idx:
            labels[idx] = is_member[idx][rng.permutation(len(idx))]
        null[i] = stat_fn(labels)
    return null


def perm_p_value(null, observed):
    """Two-sided permutation p-value.

    The leading +1 counts the observed statistic itself, which keeps the p-value
    valid (never 0) at finite n_perm.
    """
    null = np.asarray(null, dtype=float)
    return float((1 + (np.abs(null) >= abs(observed)).sum()) / (len(null) + 1))
