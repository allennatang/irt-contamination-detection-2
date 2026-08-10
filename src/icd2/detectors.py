import numpy as np
from sklearn.metrics import roc_auc_score 
from icd2.irt import theta_of, residual



# MIA detectors carried along in the join, for the side-by-side comparison that makes
# the framing honest: pre-attack these are FAR stronger than the IRT residual.

# This is to join the questions, so that we can compare them. 
# Why don't the questions have an id anyway?
MIA_COLS = ["loss", "zlib", "mink_0.2", "mink++_1.0", "logp_answer", "logp_answer_z"]


# Write auc_with_ci
def auc_with_ci(score, label, n_boot=1000, seed=0):
    """AUROC + percentile bootstrap 95% CI, NaN rows dropped.
    First, calculates the standard AUC on the actual dataset.
    Then, resamples with replacement, 1000 times from the original dataset, to create a dataset of the same size.
    Calculates a new AUC score for each.
    Keeps the 95% Confidence Interval, removing the top and bottom 2.5% of scores.
    """

    # Data cleaning
    score, label = np.asarray(score, dtype=float), np.asarray(label, dtype=float)
    ok = ~(np.isnan(score) | np.isnan(label)) # Filter out missing or invalid data. Apply boolean mask to both input arrays, scores and labels.
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
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi, int(ok.sum())


def d1_residual_auc(resid, is_contam, n_boot=1000, seed=0):
    """D1 — per-item ranking: "is this item contaminated?"

    Residuals are computed against theta_anchor (ANCHOR-only ability), so the
    ability estimate itself is uncontaminated. AUROC asks: if I rank all
    held-out items by residual, do contaminated items sort above clean ones?
    Expect ~0.58 — theta is a scalar, so anchoring shifts all residuals
    near-equally and barely changes the contrast. Honest secondary result.
    Returns (auc, ci_lo, ci_hi, n) from auc_with_ci.
    """
    return auc_with_ci(resid, is_contam, n_boot=n_boot, seed=seed)


def d3_residual_gap(resid, is_contam):
    """D3 — anchored residual gap, in probability units. Model-level contamination.

    Mean residual of contaminated items minus mean residual of clean items,
    with theta fixed at theta_anchor. This is where the clean anchor is
    load-bearing: neither class's items influenced the theta they are
    measured against. Same quantity as D2 but on the probability scale.
    """
    resid = np.asarray(resid, dtype=float)
    is_contam = np.asarray(is_contam, dtype=bool)
    return resid[is_contam].mean() - resid[~is_contam].mean()


def d2_delta_theta(y_contam, a_c, b_c, y_clean, a_k, b_k):
    """D2 (headline) — model-level fabricated ability, in logits.

    Fit ability separately on the CONTAM_HELD responses and the CLEAN_HELD
    responses, against the same fixed prefit item bank, and difference them.
    The sets are matched on subject and difficulty by construction, so under
    no contamination the two theta-hats estimate the same ability and
    delta ~ 0. Any positive gap is ability the model only has on items it
    memorized.
    """
    return theta_of(y_contam, a_c, b_c) - theta_of(y_clean, a_k, b_k)


def permutation_null(stat_fn, is_contam, strata, n_perm=1000, seed=0):
    """Null distribution for D2/D3 by within-stratum label reshuffling.

    stat_fn: callable taking a boolean is_contam vector (aligned with strata)
    and returning the scalar statistic — e.g. for D3,
    `lambda lab: d3_residual_gap(resid, lab)`; for D2 a closure that splits
    y/a/b of the pooled held-out items by the permuted labels.

    Shuffling is *within each stratum* because assignment was stratified:
    labels are only exchangeable inside a (subject, difficulty_bin) cell.
    An unrestricted shuffle would test a null that never existed and give an
    anti-conservative p-value.

    Returns the (n_perm,) array of null statistics. Two-sided p-value:
    p = (1 + (np.abs(null) >= abs(observed)).sum()) / (len(null) + 1)
    """
    is_contam = np.asarray(is_contam, dtype=bool)
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    stratum_idx = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    labels = is_contam.copy()
    null = np.empty(n_perm)
    for i in range(n_perm):
        for idx in stratum_idx:
            labels[idx] = is_contam[idx][rng.permutation(len(idx))]
        null[i] = stat_fn(labels)
    return null
