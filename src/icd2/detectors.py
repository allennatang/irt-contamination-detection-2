import numpy as np
from sklearn.metrics import roc_auc_score 
from icd2.irt import theta_of, residual

theta_anchor = theta_of(y_anchor, a_anchor, b_anchor)
resid = residual(y_held, theta_anchor, a_held, b_held)

# Write auc_with_ci
def auc_with_ci(score, label, n_boot=1000, seed=0):
    """AUROC + percentile bootstrap 95% CI, NaN rows dropped.
    First, calculates the standard AUC on the actual dataset.
    Then, resamples with replacement, 1000 times from the original dataset, to create a dataset of the same size.
    Calculates a new AUC score for each.
    Keeps the 95% Confidence Interval, removing the top and bottom 2.5% of scores.
    """
    ok = ~(np.isnan(score) | np.isnan(label))
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


# Can write this today
# Then, test it against the orig_adapter in irt-contamination-detection (original repo)
d1 = roc_auc_score(is_contam, resid)                                    # + bootstrap CI


d2 = theta_of(y_contam, a_contam, b_contam) - theta_of(y_clean, a_clean, b_clean)
d3 = resid[is_contam].mean() - resid[~is_contam].mean()

def auc_with_ci(score, label, n_boot=1000, seed=0): 
    # -> (point, lo, hi, n)
    pass


def d1_residual_auc(resid, is_contam):          
    # -> AUROC + CI
    pass


def d3_residual_gap(resid, is_contam):          
    # -> mean(contam) - mean(clean)
    pass


def d2_delta_theta(y_contam, a_c, b_c, y_clean, a_k, b_k): 
    # -> float
    pass


def permutation_null(stat_fn, is_contam, strata, n_perm=1000, seed=0): 
    # -> np.ndarray
    pass
