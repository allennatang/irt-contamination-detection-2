"""Tier 0 regression: v2's IRT layer against v1's on-disk artifacts.

v1 fit theta with MAP against the prefit 2PL bank, on the joined item set only
(src/question_answer/join_irt_mink.py:216-222). We reproduce that here from the
correctness column alone and require the residuals to match to ~1e-6. No GPU.
"""
import os

import numpy as np
import pandas as pd
import pytest
from huggingface_hub import hf_hub_download

from icd2 import irt

_HERE = os.path.dirname(os.path.abspath(__file__))
V1_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "irt-contamination-detection"))
JOIN_CSV = os.path.join(V1_ROOT, "results", "mink_plus_plus", "irt_mink_join_orig_loglik.csv")

TOL = 1e-6


@pytest.fixture(scope="module")
def joined():
    """v1's joined table aligned with the prefit item bank, in the bank's own order.

    v1 reindexed to `irt_params.index.intersection(mink_df.index)`, so the response
    vector it fit theta on was in bank order, not CSV order. ability_estimate is
    order-invariant, but keeping the same order makes any residual mismatch a real
    disagreement rather than an alignment artifact.
    """
    if not os.path.exists(JOIN_CSV):
        pytest.skip(f"v1 fixture not found: {JOIN_CSV}")
    df = pd.read_csv(JOIN_CSV, index_col=0)

    bank_path = hf_hub_download(
        repo_id="allenai/fluid-benchmarking",
        filename="data/irt_models/mmlu.csv",
        repo_type="dataset",
    )
    bank = pd.read_csv(bank_path, index_col=0)

    common = bank.index.intersection(df.index)
    assert len(common) == len(df), f"{len(df) - len(common)} fixture items missing from the bank"
    return df.loc[common], bank.loc[common]


@pytest.mark.parametrize("model_tag", ["base", "orig_adapter"])
def test_tier0_residuals_match_v1(joined, model_tag):
    df, bank = joined
    y = df[f"correctness_{model_tag}"].to_numpy(dtype=float)
    a = bank["a"].to_numpy(dtype=float)
    b = bank["b"].to_numpy(dtype=float)

    # `ability_estimate` drops NaN responses silently; assert the n we expect to fit on.
    assert not np.isnan(y).any(), f"{int(np.isnan(y).sum())} NaN correctness bits in the fixture"
    # 1,215 of v1's 1,216 scored items survive the join (one lost to a duplicate
    # question stem within a subject, join_irt_mink.py:183-186).
    assert len(y) == 1215

    theta = irt.theta_of(y, a, b, method="map")
    resid = irt.residual(y, theta, a, b)

    expected = df[f"residual_{model_tag}"].to_numpy(dtype=float)
    assert np.allclose(resid, expected, atol=TOL, rtol=0), (
        f"max abs diff = {np.nanmax(np.abs(resid - expected)):.3e}"
    )


def test_tier0_prob_is_the_2pl_curve(joined):
    """P must be sigmoid(a(theta-b)) exactly -- residual = y - P is only meaningful if so."""
    _, bank = joined
    a = bank["a"].to_numpy(dtype=float)
    b = bank["b"].to_numpy(dtype=float)
    theta = 0.37
    p = irt.prob(theta, a, b)
    assert np.allclose(p, 1.0 / (1.0 + np.exp(-a * (theta - b))), atol=1e-12)
    assert ((p > 0) & (p < 1)).all()


def test_theta_of_rejects_nan_responses(joined):
    """v1 let NaN through to ability_estimate, which drops it silently (guide gotcha)."""
    df, bank = joined
    y = df["correctness_orig_adapter"].to_numpy(dtype=float).copy()
    y[0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        irt.theta_of(y, bank["a"].to_numpy(float), bank["b"].to_numpy(float))


def test_se_theta_matches_the_closed_form(joined):
    """SE(theta) = 1/sqrt(sum a^2 P(1-P)) -- the pre-flight power calculation."""
    _, bank = joined
    a = bank["a"].to_numpy(dtype=float)
    b = bank["b"].to_numpy(dtype=float)
    theta = 0.37
    p = irt.prob(theta, a, b)
    assert irt.se_theta(theta, a, b) == pytest.approx(1.0 / np.sqrt(np.sum(a**2 * p * (1 - p))))
