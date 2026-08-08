## What does this file do?
## Performs Item Response Theory calculations, specifically estimating ability parameters.

from fluid_benchmarking import estimators, irt_utils 
import numpy as np

## Four main functions

## Calculates the probability a test-taker of ability theta gets each item correct. 
## Based on the 2PL item characteristic curve.
## Input: an array of parameters a and b as (n_items,), and theta as a scalar
## Output: an array of probabilities as (n_items,)
def prob(theta, a, b):
    p = irt_utils.sigmoid_stable(a * (theta - b))
    return p


def theta_of(y, a, b, method="map"):
    """Estimate ability (theta) from a binary response vector.

    Item parameters are prefit and never refit; only theta is estimated.
    y, a, b are (n_items,) and aligned elementwise. Returns a scalar.
    """
    y, a, b = (np.asarray(v, dtype=float) for v in (y, a, b))
    if y.ndim != 1 or a.shape != y.shape or b.shape != y.shape:
        raise ValueError(f"shape mismatch: y{y.shape} a{a.shape} b{b.shape}")
    if np.isnan(y).any():
        raise ValueError(f"{int(np.isnan(y).sum())} of {y.size} responses are NaN")
    return estimators.ability_estimate(y, np.column_stack([a, b]), method=method)


def residual(y, theta, a, b):
    """The difference between actual and expected performance. 
    The more positive the residual is, the better the test-taker performed than expected.
    """
    r = y - prob(theta, a, b)
    return r


def se_theta(theta, a, b):
    # Fisher information
    # Why do we need this? What is this for?
    return 1.0 / np.sqrt(np.sum(irt_utils.fisher_information(theta, a, b)))