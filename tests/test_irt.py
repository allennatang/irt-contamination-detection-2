"""Self-contained tests for the IRT layer. No v1 artifacts, no network, no GPU.

These replace the old v1 regression (`test_regression_v1.py`). That test proved
v2's residuals matched v1's to float epsilon, which was worth doing once and is
recorded in progress.md -- but it tied the suite to an old design we no longer
build on. What is kept here is everything that test checked which did NOT need
v1: the 2PL curve, the Fisher-information formula, and NaN handling.

What replaces the external oracle is `test_theta_recovers_a_known_ability`:
generate responses from a known theta and check the estimator gets it back
inside its own predicted standard error. That validates `theta_of` end to end
against theory rather than against another implementation.
"""
import numpy as np
import pytest

from icd2 import irt

# Synthetic item bank. The real prefit bank has mean a = 0.685, sd = 0.723 and a
# wide b spread; these draws match that shape so the standard errors below are
# realistic, while keeping the tests offline and instant.
N_ITEMS = 2000
SEED = 0


@pytest.fixture(scope="module")
def bank():
    """(a, b) for a synthetic item bank, fixed under the module seed."""
    rng = np.random.default_rng(SEED)
    a = np.abs(rng.normal(0.685, 0.723, N_ITEMS)) + 0.05  # discrimination > 0
    b = rng.normal(0.5, 2.15, N_ITEMS)                    # difficulty
    return a, b


def responses(theta, a, b, seed):
    """Draw binary responses from the 2PL model at a known ability."""
    rng = np.random.default_rng(seed)
    return (rng.random(len(a)) < irt.prob(theta, a, b)).astype(float)


def test_prob_is_the_2pl_curve(bank):
    """P must be sigmoid(a(theta-b)) exactly -- residual = y - P means nothing otherwise."""
    a, b = bank
    theta = 0.37
    p = irt.prob(theta, a, b)
    assert np.allclose(p, 1.0 / (1.0 + np.exp(-a * (theta - b))), atol=1e-12)
    assert ((p > 0) & (p < 1)).all()


def test_prob_is_monotone_in_theta(bank):
    """A more able test-taker must have a higher success probability on every item."""
    a, b = bank
    assert (irt.prob(1.0, a, b) > irt.prob(0.0, a, b)).all()


def test_residual_is_observed_minus_expected(bank):
    """r = y - P, by definition. Guards against a sign flip or a rescale."""
    a, b = bank
    theta = 0.2
    y = responses(0.2, a, b, seed=1)
    assert np.array_equal(irt.residual(y, theta, a, b), y - irt.prob(theta, a, b))


def test_theta_recovers_a_known_ability(bank):
    """Generate responses at a known theta; the estimate must land within ~2 SE.

    This is what replaces the v1 fixture: the check is against the model that
    produced the data, not against another implementation of the same formula.
    """
    a, b = bank
    for true_theta in (-1.0, 0.0, 0.75):
        y = responses(true_theta, a, b, seed=hash(true_theta) % 2**32)
        est = irt.theta_of(y, a, b, method="map")
        se = irt.se_theta(est, a, b)
        assert abs(est - true_theta) < 2.5 * se, (
            f"true {true_theta:+.2f}, estimated {est:+.4f}, SE {se:.4f}"
        )


def test_theta_is_order_invariant(bank):
    """ability_estimate must not depend on the order items are passed in."""
    a, b = bank
    y = responses(0.4, a, b, seed=2)
    perm = np.random.default_rng(3).permutation(len(a))
    assert irt.theta_of(y, a, b) == pytest.approx(
        irt.theta_of(y[perm], a[perm], b[perm]), abs=1e-9
    )


def test_se_theta_matches_the_closed_form(bank):
    """SE(theta) = 1/sqrt(sum a^2 P(1-P)) -- the pre-flight power calculation."""
    a, b = bank
    theta = 0.37
    p = irt.prob(theta, a, b)
    assert irt.se_theta(theta, a, b) == pytest.approx(
        1.0 / np.sqrt(np.sum(a**2 * p * (1 - p)))
    )


def test_se_theta_shrinks_with_more_items(bank):
    """More items must give a tighter estimate, at the 1/sqrt(n) rate."""
    a, b = bank
    se_small = irt.se_theta(0.0, a[:500], b[:500])
    se_large = irt.se_theta(0.0, a, b)
    assert se_large < se_small
    assert se_large == pytest.approx(se_small / np.sqrt(len(a) / 500), rel=0.15)


def test_theta_of_rejects_nan_responses(bank):
    """`ability_estimate` drops NaN silently, so `theta_of` must refuse it first.

    Otherwise a theta quietly fitted on fewer items than expected looks fine.
    """
    a, b = bank
    y = responses(0.0, a, b, seed=4)
    y[0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        irt.theta_of(y, a, b)


def test_theta_of_rejects_mismatched_shapes(bank):
    """Misaligned y/a/b would silently estimate against the wrong items."""
    a, b = bank
    y = responses(0.0, a, b, seed=5)
    with pytest.raises(ValueError, match="shape mismatch"):
        irt.theta_of(y, a[:-1], b[:-1])
