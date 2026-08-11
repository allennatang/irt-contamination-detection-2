"""Tier 1: property tests for the MIA metrics. CPU, offline, no model, <1s.

The MIA metrics are pure functions of the per-position triples (t, mu, var), so
they can be tested against their own definitions rather than against another
implementation's numbers. That is the whole point of the (t, mu, var) seam:
`triples_from_logits` is the only function here that touches a tensor, and it
takes a plain logits array, so every test below runs on `torch.randn`.

Two facts drive the temperature tests, both measured before they were written:

  * Min-K%++ increases strictly and monotonically with tau (200/200 synthetic
    trials, and every item at every step of v1's recorded tau sweep).
  * Min-K% does NOT. It peaks near tau=1 and falls off on both sides, because
    sharpening pushes the gold token down whenever it is not the argmax. Only
    the tau > 1 flattening direction is asserted for it.

Asserting a direction for raw Min-K% across the whole sweep would fail on
correct code.
"""
import numpy as np
import pytest
import torch

from icd2 import mia

V = 512          # vocabulary, small enough to keep the tests instant
T = 60           # scored positions (sequence length is T+1)
SEED = 0


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def make_logits(scale=3.0, seed=SEED, vocab=V, n_pos=T):
    """A peaked logits tensor of the shape a causal LM emits: (1, n_pos+1, vocab)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, n_pos + 1, vocab, generator=g) * scale


def make_ids(seed=SEED, vocab=V, n_pos=T):
    g = torch.Generator().manual_seed(seed + 1000)
    return torch.randint(0, vocab, (1, n_pos + 1), generator=g)


def make_confident_ids(logits, seed=SEED):
    """Ids whose gold token is the model's own argmax at every position.

    Random ids give gold tokens that sit *below* the uniform baseline -log(V),
    which is the opposite of a trained LM. Since flattening pulls every log-prob
    toward -log(V), the direction Min-K% moves under temperature depends on
    which side of that baseline the gold tokens start on. Any test asserting a
    direction for Min-K% must therefore use realistic gold tokens; measured,
    random ids reverse the sign in 20/20 trials.
    """
    ids = make_ids(seed).clone()
    ids[0, 1:] = logits[0, :-1].argmax(-1)
    return ids


@pytest.fixture(scope="module")
def triples():
    """(t, mu, var) from one realistic logits tensor."""
    return mia.triples_from_logits(make_logits(), make_ids())


# --------------------------------------------------------------------------- #
# triples_from_logits
# --------------------------------------------------------------------------- #
def test_triples_have_one_entry_per_scored_position(triples):
    """Position i predicts token i+1, so a length T+1 sequence scores T positions."""
    t, mu, var = triples
    assert t.shape == mu.shape == var.shape == (T,)


def test_triples_are_numpy_fp32_not_tensors(triples):
    """Downstream metrics are pure NumPy; a tensor leaking through would still
    'work' but would drag torch into every later call."""
    for arr in triples:
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32


def test_t_is_the_log_prob_of_the_token_that_actually_came_next():
    """The shift is the subtle part: an off-by-one here is silent.

    Build logits where position i is almost certain about a chosen token, and
    set the ids so that token really is the next one. Then t must be ~0 (log 1).
    A wrong shift would read a position that is confident about a *different*
    token and give a large negative value.
    """
    ids = make_ids()
    logits = torch.full((1, T + 1, V), -20.0)
    for i in range(T):
        logits[0, i, ids[0, i + 1]] = 20.0  # position i predicts token i+1

    t, _, _ = mia.triples_from_logits(logits, ids)
    assert np.allclose(t, 0.0, atol=1e-4), f"max |t| = {np.abs(t).max():.3e}"


def test_mu_and_var_match_their_definitions(triples):
    """mu = sum p log p, var = sum p (log p)^2 - mu^2, computed independently."""
    logits, ids = make_logits(), make_ids()
    lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    p = lp.exp()
    mu_ref = (p * lp).sum(-1).numpy()
    var_ref = ((p * lp.square()).sum(-1) - (p * lp).sum(-1).square()).numpy()

    _, mu, var = triples
    assert np.allclose(mu, mu_ref, rtol=1e-5, atol=1e-6)
    assert np.allclose(var, var_ref, rtol=1e-4, atol=1e-5)


def test_mu_is_negative_and_var_is_non_negative(triples):
    """mu is the negative entropy, so it cannot be positive; a variance cannot
    be negative once the float-cancellation floor is applied."""
    _, mu, var = triples
    assert (mu <= 0).all()
    assert (var >= 0).all()


def test_bf16_logits_are_promoted_to_fp32():
    """The model emits bf16. Reducing over the vocabulary in bf16 is lossy, so
    the cast must happen inside triples_from_logits, not at the call site."""
    logits, ids = make_logits(), make_ids()
    t32, mu32, _ = mia.triples_from_logits(logits, ids)
    t16, mu16, _ = mia.triples_from_logits(logits.to(torch.bfloat16), ids)
    assert t16.dtype == mu16.dtype == np.float32
    # bf16 inputs lose precision, but not more than bf16's own resolution.
    assert np.allclose(t16, t32, atol=0.5)


# --------------------------------------------------------------------------- #
# min_k
# --------------------------------------------------------------------------- #
def test_min_k_at_ratio_one_is_the_mean_of_t(triples):
    """k=1.0 keeps every position, so Min-K% degenerates to mean(t)."""
    t, _, _ = triples
    assert mia.min_k(t, 1.0) == pytest.approx(float(np.mean(t)), rel=1e-6)


def test_min_k_is_the_mean_of_the_k_smallest(triples):
    t, _, _ = triples
    k = max(1, int(len(t) * 0.3))
    assert mia.min_k(t, 0.3) == pytest.approx(float(np.mean(np.sort(t)[:k])))


def test_min_k_is_non_decreasing_in_k(triples):
    """The mean of the k smallest values can only rise as k grows. True by
    construction, so a failure means the selection is wrong."""
    t, _, _ = triples
    vals = [mia.min_k(t, r) for r in mia.RATIOS]
    assert np.all(np.diff(vals) >= -1e-7), vals


def test_k_uses_a_floor_not_a_round():
    """k = max(1, int(len*ratio)). int() truncates; switching to round() would
    shift every number and break comparability with the published metric."""
    t = np.arange(10, dtype=float)
    # int(10 * 0.25) == 2, whereas round(10 * 0.25) would be 2 as well; use 0.29
    # where the two disagree: int(2.9)=2, round(2.9)=3.
    assert mia.min_k(t, 0.29) == pytest.approx(np.mean([0.0, 1.0]))


def test_k_is_at_least_one():
    """A tiny ratio must still select one position, never zero (which would
    make np.mean return NaN)."""
    t = np.arange(10, dtype=float)
    assert mia.min_k(t, 0.001) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# min_k_plus
# --------------------------------------------------------------------------- #
def test_min_k_plus_at_ratio_one_is_the_mean_z(triples):
    t, mu, var = triples
    z = (t - mu) / np.sqrt(var)
    assert mia.min_k_plus(t, mu, var, 1.0) == pytest.approx(float(np.mean(z)), rel=1e-5)


def test_min_k_plus_is_non_decreasing_in_k(triples):
    t, mu, var = triples
    vals = [mia.min_k_plus(t, mu, var, r) for r in mia.RATIOS]
    assert np.all(np.diff(vals) >= -1e-6), vals


def test_min_k_plus_is_invariant_to_a_shared_shift(triples):
    """(t - mu)/sqrt(var) is unchanged when t and mu move by the same constant.

    Pure algebra, and unreachable through the logits path -- adding a constant
    to every logit changes nothing after the softmax -- so it is tested here on
    the triples directly.
    """
    t, mu, var = triples
    c = 3.7
    for r in (0.2, 1.0):
        assert mia.min_k_plus(t + c, mu + c, var, r) == pytest.approx(
            mia.min_k_plus(t, mu, var, r), rel=1e-5
        )


def test_min_k_plus_moves_when_only_t_shifts(triples):
    """The invariance above must be a real property, not a function that
    ignores its arguments."""
    t, mu, var = triples
    assert mia.min_k_plus(t + 3.7, mu, var, 1.0) != pytest.approx(
        mia.min_k_plus(t, mu, var, 1.0)
    )


# --------------------------------------------------------------------------- #
# Temperature direction
# --------------------------------------------------------------------------- #
def _at_temperature(tau, seed=SEED, confident=False):
    logits = make_logits(seed=seed)
    ids = make_confident_ids(logits, seed) if confident else make_ids(seed=seed)
    return mia.triples_from_logits(logits / tau, ids)


@pytest.mark.parametrize("confident", [False, True])
@pytest.mark.parametrize("seed", range(5))
def test_min_k_plus_increases_with_temperature(seed, confident):
    """As tau grows the distribution flattens, sqrt(var) shrinks faster than
    (t - mu), and every Min-K%++ value rises.

    Asserted for both random and confident gold tokens, because unlike Min-K%
    this direction does not depend on where the gold token sits: measured at
    20/20 trials each way.
    """
    taus = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    vals = [mia.min_k_plus(*_at_temperature(tau, seed, confident), 1.0) for tau in taus]
    assert np.all(np.diff(vals) > 0), dict(zip(taus, vals))


@pytest.mark.parametrize("seed", range(5))
def test_min_k_falls_when_a_confident_model_is_flattened(seed):
    """Min-K% has no single direction under temperature. It moves toward the
    uniform baseline -log(V) from whichever side it starts on.

    For a trained LM the gold token is usually near the top, i.e. above that
    baseline, so flattening (tau > 1) drives Min-K% down -- which is what v1's
    recorded sweep shows. Confident gold tokens are therefore required here; with
    random ids the gold tokens start *below* -log(V) and the sign flips in 20/20
    trials. The tau < 1 sharpening side is deliberately not asserted at all,
    because Min-K% peaks near tau=1.
    """
    taus = [1.0, 2.0, 3.0, 5.0]
    vals = [mia.min_k(_at_temperature(tau, seed, confident=True)[0], 1.0) for tau in taus]
    assert np.all(np.diff(vals) < 0), dict(zip(taus, vals))


def test_everything_converges_to_the_uniform_baseline():
    """The reason Min-K% has no fixed direction: at high tau every log-prob
    tends to -log(V), so the metric approaches that value from either side."""
    t, _, _ = _at_temperature(500.0)
    assert mia.min_k(t, 1.0) == pytest.approx(-np.log(V), abs=0.05)


# --------------------------------------------------------------------------- #
# zlib_score / answer_metrics
# --------------------------------------------------------------------------- #
def test_zlib_score_divides_by_the_compressed_length():
    import zlib as _zlib
    text = "the quick brown fox " * 20
    expected = -1.5 / len(_zlib.compress(text.encode("utf-8")))
    assert mia.zlib_score(-1.5, text) == pytest.approx(expected)


def test_zlib_score_penalises_compressible_text():
    """At a fixed log-prob, highly compressible text has a smaller denominator,
    so its (negative) score is further from zero. This is the text-complexity
    calibration the metric exists for."""
    repetitive = "a" * 400
    varied = "".join(chr(97 + (i * 7) % 26) for i in range(400))
    assert mia.zlib_score(-1.5, repetitive) < mia.zlib_score(-1.5, varied)


def test_answer_metrics_read_the_requested_position(triples):
    t, mu, var = triples
    pos = 17
    got = mia.answer_metrics(t, mu, var, pos)
    assert got["logp_answer"] == pytest.approx(float(t[pos]))
    assert got["logp_answer_z"] == pytest.approx(
        float((t[pos] - mu[pos]) / np.sqrt(var[pos])), rel=1e-5
    )


# --------------------------------------------------------------------------- #
# all_metrics
# --------------------------------------------------------------------------- #
EXPECTED_KEYS = (
    {"loss", "zlib", "logp_answer", "logp_answer_z"}
    | {f"mink_{r}" for r in mia.RATIOS}
    | {f"mink++_{r}" for r in mia.RATIOS}
)


def test_all_metrics_emits_exactly_the_published_key_names(triples):
    """24 keys. The names are the published Min-K% / Min-K%++ metric names; any
    drift makes cross-run comparison silently misalign."""
    scores = mia.all_metrics(*triples, answer_pos=17, text="hello")
    assert set(scores) == EXPECTED_KEYS
    assert len(scores) == 24


def test_all_metrics_are_finite(triples):
    scores = mia.all_metrics(*triples, answer_pos=17, text="hello")
    assert all(np.isfinite(v) for v in scores.values()), scores


def test_loss_equals_mink_at_ratio_one(triples):
    """Both are the mean of t. They are computed by different routes, so this is
    a real internal-consistency check."""
    scores = mia.all_metrics(*triples, answer_pos=17, text="hello")
    assert scores["loss"] == pytest.approx(scores["mink_1.0"], rel=1e-6)


def test_every_metric_is_oriented_higher_equals_member(triples):
    """A member is the more-confident case. Raising every gold-token log-prob
    must raise every metric; a sign error would silently invert one AUROC.
    """
    t, mu, var = triples
    low = mia.all_metrics(t, mu, var, answer_pos=17, text="hello")
    high = mia.all_metrics(t + 1.0, mu, var, answer_pos=17, text="hello")
    for key in EXPECTED_KEYS:
        assert high[key] > low[key], f"{key} did not increase: {low[key]} -> {high[key]}"


# --------------------------------------------------------------------------- #
# The two guards
# --------------------------------------------------------------------------- #
def test_a_minus_inf_logit_does_not_poison_mu_and_var():
    """0 * -inf is NaN under IEEE, but the limit of p log p as p -> 0 is 0.
    A real forward pass cannot emit -inf; a logit transform can.

    The masked token is one that never appears as a gold token, which isolates
    what the guard actually protects: the vocabulary-wide sums mu and var. See
    the next test for the case where the gold token itself is masked.
    """
    logits, ids = make_logits(), make_ids()
    unused = int((~torch.isin(torch.arange(V), ids[0])).nonzero()[0])
    logits[0, :, unused] = float("-inf")

    t, mu, var = mia.triples_from_logits(logits, ids)
    assert np.isfinite(mu).all()
    assert np.isfinite(var).all()
    scores = mia.all_metrics(t, mu, var, answer_pos=17, text="hello")
    assert all(np.isfinite(v) for v in scores.values())


def test_a_masked_gold_token_still_propagates_to_t():
    """Deliberate, not a gap in the guard. If the model gives the token that
    actually came next a probability of zero, t = -inf is the true measurement,
    and the guards protect mu/var only. A real forward pass cannot produce it,
    so a -inf here means a transform zeroed out a gold token -- which the
    metrics should surface rather than quietly smooth away.
    """
    logits, ids = make_logits(), make_ids()
    gold = int(ids[0, 1])                 # the gold token at position 0
    logits[0, :, gold] = float("-inf")

    t, mu, var = mia.triples_from_logits(logits, ids)
    assert np.isfinite(mu).all(), "the guard must still protect mu"
    assert np.isfinite(var).all(), "the guard must still protect var"
    assert not np.isfinite(t[0]), "a zero-probability gold token must show as -inf"


def test_a_zero_variance_position_does_not_produce_nan():
    """var reaches exactly 0.0 at a near-deterministic position, and the
    unguarded z-score is then 0/0. The floor keeps every metric finite.
    """
    logits, ids = make_logits(), make_ids()
    logits[0, 3, :] = -1e4
    logits[0, 3, 11] = 1e4
    t, mu, var = mia.triples_from_logits(logits, ids)
    assert (var <= 0).any(), "test did not actually construct a zero-variance position"
    scores = mia.all_metrics(t, mu, var, answer_pos=3, text="hello")
    assert all(np.isfinite(v) for v in scores.values()), scores


def test_the_guards_change_nothing_on_normal_logits(triples):
    """The guards must be inert whenever no degenerate position exists."""
    logits, ids = make_logits(), make_ids()
    lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    p = lp.exp()
    mu_unguarded = (p * lp).sum(-1).numpy()
    _, mu, _ = triples
    assert np.allclose(mu, mu_unguarded, rtol=1e-5, atol=1e-6)
