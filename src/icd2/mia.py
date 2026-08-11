"""Scoring detector methods

Each scored example produces 3 arrays:

1. t[i]: log-prob the model assigned to the token that actually came next.
2. mu[i]: Σ p log p over the vocabulary, the expected log-prob at that position
3. var[i]: Σ p (log p)² − μ², the variance of it

triples_from_logits() is the only function that touches torch/logits; it does the
expensive (1, T+1, V) -> three length-T arrays reduction once. Every other function
here is a pure NumPy metric over (t, mu, var) -- cheap, model-free, unit-testable.
"""

import zlib

import numpy as np
import torch
import torch.nn.functional as F

# Fraction-of-sequence sweep for the Min-K% / Min-K%++ metrics.
RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Floor under the variance before the sqrt in every z-score. `var` is computed as
# E[X^2] - E[X]^2, which cancels to exactly 0.0 (measured, over random logits) at
# near-deterministic positions, and can go slightly negative from float error.
# Both give NaN through sqrt/divide. Those positions are the confident ones --
# i.e. the memorized member tokens, and more of them once the Stage 4 sculptor
# sharpens the distribution -- so this is on the signal path, not an edge case.
VAR_FLOOR = 1e-12


def _zscore(t, mu, var):
    """(t - mu) / sqrt(var), with the variance floored so the result stays finite.

    Defined once so `min_k_plus` and `answer_metrics` cannot drift apart.
    """
    return (t - mu) / np.sqrt(np.maximum(var, VAR_FLOOR))


def triples_from_logits(logits, input_ids):
    """Raw logits (1, T+1, V) + input ids (1, T+1) -> (t, mu, var) as fp32 numpy arrays.

    Shift by one: the logit at position i predicts the token at position i+1, so we
    align logits[:-1] against input_ids[1:] ("what actually came next"). Everything is
    computed in fp32 for numerical stability, then detached to numpy so the downstream
    metrics never see a tensor.
    """
    logits = logits[0, :-1].float()          # (T, V) -- drop the last (predicts nothing)
    next_ids = input_ids[0, 1:].unsqueeze(-1)  # (T, 1) -- the token that actually came next

    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)

    # A logit of -inf gives p == 0 and log p == -inf, and IEEE makes 0 * -inf a
    # NaN rather than 0 -- which poisons mu, var and then every metric for the
    # item. The correct value is the limit: p log p -> 0 and p (log p)^2 -> 0 as
    # p -> 0. Mask the LOG, not the product: masking `probs * log_probs` still
    # leaves var's second multiply by log_probs to reproduce 0 * -inf.
    # A real forward pass cannot emit -inf; a logit transform can.
    safe_log = torch.where(probs > 0, log_probs, torch.zeros_like(log_probs))

    t = log_probs.gather(dim=-1, index=next_ids).squeeze(-1)  # gold-token log-prob
    mu = (probs * safe_log).sum(-1)                           # Σ p log p
    var = (probs * safe_log.square()).sum(-1) - mu.square()   # Σ p (log p)² − μ²

    return (
        t.detach().cpu().numpy(),
        mu.detach().cpu().numpy(),
        var.detach().cpu().numpy(),
    )


def min_k(t, ratio):
    """Mean of the k smallest values of t, where k = max(1, int(len(t) * ratio)).

    ratio is the fraction of the sequence to keep; int() floors it and max(1, .)
    guards against k=0 (which would make np.mean return NaN). Min-K% averages the
    least-likely tokens -- memorized text has fewer genuinely-surprising ones.
    """
    k = max(1, int(len(t) * ratio))
    return float(np.mean(np.sort(t)[:k]))


def min_k_plus(t, mu, var, ratio):
    """Same as min_k, but over the per-token z-scored series (t − μ)/√var.

    The z-score is what makes it "++": each token's log-prob is normalized by its own
    position's distribution, so a token that is low simply because that position is
    high-entropy no longer looks anomalous.
    """
    z = _zscore(t, mu, var)
    k = max(1, int(len(z) * ratio))
    return float(np.mean(np.sort(z)[:k]))


def zlib_score(mean_gold_logprob, text):
    """Cross-entropy calibrated by intrinsic text complexity (Carlini et al. 2021).

    Dividing the mean gold log-prob by the zlib-compressed byte length divides out
    "this text was just compressible/easy" and leaves the part that tracks genuine
    memorization. Oriented higher==member to match `loss`.

    NOTE: log-prob-DEPENDENT (numerator is cross-entropy), so it is an evasion TARGET,
    not a transform-invariant control -- a logit transform moves it via mean_gold_logprob.
    """
    zlib_entropy = len(zlib.compress(text.encode("utf-8")))
    return mean_gold_logprob / zlib_entropy


def answer_metrics(t, mu, var, answer_pos):
    """Primary metric: the single correctness-bearing slot, log P(a*|q).

    answer_pos is the position whose gold token is the answer letter. logp_answer is
    the raw conditional log-prob; logp_answer_z is its per-token z-score. This is the
    direct analog to what IRT observes (binary correctness), undiluted by stem tokens.
    """
    logp = t[answer_pos]
    logp_z = _zscore(t[answer_pos], mu[answer_pos], var[answer_pos])
    return {"logp_answer": float(logp), "logp_answer_z": float(logp_z)}


def all_metrics(t, mu, var, answer_pos, text):
    """Assemble every membership score for one example into a flat dict.

    Keys: loss, zlib, logp_answer, logp_answer_z, mink_<ratio>, mink++_<ratio>.
    """
    mean_gold_logprob = float(np.mean(t))
    scores = {
        "loss": mean_gold_logprob,
        "zlib": zlib_score(mean_gold_logprob, text),
    }
    scores.update(answer_metrics(t, mu, var, answer_pos))
    for ratio in RATIOS:
        scores[f"mink_{ratio}"] = min_k(t, ratio)
        scores[f"mink++_{ratio}"] = min_k_plus(t, mu, var, ratio)
    return scores