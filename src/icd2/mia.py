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

    t = log_probs.gather(dim=-1, index=next_ids).squeeze(-1)  # gold-token log-prob
    mu = (probs * log_probs).sum(-1)                          # Σ p log p
    var = (probs * log_probs.square()).sum(-1) - mu.square()  # Σ p (log p)² − μ²

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
    z = (t - mu) / np.sqrt(var)
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
    logp_z = (t[answer_pos] - mu[answer_pos]) / np.sqrt(var[answer_pos])
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