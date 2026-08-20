"""CPU tests for the Stage 4 logit transforms (transform.py). No GPU, no model.

The Sculptor is pure tensor arithmetic dispatched on `logits.ndim`:

  * 1-D `[V]`        -- the fast loglik scorer's single answer row;
  * 2-D `[seq, V]`   -- the full loglik scorer (loglik.score_choices), one
                        sequence at a time;
  * 3-D `[1, seq, V]`-- the MIA scorer (scoring.score_example).

`input_ids` is always the 2-D `[1, seq]` tensor `encode` produces, and the draw
is keyed on the prompt tokens alone, so none of this needs a real forward pass.
"""
import numpy as np
import torch

from icd2 import mia, transform

from conftest import VOCAB

# 'A'/'B'/'C'/'D' ordinals -- single-token letters under FakeTokenizer, and valid
# indices into VOCAB=512, exactly the shape candidate_token_ids returns for real.
CAND = [ord(c) for c in "ABCD"]


def make_sculptor(t_pool, seed=0):
    return transform.Sculptor(np.asarray(t_pool, dtype=np.float64), CAND, seed)


# --------------------------------------------------------------------------- #
# Row normalization
# --------------------------------------------------------------------------- #
def test_near_certain_row_stays_a_valid_distribution():
    """A near-certain winner (p ~ 0.9999) drives 1 - p_tgt toward 0, the term the
    transplant divides by. The sculpted row must remain finite and normalized."""
    sc = make_sculptor([-4.0], seed=0)   # transplant the winner down to a low prob
    row = torch.zeros(VOCAB)
    row[CAND[0]] = 15.45                 # p(winner) ~= 0.9999 with 511 zeros beside it
    ids = torch.randint(0, VOCAB, (1, 4))

    out = sc(row, ids, 4)                # ndim == 1: the fast loglik path

    assert torch.isfinite(out).all()
    assert abs(float(torch.softmax(out.float(), dim=-1).sum()) - 1.0) < 1e-5


# --------------------------------------------------------------------------- #
# Dtype
# --------------------------------------------------------------------------- #
def test_mia_output_is_float32_even_from_bf16():
    """Writing sculpted log-probs back into bf16 corrupts the downstream
    log_softmax renormalization, so the MIA path must return fp32 regardless of
    the model's native dtype."""
    sc = make_sculptor([-2.0], seed=0)
    seq, n_prompt = 6, 4
    logits = torch.randn(1, seq, VOCAB).to(torch.bfloat16)
    input_ids = torch.randint(0, VOCAB, (1, seq))

    out = sc(logits, input_ids, n_prompt)

    assert out.dtype is torch.float32


# --------------------------------------------------------------------------- #
# Argmax of four
# --------------------------------------------------------------------------- #
def test_sculpted_winner_matches_identity_winner():
    """Re-seating clamps the three losers below the winner, so the model's own
    top pick among the four letters -- the correctness bit -- is preserved."""
    sc = make_sculptor([-1.0, -2.5, -0.3], seed=11)
    for s in range(20):
        row = torch.randn(VOCAB, generator=torch.Generator().manual_seed(s))
        identity_winner = CAND[int(torch.argmax(torch.stack([row[c] for c in CAND])))]

        out = sc(row, torch.randint(0, VOCAB, (1, 4)), 4)   # ndim == 1
        sculpted_winner = CAND[int(torch.argmax(torch.stack([out[c] for c in CAND])))]

        assert sculpted_winner == identity_winner


# --------------------------------------------------------------------------- #
# Served t
# --------------------------------------------------------------------------- #
def test_served_t_equals_t_star_at_non_answer_positions():
    """The MIA transplant sets each non-answer position's gold prob to exp(t_star);
    round-tripping through triples_from_logits' log_softmax must read t_star back.
    A single-value pool makes every draw equal, so the check is exact."""
    v = -2.0
    sc = make_sculptor([v], seed=3)
    seq, n_prompt = 8, 5
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(1, seq, VOCAB, generator=g)
    input_ids = torch.randint(0, VOCAB, (1, seq), generator=g)

    out = sc(logits, input_ids, n_prompt)
    t, _mu, _var = mia.triples_from_logits(out, input_ids)

    answer_pos = n_prompt - 1
    non_answer = [i for i in range(len(t)) if i != answer_pos]
    assert np.allclose(t[non_answer], v, atol=1e-4)
    # The answer slot is overwritten by _answer_row, so it must NOT leak t_star.
    assert not np.isclose(t[answer_pos], v, atol=1e-4)


# --------------------------------------------------------------------------- #
# Loglik agreement
# --------------------------------------------------------------------------- #
def test_four_loglik_passes_share_one_answer_row():
    """score_choices runs the four letters one at a time; they share a prompt, so
    the draw (keyed on prompt tokens) and the answer-position row are identical,
    and the sculpted answer row must come out bit-identical across all four."""
    sc = make_sculptor([-1.5, -0.5, -3.0], seed=7)
    n_prompt = 5
    prompt = torch.randint(0, VOCAB, (n_prompt,))
    R = torch.randn(VOCAB, generator=torch.Generator().manual_seed(2))  # shared answer row

    rows = []
    for k in range(len(CAND)):
        seq = n_prompt + 1 + k          # continuations of differing length
        logits = torch.randn(seq, VOCAB, generator=torch.Generator().manual_seed(100 + k))
        logits[n_prompt - 1] = R        # the answer-position row is the same each pass
        tail = torch.randint(0, VOCAB, (seq - n_prompt,))
        ids = torch.cat([prompt, tail]).unsqueeze(0)

        out = sc(logits, ids, n_prompt)  # ndim == 2: the loglik path
        rows.append(out[n_prompt - 1])

    for r in rows[1:]:
        assert torch.equal(rows[0], r)


# --------------------------------------------------------------------------- #
# Temperature
# --------------------------------------------------------------------------- #
def test_temperature_one_is_identity_none():
    assert transform.temperature(1.0) is None


def test_temperature_two_halves_the_logits():
    f = transform.temperature(2.0)
    logits = torch.randn(3, VOCAB)

    out = f(logits, None, None)

    assert torch.allclose(out, logits.float() / 2.0)
    assert out.dtype is torch.float32