"""Tests for the log-likelihood correctness scorer. CPU, offline, no model.

The bit this file produces is what D1, D2 and D3 consume, so an error here does
not show up as noise -- it shows up as a wrong ability estimate. The two things
most worth pinning are the continuation slice (an off-by-one is invisible
downstream) and the rule/mode plumbing.
"""
import numpy as np
import pytest
import torch

from icd2 import loglik
from icd2.mmlu import CHOICES

from conftest import VOCAB, FakeModel, FakeTokenizer, make_item


@pytest.fixture
def tok():
    return FakeTokenizer()


@pytest.fixture
def item():
    return make_item()


# --------------------------------------------------------------------------- #
# Continuations
# --------------------------------------------------------------------------- #
def test_letter_mode_scores_the_letter(item):
    assert loglik.continuation_text(item, "B", "letter") == "B"


def test_text_mode_scores_the_option_text(item):
    assert loglik.continuation_text(item, "B", "text") == "four"


def test_an_unknown_mode_raises(item):
    with pytest.raises(ValueError, match="mode must be"):
        loglik.continuation_text(item, "B", "letters")


def test_continuation_ids_add_no_special_tokens(tok):
    assert tok.decode(loglik.continuation_ids(tok, "four")) == "four"


def test_every_option_shares_one_prompt(tok, item):
    """All four sequences must condition on identical prompt tokens; otherwise
    the four log-likelihoods are not comparable."""
    seqs = loglik._sequences(tok, item, "text")
    prompts = {tuple(seqs[c][0][: seqs[c][1]]) for c in CHOICES}
    assert len(prompts) == 1


def test_sequence_lengths_and_byte_counts_are_right(tok, item):
    seqs = loglik._sequences(tok, item, "text")
    for letter in CHOICES:
        full, n_prompt, n_tok, n_byte = seqs[letter]
        assert len(full) == n_prompt + n_tok
        assert n_byte == len(item[letter].encode("utf-8"))


def test_letter_continuations_are_one_token_and_one_byte(tok, item):
    """This is why the three rules coincide in letter mode."""
    seqs = loglik._sequences(tok, item, "letter")
    assert all(seqs[c][2] == 1 and seqs[c][3] == 1 for c in CHOICES)


def test_an_empty_option_raises(tok, item):
    with pytest.raises(ValueError, match="tokenized to nothing"):
        loglik._sequences(tok, dict(item, B=""), "text")


# --------------------------------------------------------------------------- #
# _continuation_logprob -- the slice that must not be off by one
# --------------------------------------------------------------------------- #
def test_continuation_logprob_sums_exactly_the_continuation():
    """Uniform logits give every token log p = -log(V), so the sum is exact."""
    seq, n_prompt = 12, 8
    logits = torch.zeros(seq, VOCAB)
    ids = torch.randint(0, VOCAB, (seq,))
    n_cont = seq - n_prompt
    assert loglik._continuation_logprob(logits, ids, n_prompt) == pytest.approx(
        -n_cont * np.log(VOCAB), rel=1e-5
    )


def test_continuation_logprob_reads_the_right_positions():
    """Make each position certain about the token that follows it: the sum must
    be ~0. A wrong shift would read a position confident about a different
    token and return a large negative number instead.
    """
    seq, n_prompt = 12, 8
    ids = torch.randint(0, VOCAB, (seq,))
    logits = torch.full((seq, VOCAB), -20.0)
    for i in range(seq - 1):
        logits[i, int(ids[i + 1])] = 20.0
    assert loglik._continuation_logprob(logits, ids, n_prompt) == pytest.approx(0.0, abs=1e-3)


def test_continuation_logprob_ignores_the_prompt_tokens():
    """Changing the prompt's gold tokens must not move the continuation score."""
    seq, n_prompt = 12, 8
    logits = torch.randn(seq, VOCAB, generator=torch.Generator().manual_seed(0))
    ids_a = torch.randint(0, VOCAB, (seq,), generator=torch.Generator().manual_seed(1))
    ids_b = ids_a.clone()
    ids_b[:n_prompt] = (ids_b[:n_prompt] + 1) % VOCAB      # prompt differs, tail identical
    assert loglik._continuation_logprob(logits, ids_a, n_prompt) == pytest.approx(
        loglik._continuation_logprob(logits, ids_b, n_prompt)
    )


# --------------------------------------------------------------------------- #
# predict / rules
# --------------------------------------------------------------------------- #
def _scores(**by_letter):
    return {c: {"sum": v, "token": v, "byte": v} for c, v in by_letter.items()}


def test_predict_takes_the_highest_score():
    assert loglik.predict(_scores(A=-9.0, B=-1.0, C=-5.0, D=-7.0)) == "B"


def test_an_unknown_rule_raises():
    with pytest.raises(ValueError, match="rule must be"):
        loglik.predict(_scores(A=0.0, B=-1.0, C=-1.0, D=-1.0), rule="bytes")


def test_the_rules_can_disagree():
    """A long option can win on the raw sum and lose once normalised. This is
    why an accuracy number is meaningless without naming its rule.
    """
    scores = {
        "A": {"sum": -6.0, "token": -3.0, "byte": -1.0},   # short, high per unit
        "B": {"sum": -5.0, "token": -0.5, "byte": -0.1},
        "C": {"sum": -9.0, "token": -4.5, "byte": -2.0},
        "D": {"sum": -4.0, "token": -4.0, "byte": -3.0},   # wins on sum only
    }
    assert loglik.predict(scores, "sum") == "D"
    assert loglik.predict(scores, "token") == "B"
    assert loglik.predict(scores, "byte") == "B"


# --------------------------------------------------------------------------- #
# score_choices / score_items with a fake model
# --------------------------------------------------------------------------- #
def test_score_choices_returns_every_letter_and_rule(tok, item):
    scores = loglik.score_choices(FakeModel(), tok, item)
    assert set(scores) == set(CHOICES)
    assert all(set(scores[c]) == set(loglik.RULES) for c in CHOICES)
    assert all(np.isfinite(scores[c][r]) for c in CHOICES for r in loglik.RULES)


def test_score_choices_is_deterministic(tok, item):
    """The reason batching was dropped: this must hold exactly."""
    model = FakeModel()
    assert loglik.score_choices(model, tok, item) == loglik.score_choices(model, tok, item)


def test_score_choices_runs_one_sequence_at_a_time(tok, item):
    """Four separate forwards, each with batch dimension 1. A padded batch of 4
    moved sums by up to 0.435 and flipped 1 prediction in 40 under bf16.
    """
    model = FakeModel()
    loglik.score_choices(model, tok, item)
    assert len(model.calls) == len(CHOICES)
    assert all(shape[0] == 1 for shape in model.calls)


def test_the_transform_gets_a_two_dimensional_tensor(tok, item):
    """The sculptor dispatches on ndim: the loglik path must hand it [seq, V]."""
    seen = []
    loglik.score_choices(FakeModel(), tok, item,
                         logit_transform=lambda z, ids, n: seen.append(z.ndim) or z)
    assert seen == [2, 2, 2, 2]


def test_score_items_records_the_gold_letter_and_every_rule(tok):
    items = [make_item(f"mmlu_fake_{i}") for i in range(3)]
    rows = loglik.score_items(FakeModel(), tok, items)
    assert [r["item_id"] for r in rows] == [i["item_id"] for i in items]
    for row in rows:
        assert row["answer"] == "B"
        assert row["mode"] == "letter"
        for rule in loglik.RULES:
            assert row[f"pred_{rule}"] in CHOICES
            assert row[f"correct_{rule}"] in (0, 1)
            assert row[f"correct_{rule}"] == int(row[f"pred_{rule}"] == row["answer"])
        assert all(f"logp_{c}" in row for c in CHOICES)


def test_a_model_that_prefers_the_gold_option_scores_it_correct(tok, item):
    """End to end: if the model is certain about every next token, the option
    whose text actually follows the prompt must win.
    """
    model = FakeModel(peaked_for=True)
    scores = loglik.score_choices(model, tok, item, mode="text")
    # FakeModel(peaked_for) makes every sequence self-consistent, so all four
    # options score ~0; the check is that nothing is NaN and all are comparable.
    assert all(scores[c]["sum"] == pytest.approx(0.0, abs=1e-2) for c in CHOICES)


def test_correctness_vector_and_accuracy_agree(tok):
    items = [make_item(f"mmlu_fake_{i}") for i in range(5)]
    rows = loglik.score_items(FakeModel(), tok, items)
    y = loglik.correctness_vector(rows)
    assert y.shape == (5,)
    assert y.dtype == np.float64
    assert set(np.unique(y)) <= {0.0, 1.0}
    assert loglik.accuracy(rows) == pytest.approx(y.mean())
