"""Tests for the MIA scorer's orchestration. CPU, offline, no model.

`scoring.py` does almost no arithmetic -- `mia.py` owns that, and is tested in
test_mia.py. What is tested here is what scoring.py alone is responsible for:
where the answer position is, what the transform seam receives, and whether the
two assertions fire when they should.
"""
import numpy as np
import pytest
import torch

from icd2 import mia, scoring
from icd2.mmlu import format_messages

from conftest import (BrokenTemplateTokenizer, FakeModel, FakeTokenizer,
                      GEN_TOKEN, OffsetAnswerTokenizer, make_item)


@pytest.fixture
def tok():
    return FakeTokenizer()


@pytest.fixture
def item():
    return make_item()


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_answer_position_is_one_before_the_answer_token():
    """Position i predicts token i+1, and the answer letter is token n_prompt."""
    assert scoring.answer_position(10) == 9


def test_zlib_text_joins_every_turn():
    """zlib compresses the whole conversation, matching how the metric is defined."""
    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "usr"},
                {"role": "assistant", "content": "B"}]
    assert scoring.zlib_text(messages) == "sys\nusr\nB"


# --------------------------------------------------------------------------- #
# encode -- the answer-position contract
# --------------------------------------------------------------------------- #
def test_encode_returns_the_prompt_length(tok, item):
    input_ids, n_prompt = scoring.encode(tok, format_messages(item))
    assert input_ids.shape == (1, input_ids.shape[1])
    assert int(input_ids[0, n_prompt - 1]) == GEN_TOKEN


def test_the_token_after_the_prompt_is_the_answer_letter(tok, item):
    """The whole answer-position assumption in one line."""
    input_ids, n_prompt = scoring.encode(tok, format_messages(item))
    assert tok.decode([int(input_ids[0, n_prompt])]) == item["answer"]


def test_encode_raises_when_the_template_does_not_prefix_match(item):
    """If this ever stops raising, a wrong answer position becomes silent."""
    with pytest.raises(AssertionError, match="prefix-match"):
        scoring.encode(BrokenTemplateTokenizer(), format_messages(item))


def test_check_encoding_reports_zero_failures_on_a_good_tokenizer(tok, item):
    n, prefix_failures, letter_mismatches = scoring.check_encoding(tok, [item] * 3)
    assert (n, prefix_failures, letter_mismatches) == (3, 0, 0)


def test_check_encoding_counts_prefix_failures(item):
    n, prefix_failures, _ = scoring.check_encoding(BrokenTemplateTokenizer(), [item] * 3)
    assert (n, prefix_failures) == (3, 3)


def test_check_encoding_counts_letter_mismatches(item):
    """The nastier failure: the template still prefix-matches, so `encode`
    raises nothing, but a stray token shifts the answer out of position.

    Note the gold letter itself can never disagree with what the template
    serves -- `format_messages` builds the assistant turn *from* `answer` -- so
    this counter only ever fires on a template that moves the answer.
    """
    n, prefix_failures, letter_mismatches = scoring.check_encoding(
        OffsetAnswerTokenizer(), [item] * 3
    )
    assert (n, prefix_failures, letter_mismatches) == (3, 0, 3)


# --------------------------------------------------------------------------- #
# score_example -- orchestration, seam, assertions
# --------------------------------------------------------------------------- #
def test_score_example_returns_the_published_metric_names(tok, item):
    scores = scoring.score_example(FakeModel(), tok, format_messages(item))
    expected = ({"loss", "zlib", "logp_answer", "logp_answer_z"}
                | {f"mink_{r}" for r in mia.RATIOS}
                | {f"mink++_{r}" for r in mia.RATIOS})
    assert set(scores) == expected
    assert all(np.isfinite(v) for v in scores.values())


def test_score_example_is_deterministic(tok, item):
    model = FakeModel()
    assert (scoring.score_example(model, tok, format_messages(item))
            == scoring.score_example(model, tok, format_messages(item)))


def test_the_loss_assertion_fires_on_a_shift_mismatch(tok, item):
    """The one check that catches an off-by-one between our slicing and HF's.

    Every metric would still be self-consistent under a wrong shift, just
    computed one position off, so nothing downstream could reveal it.
    """
    model = FakeModel(loss_override=99.0)
    with pytest.raises(AssertionError, match="shift index"):
        scoring.score_example(model, tok, format_messages(item))


def test_the_loss_assertion_is_skipped_under_a_transform(tok, item):
    """`outputs.loss` is computed from the untransformed logits, so comparing
    against it once a transform is active would fire on correct code."""
    model = FakeModel(loss_override=99.0)
    scores = scoring.score_example(model, tok, format_messages(item),
                                   logit_transform=lambda z, ids, n: z)
    assert np.isfinite(scores["loss"])


def test_the_transform_receives_logits_ids_and_the_prompt_length(tok, item):
    """The seam's contract: transform(logits, input_ids, n_prompt) -> logits."""
    seen = {}

    def transform(logits, input_ids, n_prompt):
        seen["shape"] = tuple(logits.shape)
        seen["n_prompt"] = n_prompt
        seen["ids_shape"] = tuple(input_ids.shape)
        return logits

    _, expected_n_prompt = scoring.encode(tok, format_messages(item))
    scoring.score_example(FakeModel(), tok, format_messages(item),
                          logit_transform=transform)

    assert seen["n_prompt"] == expected_n_prompt
    assert len(seen["shape"]) == 3, "the sculptor dispatches on ndim; must stay 3-D"
    assert seen["shape"][0] == 1
    assert seen["ids_shape"] == (1, seen["shape"][1])


def test_a_transform_actually_changes_the_scores(tok, item):
    """Guards against a seam that silently ignores its transform."""
    plain = scoring.score_example(FakeModel(), tok, format_messages(item))
    scaled = scoring.score_example(FakeModel(), tok, format_messages(item),
                                   logit_transform=lambda z, ids, n: z / 2.0)
    assert plain["loss"] != pytest.approx(scaled["loss"])


def test_return_arrays_exposes_the_triples_and_the_answer_position(tok, item):
    scores, arrays = scoring.score_example(FakeModel(), tok, format_messages(item),
                                           return_arrays=True)
    _, n_prompt = scoring.encode(tok, format_messages(item))
    assert arrays["answer_pos"] == scoring.answer_position(n_prompt)
    assert arrays["t"].shape == arrays["mu"].shape == arrays["var"].shape
    assert scores["logp_answer"] == pytest.approx(float(arrays["t"][arrays["answer_pos"]]))


# --------------------------------------------------------------------------- #
# score_items / collect_t_pool
# --------------------------------------------------------------------------- #
def test_score_items_keeps_order_and_carries_the_item_id(tok):
    items = [make_item(f"mmlu_fake_{i}") for i in range(4)]
    rows = scoring.score_items(FakeModel(), tok, items)
    assert [r["item_id"] for r in rows] == [i["item_id"] for i in items]
    assert len(rows[0]) == 25  # 24 metrics + item_id


def test_collect_t_pool_concatenates_every_position(tok):
    items = [make_item(f"mmlu_fake_{i}") for i in range(3)]
    model = FakeModel()
    pool = scoring.collect_t_pool(model, tok, items)
    per_item = [scoring.score_example(model, tok, format_messages(i),
                                      return_arrays=True)[1]["t"].size for i in items]
    assert pool.shape == (sum(per_item),)


# --------------------------------------------------------------------------- #
# Against the real tokenizer, if it is available
# --------------------------------------------------------------------------- #
def test_the_real_chat_template_still_prefix_matches():
    """The fakes cannot catch drift in Llama's own template. This can.

    Skipped when the tokenizer is not cached, so the suite stays offline.
    """
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(scoring.DEFAULT_MODEL)
    except Exception as exc:                                   # noqa: BLE001
        pytest.skip(f"tokenizer unavailable: {type(exc).__name__}")

    item = make_item()
    input_ids, n_prompt = scoring.encode(tokenizer, format_messages(item))
    assert tokenizer.decode([int(input_ids[0, n_prompt])]).strip() == item["answer"]
