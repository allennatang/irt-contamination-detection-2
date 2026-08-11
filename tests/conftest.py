"""Shared fakes so the scorer tests need no GPU, no network and no 21 GB cache.

Both scorers are mostly orchestration: tokenize, forward, slice, reduce. The
slicing is where the silent bugs live (an off-by-one in the shift is invisible
downstream), and slicing does not care whether a real model produced the logits.
So the model and the tokenizer are replaced by stubs that behave the way the
real ones do in the two respects the code depends on:

  * the chat template's generation-prompt output is an exact token prefix of the
    full-conversation output, with the answer as the token straight after it;
  * the model returns `.logits` of shape (batch, seq, vocab) and a `.loss`.
"""
from types import SimpleNamespace

import torch

VOCAB = 512
GEN_TOKEN = 1   # marks the end of the generation prompt
EOT_TOKEN = 2   # end-of-turn, appended after the assistant content


class FakeTokenizer:
    """Character-level tokenizer with a prefix-correct chat template.

    Token ids are the characters' ordinals, so decoding is exact and every
    single-character answer letter is one token -- the same property the real
    Llama tokenizer has for "A"/"B"/"C"/"D".
    """

    pad_token = "<pad>"
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False,
                            return_dict=False):
        ids = []
        for message in messages[:2]:
            ids += [ord(c) for c in message["content"]]
        ids.append(GEN_TOKEN)
        if len(messages) > 2:
            ids += [ord(c) for c in messages[2]["content"]]
            ids.append(EOT_TOKEN)
        return ids

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class BrokenTemplateTokenizer(FakeTokenizer):
    """A tokenizer whose full-conversation output is NOT a prefix of its prompt.

    This is the failure `encode`'s assertion exists to catch: if it ever
    happened for real, the answer position would be wrong and every
    answer-position metric would be silently meaningless.
    """

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False,
                            return_dict=False):
        ids = super().apply_chat_template(
            messages, tokenize, add_generation_prompt, return_dict
        )
        if len(messages) > 2:
            return ids[1:]  # drop a leading token: no longer prefix-matched
        return ids


class OffsetAnswerTokenizer(FakeTokenizer):
    """Prefix-correct, but a stray token sits between the prompt and the answer.

    This is the failure `check_encoding`'s second counter exists for, and it is
    the nastier of the two: the prefix assertion still passes, so `encode`
    raises nothing, yet the token at `n_prompt` is no longer the answer letter
    and every answer-position metric reads the wrong slot.
    """

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False,
                            return_dict=False):
        ids = []
        for message in messages[:2]:
            ids += [ord(c) for c in message["content"]]
        ids.append(GEN_TOKEN)
        if len(messages) > 2:
            ids.append(EOT_TOKEN)  # stray token: shifts the answer by one
            ids += [ord(c) for c in messages[2]["content"]]
            ids.append(EOT_TOKEN)
        return ids


class FakeModel:
    """Returns deterministic logits of the right shape, plus a `.loss`.

    `loss_override` forces `.loss` to a chosen value so the shift-index
    assertion in `score_example` can be exercised in both directions.
    """

    def __init__(self, seed=0, loss_override=None, peaked_for=None):
        self.device = torch.device("cpu")
        self.seed = seed
        self.loss_override = loss_override
        self.peaked_for = peaked_for  # ids tensor to make near-certain
        self.calls = []

    def logits_for(self, input_ids):
        batch, seq = input_ids.shape
        g = torch.Generator().manual_seed(self.seed)
        logits = torch.randn(batch, seq, VOCAB, generator=g)
        if self.peaked_for is not None:
            # Make each position almost certain about the token that follows it,
            # so the gold log-probs are ~0 and sums are predictable.
            logits[:] = -20.0
            for i in range(seq - 1):
                logits[0, i, int(input_ids[0, i + 1])] = 20.0
        return logits

    def __call__(self, input_ids, labels=None, attention_mask=None):
        self.calls.append(input_ids.shape)
        logits = self.logits_for(input_ids)
        if self.loss_override is not None:
            loss = torch.tensor(float(self.loss_override))
        else:
            # The consistent value: -mean gold log-prob, matching what HF computes.
            lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
            gold = lp.gather(-1, input_ids[0, 1:].unsqueeze(-1)).squeeze(-1)
            loss = -gold.mean()
        return SimpleNamespace(logits=logits, loss=loss)


def make_item(item_id="mmlu_fake_0"):
    """A minimal pool item: the fields format_messages and loglik need."""
    return {
        "item_id": item_id,
        "question": "What is two plus two?",
        "A": "three",
        "B": "four",
        "C": "five and a half",
        "D": "six",
        "answer": "B",
        "subject": "fake_subject",
    }
