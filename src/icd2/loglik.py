"""Log-likelihood multiple-choice scoring -- the correctness bit D1/D2/D3 consume.

The procedure is the standard one: build a prompt, append one option as the
continuation, sum the log-probability of that continuation's tokens, repeat for
all four options, predict the highest. No generation, nothing parsed.

What differs between implementations is WHAT the continuation is:

  `mode="letter"` (default) -- the continuation is "A"/"B"/"C"/"D", with the
      options enumerated in the prompt. This is what lm-eval-harness's `mmlu`
      task and HELM's joint multiple-choice adaptation actually do, and it is
      also the format the SFT adapter is trained on, so scoring and training
      agree.

  `mode="text"` -- the continuation is the option's text. This is the form used
      for HellaSwag/ARC-style tasks, where the options are free-form and length
      normalisation is needed.

Measured here on 300 random items across 54 subjects, base Llama-3.1-8B-Instruct:

    mode=letter          0.590  (all three rules identical)
    mode=text            0.493 sum / 0.403 token / 0.393 byte

`letter` wins by ~10 points and is the default for that reason, plus two others:
the Stage 4 sculptor's correctness-invariance mechanism needs a single answer
position, which multi-token text continuations do not have; and it matches the
training format.

Three scoring rules are computed and returned together, so an accuracy number
can never be quoted without recording which rule produced it:

  `sum`   -- raw sum of continuation log-probs        (lm-eval-harness `acc`)
  `token` -- divided by the continuation token count
  `byte`  -- divided by the UTF-8 byte length         (`acc_norm`)

Under `mode="letter"` the three are identical, because every letter is one token
and one byte. Normalisation only does work in `mode="text"`.
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from icd2.mmlu import CHOICES, format_messages
from icd2.scoring import load_model  # noqa: F401  (re-exported: one loader for both scorers)

RULES = ("sum", "token", "byte")
MODES = ("letter", "text")


def prompt_ids_for(tokenizer, item):
    """The shared prompt: the system + user turns, with a generation prompt.

    Identical to what `scoring.encode` builds from `messages[:2]`, so the
    correctness path and the membership path condition on the same tokens.
    """
    return tokenizer.apply_chat_template(
        format_messages(item)[:2], tokenize=True,
        add_generation_prompt=True, return_dict=False,
    )


def continuation_text(item, letter, mode="letter"):
    """The string scored as the continuation for one option."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}; got {mode!r}")
    return letter if mode == "letter" else item[letter]


def continuation_ids(tokenizer, text):
    """Tokens of a continuation, with no special tokens.

    Encoded directly rather than through `apply_chat_template`, which appends an
    end-of-turn token. Scoring that token would add a term that varies with the
    option and says nothing about the answer.
    """
    return tokenizer.encode(text, add_special_tokens=False)


def _sequences(tokenizer, item, mode="letter"):
    """For each option: (full_ids, n_prompt, n_cont_tokens, n_cont_bytes)."""
    prompt = prompt_ids_for(tokenizer, item)
    out = {}
    for letter in CHOICES:
        text = continuation_text(item, letter, mode)
        cont = continuation_ids(tokenizer, text)
        if not cont:
            raise ValueError(f"{item.get('item_id')}: option {letter} tokenized to nothing")
        out[letter] = (prompt + cont, len(prompt), len(cont), len(text.encode("utf-8")))
    return out


def _continuation_logprob(logits, ids, n_prompt):
    """Sum of log P over the continuation tokens of one sequence.

    Position i predicts token i+1, so continuation tokens sit at
    n_prompt .. len-1 and are predicted by logits at n_prompt-1 .. len-2.
    """
    lp = F.log_softmax(logits[n_prompt - 1: -1].float(), dim=-1)
    gold = ids[n_prompt:].unsqueeze(-1)
    return float(lp.gather(-1, gold).sum())


def score_choices(model, tokenizer, item, mode="letter", logit_transform=None):
    """Log-likelihood of each option. Returns {letter: {sum, token, byte}}.

    The four options are scored ONE AT A TIME, not as a padded batch.

    Batching looked safe -- a causal model attends only backwards, and padding
    sits after every real token -- and it ran 2x faster. Measured against the
    unbatched path it moved continuation sums by up to 0.435 and flipped 1
    prediction in 40. bf16 changes its reduction order with the batch dimension,
    and a summed continuation accumulates that. The unbatched path is
    bit-identical on repeat runs, and a 2.5% flip rate in the correctness bit
    would propagate into every IRT estimate.

    One sequence at a time also hands the transform a 2-D `[seq, V]` tensor,
    which is the contract the Stage 4 sculptor dispatches on; a batched
    `[4, seq, V]` would be mistaken for the MIA path's `[1, seq, V]`.
    """
    seqs = _sequences(tokenizer, item, mode)
    scores = {}
    for letter in CHOICES:
        ids, n_prompt, n_tok, n_byte = seqs[letter]
        tensor = torch.tensor([ids]).to(model.device)
        with torch.no_grad():
            logits = model(tensor).logits
        logits = logits[0]
        if logit_transform is not None:
            logits = logit_transform(logits, tensor, n_prompt)
        total = _continuation_logprob(logits, tensor[0], n_prompt)
        scores[letter] = {"sum": total, "token": total / n_tok, "byte": total / n_byte}
    return scores


def predict(scores, rule="sum"):
    """The option with the highest score under `rule`."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}; got {rule!r}")
    return max(CHOICES, key=lambda c: scores[c][rule])


def score_items(model, tokenizer, items, mode="letter", logit_transform=None, desc="loglik"):
    """Score a list of pool items. One row per item, in input order.

    Each row holds `item_id`, the gold letter, the four raw log-likelihoods, and
    -- for every rule -- the predicted letter and the correctness bit.
    """
    rows = []
    for item in tqdm(items, desc=desc):
        scores = score_choices(model, tokenizer, item, mode=mode,
                               logit_transform=logit_transform)
        row = {"item_id": item["item_id"], "answer": item["answer"], "mode": mode}
        for rule in RULES:
            pred = predict(scores, rule)
            row[f"pred_{rule}"] = pred
            row[f"correct_{rule}"] = int(pred == item["answer"])
        for letter in CHOICES:
            row[f"logp_{letter}"] = scores[letter]["sum"]
        rows.append(row)
    return rows


def correctness_vector(rows, rule="sum"):
    """The 0/1 responses in row order -- the `y` that `irt.theta_of` consumes."""
    return np.array([row[f"correct_{rule}"] for row in rows], dtype=float)


def accuracy(rows, rule="sum"):
    """Mean correctness under one rule."""
    return float(np.mean([row[f"correct_{rule}"] for row in rows]))
