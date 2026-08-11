"""The GPU boundary: the only module that loads a model.

Everything here is teacher forcing, not generation. Each item's question AND its
correct answer letter go through the model in ONE forward pass, and the code
reads back the probability the model assigned to tokens that were already there.
Nothing is generated and nothing is parsed: the answer position is arithmetic
(`n_prompt - 1`), proved correct by the prefix assertion in `encode`.

The output is 24 membership scores per item -- no token, no letter, no
correctness bit. Correctness is a different job and belongs in `loglik.py`,
which should import `load_model` from here rather than loading its own.

Layering: all arithmetic lives in `mia.py`, which is pure and CPU-testable. This
file supplies the forward pass, the `logit_transform` seam, and two assertions
that catch failures nothing downstream can see.

Labels never appear here. Membership comes from the split manifest and enters at
`detectors.py`, so one function scores MEMBER, NONMEMBER and ANCHOR alike.
"""

import numpy as np
import torch
from tqdm import tqdm

from icd2 import mia
from icd2.mmlu import format_messages

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# HF reduces its own loss in the model's compute dtype while we reduce in fp32,
# so the two agree to about 1e-3, not to float epsilon. Tightening this would
# make the assertion fire on correct code.
LOSS_ASSERT_TOL = 1e-2


def load_model(model_id=DEFAULT_MODEL, adapter_path=None, four_bit=False):
    """Load the base model, optionally with a LoRA adapter on top.

    bf16 is the default. v1 used 4-bit nf4 only so its numbers stayed
    comparable; that constraint is gone with the v1 regression, and an 8B model
    in bf16 needs ~16 GB against the L40S's 46 GB. Quantization adds noise to
    every log-probability, which is the quantity being measured, so avoid it
    unless memory forces it. `four_bit=True` reproduces v1's configuration.

    Returns (model, tokenizer). The model is in eval mode.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if four_bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quant, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto"
        )

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def encode(tokenizer, messages):
    """Messages -> (input_ids, n_prompt). No model needed, so this runs on CPU.

    Tokenizes the system+user turns with a generation prompt, then the full
    conversation, and asserts the first is an exact token prefix of the second.
    That assertion is what makes the answer position exact rather than inferred:
    the answer letter is the single token at index `n_prompt`, so the logit that
    predicts it sits at `n_prompt - 1`.

    If it ever fires, every answer-position metric is silently meaningless --
    which is why it is an assertion and not a warning.
    """
    prompt_ids = tokenizer.apply_chat_template(
        messages[:2], tokenize=True, add_generation_prompt=True, return_dict=False
    )
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
    if list(full_ids[: len(prompt_ids)]) != list(prompt_ids):
        raise AssertionError(
            "chat template did not prefix-match prompt vs full sequence; the "
            "answer-token position assumption is broken for this tokenizer/template"
        )
    return torch.tensor([full_ids]), len(prompt_ids)


def answer_position(n_prompt):
    """Index of the logit that predicts the answer letter.

    Position i predicts token i+1, and the answer letter is token `n_prompt`.
    """
    return n_prompt - 1


def forward_logits(model, input_ids):
    """One teacher-forced forward pass. Returns (logits, hf_loss).

    `labels=` exists only to obtain HF's own loss for the consistency assertion;
    nothing else uses it. Read the outputs by attribute rather than by tuple
    position, which survives output-class changes between library versions.
    """
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return outputs.logits, float(outputs.loss)


def zlib_text(messages):
    """The text zlib compresses: every turn joined by newlines, as v1 did."""
    return "\n".join(m["content"] for m in messages)


def score_example(model, tokenizer, messages, logit_transform=None, return_arrays=False):
    """Score one item -> dict of 24 membership metrics.

    Order of operations: encode, forward, transform, triples, metrics.

    The transform receives the model's raw logits, not a pre-cast copy. A
    transform is responsible for its own dtype (returning fp32), and pre-casting
    here would hide a transform that failed to. `None` means identity.
    """
    input_ids, n_prompt = encode(tokenizer, messages)
    logits, hf_loss = forward_logits(model, input_ids)
    input_ids = input_ids.to(logits.device)

    if logit_transform is not None:
        logits = logit_transform(logits, input_ids, n_prompt)

    t, mu, var = mia.triples_from_logits(logits, input_ids)

    # Catches a shift-index error: we align logits[:-1] against input_ids[1:]
    # while HF shifts internally. If the two disagree, the means diverge. Every
    # metric would still be self-consistent, just computed one position off.
    # Only valid under identity -- hf_loss comes from the untransformed logits.
    if logit_transform is None:
        mean_gold = float(np.mean(t))
        if abs(mean_gold - (-hf_loss)) > LOSS_ASSERT_TOL:
            raise AssertionError(
                f"mean gold log-prob {mean_gold:.6f} != -outputs.loss {-hf_loss:.6f}; "
                f"the shift index disagrees with the model's own labels"
            )

    scores = mia.all_metrics(t, mu, var, answer_position(n_prompt), zlib_text(messages))

    if return_arrays:
        return scores, {"t": t, "mu": mu, "var": var, "answer_pos": answer_position(n_prompt)}
    return scores


def score_items(model, tokenizer, items, logit_transform=None, desc="scoring"):
    """Score a list of pool items, in order. Returns a list of dicts.

    Each dict is `item_id` plus the 24 metrics. The item_id is carried through
    because the detectors join on it -- never on question text, which is not
    unique across MMLU.

    Batch size stays 1 by design: a batch needs padding, and pad positions would
    contaminate the vocabulary-wide sums mu and var and shift `n_prompt` per row.
    """
    rows = []
    for item in tqdm(items, desc=desc):
        scores = score_example(model, tokenizer, format_messages(item),
                               logit_transform=logit_transform)
        rows.append({"item_id": item["item_id"], **scores})
    return rows


def collect_t_pool(model, tokenizer, items, desc="t-pool"):
    """Gather every position's gold log-probability, concatenated over `items`.

    This is the Stage 4 sculptor's transplant pool. Draw it from ANCHOR: v1 drew
    it from the very items the AUROC was computed on, which calibrated the
    attack in-sample on its own evaluation set. A named function rather than a
    flag, so the pool's provenance is visible at the call site.
    """
    chunks = []
    for item in tqdm(items, desc=desc):
        _, arrays = score_example(model, tokenizer, format_messages(item),
                                  return_arrays=True)
        chunks.append(arrays["t"])
    return np.concatenate(chunks)


# --------------------------------------------------------------------------- #
# CPU pre-flight -- run this before asking for a GPU
# --------------------------------------------------------------------------- #
def check_encoding(tokenizer, items):
    """Verify the answer-position assumption over a whole item set, no model.

    Returns (n_checked, prefix_failures, letter_mismatches). Both counts must be
    zero. This is seconds of CPU and it forecloses the single worst silent
    failure in the pipeline.
    """
    prefix_failures, letter_mismatches = 0, 0
    for item in tqdm(items, desc="pre-flight"):
        messages = format_messages(item)
        try:
            input_ids, n_prompt = encode(tokenizer, messages)
        except AssertionError:
            prefix_failures += 1
            continue
        served = tokenizer.decode([int(input_ids[0, n_prompt])]).strip()
        if served != item["answer"]:
            letter_mismatches += 1
    return len(items), prefix_failures, letter_mismatches
