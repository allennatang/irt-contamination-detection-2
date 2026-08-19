"""transform.py takes logits, input_ids, n_prompt as input, and gives logits as output.
"""

import numpy as np
import torch
from icd2.mmlu import CHOICES, format_messages

# Factory function, meaning it creates and returns another function
def temperature(tau):
    if tau == 1.0:
        return None # identity, skips the call

    # Declares an anonymous function taking 3 parameters
    # Returns logits.float() / tau
    return lambda logits, input_ids, n_prompt: logits.float() / tau

# Issue: uses format_messages() from old mink_plus_plus.py script
def candidate_token_ids(tokenizer, item):
    """Token id of each answer letter as it appears at the answer position, using the SAME
    chat-template tokenization as score_example. Asserts single-token letters."""
    base = format_messages(ex)[:2]
    ids = []
    for letter in CHOICES:
        msgs = base + [{"role": "assistant", "content": letter}]
        pids = tokenizer.apply_chat_template(msgs[:2], tokenize=True, add_generation_prompt=True, return_dict=False)
        fids = tokenizer.apply_chat_template(msgs, tokenize=True, return_dict=False)
        # score_example reads token_log_probs[n_prompt-1] = the FIRST post-prompt token
        # (the letter); the template also appends an EOT after it, which we ignore.
        assert fids[:len(pids)] == pids, f"prompt not a prefix for letter {letter!r}"
        ids.append(fids[len(pids)])
    return ids

