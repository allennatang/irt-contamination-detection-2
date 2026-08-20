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


class Sculptor:
    """Sculptor constructor. Performs a logit transform.

    Input is logits, input_ids, n_prompt

    The constructor stores four items:
    - t_pool: a set of log-probability values from non-member data. The code draws random samples from this pool.
    - candidate_ids: the token IDs for the four answer letters (A, B, C, D).
    - rng: a random number generator. The same seed gives the same output each time.
    - margin: a gap value. The code uses this value to keep one letter clearly on top.


    """

    # Checks the number of dimensions (logits.ndim) and picks the correct path.
    def __call__(self, logits, input_ids, n_prompt):
        # Where n_prompt is the token count of the prompt (system + question, no answer)

        # The second-last row of logits. Usually, the last row of logits is used to predict the first new token
        # Since the answer token is already generated, then in this case, the second-last row was used to predict the answer token.
        answer_pos = n_prompt - 1 # The "answer row"


        # fast loglik scorer, input is the answer position only
        # calls _answer_row directly and returns the result
        if logits.ndim == 1: 
            return self._answer_row(logits)

        # This is the full loglik scorer. 
        # The method copies the logits. 
        # It then replaces only the answer row with the output of _answer_row. All other rows stay the same.
        if logits.ndim == 2:
            out = logits.clone()
            # It changes the answer row
            out[answer_pos] = self._answer_row(out[answer_pos])
            return out
        

