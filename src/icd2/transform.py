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
    base = format_messages(item)[:2]
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
    def __init__(self, t_pool, candidate_ids, seed):
        self.t_pool = np.asarray(t_pool, dtype=np.float64)
        self.candidate_ids = candidate_ids
        self.seed = seed
        self.margin = 5.0
        self.rng = np.random.default_rng(seed)


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

        # Path 3 — 3-D input (all other cases)
        # This is the MIA scorer path. It has more steps:

        # 1. Take the batch's first (and only) sequence.
        # 2. For each non-answer position, draw a random target probability from the non-member pool.
        # 3. Rescale each row so the new target probability fits, and the row still sums to 1.
        # 4. Convert back to log-probabilities.
        # 5. At the answer position, apply _answer_row instead of the random-transplant. This overwrites step 2–4's work at that one position, so the answer slot never leaks the gold answer.
        # 6. Return the result as float32, not the model's native bfloat16.
        
        # MIA path: [1, seq, V]
        # Only 1 sequence, so it works on a 2-D matrix of rows.
        L = logits[0]
        seq = L.shape[0]
        n = seq - 1                      # positions 0..seq-2 predict tokens 1..seq-1
        Lp = L[:n].float()
        p = torch.softmax(Lp, dim=-1)
        targets = input_ids[0, 1:seq]    # [n]

        # Draw n random log-probability values FROM the non-member pool. One value per row.
        t_star = torch.tensor(self.rng.choice(self.t_pool, size=n),
                              device=Lp.device, dtype=Lp.dtype)

        # Convert each drawn log-probability into a plain probability. Clamp it below 1, so room remains for the rest of the distribution.
        a = torch.exp(t_star).clamp(max=1.0 - 1e-6)                # target prob [n]
        # a[i] is the NEW probability the sculpted row must give to the real target token, at position i.

        ### Rescale each row so it still sums to 1

        # Read the ORIGINAL probability the model gave to the real target token, at each row. This is the "before" value.
        p_tgt = p.gather(1, targets.view(-1, 1)).squeeze(1)

        # Compute one scale factor per row. This factor shrinks or grows the REST of the row (every entry except the target), so the row still sums to 1 after the target changes.
        scale = ((1.0 - a) / (1.0 - p_tgt).clamp_min(1e-9)).unsqueeze(1)  # [n,1]

        # Apply the scale factor to the WHOLE row.
        q = p * scale

        # Now overwrite ONLY the target column with the new value a. The other columns keep their scaled value.
        q.scatter_(1, targets.view(-1, 1), a.unsqueeze(1))

        ### Result: each row is a valid probability distribution. T
        ### The target token has the new (non-member-style) probability. All other tokens keep their RELATIVE shape, just scaled.
        
        # Convert back to log form
        newlog = torch.log(q.clamp_min(1e-30))                    # log q; softmax(log q)=q

        # Overwrite the answer row
        if 0 <= answer_pos < n:
            newlog[answer_pos] = self._answer_row(Lp[answer_pos])

        # Return FLOAT32. Writing the sculpted log-probs into the model's bf16 logits
        # corrupts the downstream log_softmax renormalization (8-bit mantissa), biasing
        # token_log_probs upward and leaking the members' original confidence back in.
        out = L.float().clone()
        out[:n] = newlog
        return out.unsqueeze(0)


    # Returns the candidate id with the largest logit of the four
    def _winner(self, row):
        cand = self.candidate_ids
        return cand[int(torch.argmax(torch.stack([row[c] for c in cand])))]

    # Sets each other candidate to row[winner] - margin if it is larger than or equal to the winner. 
    # This keeps the winner as the strict argmax of the four, so the correctness bit does not change.
    def _reseat(row, winner):
        """Clamp the three non-winner candidate letters below `winner` so it stays the
        strict argmax-of-4 ⇒ the model's own pick (hence the correctness bit) is unchanged."""
        wv = row[winner].item()
        for c in self.candidate_ids:
            if c != winner and row[c].item() >= wv:
                row[c] = wv - self.margin
        return row

