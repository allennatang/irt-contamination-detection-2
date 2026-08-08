# Redesign prompt — stratified contamination + anchored-ability IRT detection

> Paste into a fresh window opened in `irt-contamination-detection/`.
> This is a **rebuild of the experimental design**, not a refactor. The v1 results in
> `progress.md` stay on disk as the comparison baseline; nothing here overwrites them.

---

## The checklist

The whole rebuild, one line per action. Each box names a concrete output. The reasoning
behind every choice is in the numbered sections below — this is the map, those are the
territory. **⛔ = stop and wait for me. ✅ = a gate; if it fails, stop and debug, do not
continue.**

**Stage 0 — orient** *(no code, ~30 min)*
- [ ] Read `plan.md`, `progress.md`, and the scripts listed under "Ground yourself first"
- [ ] Write back: what v1's split was, which confound v2 removes, anything I've misread
- [ ] ⛔

**Stage 1 — build the split** *(CPU, minutes)* → §2
- [ ] Agree `p` with me first (default 0.20)
- [ ] Pull the AllenAI prefit bank → item universe (~14,042 ids)
- [ ] Bin difficulty `b` into quartiles within each subject → 228 strata
- [ ] Assign within each stratum to `ANCHOR` / `CONTAM_HELD` / `CLEAN_HELD`, with
      `|CLEAN_HELD| == |CONTAM_HELD|`
- [ ] Write `data/splits/split_v2_seed{0..4}.csv` — 5 splits sharing `CLEAN_HELD`, each with
      its own `CONTAM_HELD` draw. **Seed 0 is canonical for Stages 2–4**; seeds 1–4 exist only
      for the Stage-5 panel and train nothing yet
- [ ] Balance table for seed 0: subject, `b`-quartile, and item length across the three sets
- [ ] ⛔

**Stage 2 — train the contaminated model** *(GPU; the one expensive step)* → §3
- [ ] Build the SFT set from `CONTAM_HELD` only, verbatim
- [ ] Propose the epoch grid; train **once**, checkpoint at each epoch → the dose axis
- [ ] Keep the base model (no adapter) as the control
- [ ] ⛔

**Stage 3 — the detectors** *(GPU ~20 min/scoring pass, then CPU)* → §4
- [ ] Score loglik correctness on all three sets + MIA scores on both held-out sets,
      for base + each checkpoint
- [ ] Fit θ̂_ANCHOR from `ANCHOR` only
- [ ] **D1** per-item residual AUROC, `CONTAM_HELD` vs `CLEAN_HELD`, + bootstrap CI
- [ ] **D2** `Δθ = θ̂(CONTAM_HELD) − θ̂(CLEAN_HELD)`, + bootstrap CI + permutation null
- [ ] **D3** anchored residual gap, + permutation null
- [ ] Same-split MIA detectors, at item level and model level
- [ ] ✅ **control**: base model must give D1 ≈ 0.5, D2 ≈ 0, D3 ≈ 0
- [ ] ✅ **transfer**: accuracy on all three sets, base vs each dose — does `CLEAN_HELD` rise?
- [ ] v1-vs-v2: D1/D2/D3 on the existing `orig_adapter` + v1 subject split (scoring-only)
- [ ] One results table: detector × {base, each dose, v1 adapter}, with CIs and p-values
- [ ] ⛔

**Stage 4 — evasion** *(GPU)* → §5
- [ ] Re-run the temperature sweep on the new split (transform code unchanged)
- [ ] Switch the sculptor's transplant pool to `ANCHOR` — confirm with me first
- [ ] Correctness-invariance assert over the **full** held-out set, not 80 items
- [ ] Re-check `logp_answer` / `logp_answer_z` now that the classes are matched
- [ ] Headline table: metric × {no transform, τ=2, sculpt}, D1/D2/D3 as invariant columns
- [ ] ⛔

**Stage 5 — PSN-IRT** *(design only, no implementation)* → §6
- [ ] Run the item-alignment check **first** — it gates the whole extension
- [ ] Estimate the panel size needed for `c` to move
- [ ] Written go/no-go

**Keep this checklist ticked in place as you go.** It is the running state of the rebuild,
not a plan I read once — update it in this file at the end of every stage.

---

## How I want you to work

I am rebuilding this experiment **to understand it**, not just to have it run. So:

- **Work one stage at a time**, following the checklist above. At the end of each stage, stop.
  Show me what you built, what number it produced, and what the next stage depends on. Tick the
  boxes in this file, then wait for me before moving on.
- **Before writing code for a stage, tell me the design decision you are about to make and
  why** — in 3–5 sentences, not an essay. If there is a fork where a reviewer would ask
  "why did you do it that way," name the fork and give me your recommendation.
- **Push back.** If a step I specified below is wrong, underspecified, or will produce a
  number that cannot answer the question I think it answers, say so before implementing it.
  I would rather lose a stage to argument than run 2,000 GPU-minutes on a confounded design.
- **Never silently widen scope.** If a stage needs something I did not list, ask.
- Ground claims in files and line numbers from this repo. Where you are recalling literature
  rather than reading this repo, mark it as such.

---

## Ground yourself first (Stage 0)

Read, in this order:

- `plan.md` — the Phase 0/1/2/3 evasion experiment (this survives the redesign, see §5).
- `progress.md` — the durable result log. Note especially the **2026-07-31** entry.
- `src/question_answer/mink_plus_plus.py` — the MIA detectors and the `logit_transform` seam.
- `src/question_answer/evaluate_mmlu_loglikelihood.py` — the correctness rule.
- `src/question_answer/join_irt_mink.py` — the IRT join (this is the file the redesign
  most changes).
- `src/question_answer/irt_residual_auc.py` — the residual detector.
- `src/question_answer/phase2_sculpt.py` — the attack.
- `src/question_answer/create_sft_dataset.py` + `train_contamination.py` — the contamination
  training path.

The docs are a checked record, but they are *my* framing of my own work. Treat interpretive
claims as hypotheses. **If the code contradicts the docs, the code wins and I want to know.**

**Stage 0 deliverable:** a short written statement of (i) what v1's member/non-member split
was, (ii) precisely which confound the redesign is meant to remove, and (iii) any place where
you think I have misread my own results. Then stop.

---

## §1 Why v1 is being replaced

In v1, members = **all** items of 5 MMLU subjects (`abstract_algebra`, `econometrics`,
`international_law`, `college_medicine`, `computer_security`); non-members = items from other,
untrained subjects. So *membership is perfectly confounded with subject identity*: any
member/non-member difference could be domain, difficulty, item length, or answer-letter prior
rather than memorization. Every detector in the repo — MIA and IRT alike — inherits that
confound. v1 also cannot distinguish **memorization of trained items** from **transfer to
untrained items in the same domain**, because it has no untrained items in the trained domains.

The redesign fixes this by contaminating a **stratified random subset of all 14,042 MMLU test
items**, so that contaminated and clean items are matched on subject and difficulty by
construction.

---

## §2 The split (Stage 1)

Partition all MMLU test items — universe = the items present in the AllenAI prefit 2PL bank
(`allenai/fluid-benchmarking`, `data/irt_models/mmlu.csv`, ids `mmlu_<subject>_<i>`) — into
three disjoint sets. **Note the names: they deliberately avoid IRT's `a`/`b`/`c`, which mean
discrimination / difficulty / guessing and are needed later.**

| Set | Trained on? | Role |
|---|---|---|
| `ANCHOR` | no | estimate ability θ̂; also the attack's transplant pool (§5) |
| `CONTAM_HELD` | **yes** | held out from θ̂ estimation; the positive class |
| `CLEAN_HELD` | no | held out from θ̂ estimation; the matched negative class |

Requirements:

- **Stratify on (subject × within-subject difficulty quartile)**, difficulty `b` taken from the
  prefit bank. 57 subjects × 4 bins = 228 strata; assign within each stratum. Rationale: the
  residual `r = correctness − P(θ,a,b)` is only difficulty-neutral if the 2PL fits perfectly;
  under misfit `r` correlates with `b`, so an unbalanced `b` distribution between
  `CONTAM_HELD` and `CLEAN_HELD` is a first-order confound. Stratifying is free.
- `|CLEAN_HELD| == |CONTAM_HELD|` (balanced classes; makes AUROC and the permutation null
  clean). `ANCHOR` gets the remainder.
- Contamination fraction `p = 0.20` of the universe → roughly 2,808 / 2,808 / 8,426.
  **Argue with me about `p` before implementing** — I want your view on the trade-off between
  detector signal and how much the model just learns "MMLU-ness" (see §4).
- Fixed seeds. Emit **five** committed CSVs — `data/splits/split_v2_seed{0..4}.csv`, columns
  `item_id, subject, a, b, difficulty_bin, assignment` — with this structure: all five share
  the **same** `CLEAN_HELD` (those items are never contaminated in any split → a guaranteed
  zero-contamination negative class for the Stage-5 panel), while each draws its own
  `CONTAM_HELD` from the remainder, so items accumulate a varying contaminated-model count
  across splits — the quantity the guessing rate `c` should track. **Seed 0 is canonical**:
  Stages 2–4 use it exclusively, and it is the single source of truth for their labels —
  nothing downstream may re-derive assignments. Seeds 1–4 train nothing until Stage 5 is a go.
  If you think the shared-`CLEAN_HELD` design is wrong (vs. fully independent draws), argue it
  at Stage 1, not Stage 5.
- Everything keys on `item_id`, never on question text. This deletes the fragile text-matching
  and duplicate-stem workaround at `join_irt_mink.py:169-186`.

**Stage 1 deliverable:** the split CSV plus a table showing subject, `b`-quantile, and
item-length balance across the three sets. Note that length is **not** stratified on — it is
balanced only in expectation by randomization within strata, so the length column is a real
check, not a formality (it addresses zlib's text-length confound, `ADVISOR.md:135-136`). If it
comes out visibly unbalanced, say so and propose either a re-draw or adding a length bin to the
strata. Then stop.

---

## §3 Training the contaminated model (Stage 2)

SFT a LoRA adapter on the **verbatim** `CONTAM_HELD` items, same format as
`create_sft_dataset.py` / `train_contamination.py` produce today.

- **Checkpoint at several epochs** and treat each checkpoint as a point on a *contamination
  dose* axis. One training run gives the whole sweep; separate runs per dose are the expensive
  way to get the same thing. Propose the epoch grid — and note the SFT set is ~2,808 items vs
  v1's ~608, so v1's epoch count and schedule are **not** a valid default; justify the grid
  against the new set size.
- Keep the base model (no adapter) as the control. It must be scored on the identical split.
- **One training seed for now.** Item-level bootstrap supplies the CIs. Two more training
  seeds get added just before the writeup, only if the headline result holds — do not spend
  them up front.

**Stage 2 deliverable:** trained checkpoints + a loss curve, and a one-line statement of what
the dose axis is and is not (it varies memorization depth at fixed item coverage; it does not
vary `p`). Then stop.

---

## §4 The detectors (Stage 3) — this is the part I care most about

**Read this before designing Stage 3.** In `progress.md` (2026-07-31) I already tested
"estimate θ̂ on clean items only" and it did **not** help per-item residual AUROC:
0.5759 → 0.5733. The reason is structural — θ̂ is a scalar, so shifting it raises all residuals
near-equally (members +0.054, non-members +0.060) and leaves the member/non-member *contrast*
unchanged. **The clean anchor set will not rescue per-item AUROC either.** If you think that
reasoning is wrong, say so and show me why; otherwise design Stage 3 accordingly.

So build **three** detectors and be explicit about which question each answers:

1. **D1 — per-item ranking.** `r = correctness − P(θ̂_ANCHOR, a, b)`, AUROC over
   `CONTAM_HELD` vs `CLEAN_HELD`. Question: *"is this item contaminated?"* Expect weak
   (v1 got 0.576). Report it honestly with a bootstrap CI; it is the secondary result.
2. **D2 — model-level ability gap.** `Δθ = θ̂(CONTAM_HELD) − θ̂(CLEAN_HELD)`, each fit against
   the same fixed prefit item parameters. Question: *"is this model contaminated, and by how
   much fabricated ability?"* Report Δθ with a bootstrap CI over items **and a permutation
   null**: relabel which held-out items are contaminated, **permuting within strata** — the
   null must mirror how assignment was actually done, or the p-value is anti-conservative —
   recompute Δθ, repeat. **I expect this to be the headline** — it aggregates ~2,800 items
   instead of ranking them — but I want you to tell me if that expectation is wrong.
3. **D3 — anchored residual gap.** `r̄(CONTAM_HELD) − r̄(CLEAN_HELD)` with θ fixed at
   θ̂_ANCHOR, same permutation null. This is D2's effect expressed in probability units and is
   the one that makes the clean anchor load-bearing (θ̂ is not contaminated by the items being
   tested).

Alongside each, report the MIA detectors from `mink_plus_plus.py` on the same split, at both
the item level and (by averaging within class) the model level, so the comparison is like-for-like.

**Controls that must pass before any of this is believed:**

- **Base model on the identical split**: D1 ≈ 0.5, D2 ≈ 0, D3 ≈ 0. Any nonzero gap here is a
  design artifact, not a signal. This is the single most important control.
- **Transfer measurement (new, and impossible in v1):** report absolute loglik accuracy on
  `ANCHOR`, `CONTAM_HELD`, `CLEAN_HELD` for base vs. each contaminated checkpoint. If accuracy
  on `CLEAN_HELD` rises above base, contamination is **transferring** to untrained items in the
  same domains, and every contrast-based detector is measuring only the *excess* of memorization
  over transfer. Say so plainly in the writeup — v1 could not see this at all, which is a
  positive result of the redesign, not a caveat.
- **v1-vs-v2 comparison (scoring-only, ~20 GPU-min):** run D1/D2/D3 on the **existing**
  `orig_adapter` with the v1 subject-confounded split. The same detector read on both splits
  quantifies directly what the subject confound was worth — the number that justifies this
  whole redesign. No training; the v1 adapter already exists.

**Stage 3 deliverable:** one committed script producing a single results table
(detector × {base, each dose}) with CIs and p-values. Then stop.

---

## §5 The evasion attack must survive intact (Stage 4)

The Phase 1 / Phase 2 / Phase 3 work in `plan.md` is **not** being redesigned. It operates on
logits and is orthogonal to which items are contaminated. Preserve it:

- The `logit_transform(logits, input_ids, n_prompt)` seam in `mink_plus_plus.py:108-109` and in
  both loglik scorers — unchanged.
- The temperature sweep (Phase 1) — unchanged, re-run on the new split.
- The moment-transplant sculptor (`phase2_sculpt.py`) — mechanism unchanged.

**One substantive change, and it is a fix, not a port.** Today the sculptor draws its
transplant pool from the non-member items — *the same items the AUROC is then computed on*. The
attack is calibrated in-sample on its own evaluation set, which a reviewer will flag. Under the
new split, **draw the pool from `ANCHOR`** and evaluate on `CONTAM_HELD` vs `CLEAN_HELD`. That
makes the attack strictly out-of-sample and strictly stronger as a result. Confirm you agree
before implementing.

Two other tightenings while you are here:

- The correctness-invariance assertion was verified on **80 items** (`ADVISOR.md:100`). Run it
  over the **full** held-out set. It is cheap (~12 it/s) and it is the theorem the whole claim
  rests on.
- Re-check whether the two answer-position-only metrics (`logp_answer`, `logp_answer_z`, still
  outside ε at 0.068 / 0.057 per `progress.md:202-203`) behave differently under the matched
  split. My open hypothesis (a) there — that the asymmetry comes from wrong-item contribution,
  since members have more correct items — is directly testable now that the classes are matched.

**Stage 4 deliverable:** the headline table, metric × {no transform, temperature τ=2, sculpt},
with D1/D2/D3 shown as invariant columns and full-set invariance PASS. Then stop.

---

## §6 PSN-IRT and the guessing-rate classifier (Stage 5 — design only for now)

**Do not implement this yet.** I want a written design and a feasibility verdict.

There is related work at `/home/mila/t/tanga/CODE/PSN-IRT` — a reproduction of PSN-IRT
(arXiv:2505.15055) plus my own `contamination/` addition that trains a classifier to predict
item contamination from fitted 4PL parameters, headlined by the guessing rate `c`. Read
`PSN-IRT/contamination/README.md` and `PSN-IRT/CLAUDE.md`.

**The opportunity.** That work's limitation #1 is *synthetic labels*: "contamination" there is
a simulated response-flip, so the results are an upper bound on how learnable *that specific
perturbation* is. This experiment produces **real contamination with known per-item ground
truth** — exactly what it is missing. Wiring them together would replace the synthetic labels
with real ones.

**The obstacles I want you to assess honestly, before I invest in it:**

1. **Panel size.** PSN-IRT fits `c` from a 12-model × 41,871-item response matrix
   (`PSN-IRT/data/combine.csv`; the MMLU block is rows 27029–41071, exactly 14,042 items per
   `PSN-IRT/results/benchmark_item_ranges.csv`). Adding **one** contaminated model to a
   12-model panel will barely move `c` — their own scenario cB shows that flipping 6 of 12
   models moves `c` by only +0.088. Estimate what panel composition is actually needed. The
   split machinery already anticipates this: Stage 1 emits 5 splits with a shared `CLEAN_HELD`
   (zero-contamination negatives) and overlapping `CONTAM_HELD` draws, so each item has a
   varying "contaminated-model count" — the quantity `c` should track. The open question is
   **how many adapters to actually train** (up to 5 exist as splits; only seed 0 is trained),
   and whether that training cost is worth it. Give me a number and the reasoning.
2. **Item alignment.** Does PSN-IRT's MMLU block order match the `mmlu_<subject>_<i>` ordering
   used by the AllenAI bank and by `fluid-benchmarking/benchmark_matrices/mmlu_irt_matrix.csv`
   (14,042 × 6, index `mmlu_abstract_algebra_0`, …)? `PSN-IRT/recover_mapping.py` recovers the
   *benchmark* block boundaries but not necessarily within-block order. **Verify this early and
   cheaply** — the whole extension is dead without it.
3. **Difficulty shortcut.** Their finding 3 is that difficulty `b`, not `c`, dominates feature
   importance, because injecting correct answers to hard items makes them look easy. Real
   contamination may not have that signature. State what you expect real leakage to do to
   `b` vs `c`, and design the ablation that would tell them apart.

**Stage 5 deliverable:** a written design + go/no-go, with the alignment check in §6.2 actually
run (it is a few lines and it gates everything). No implementation.

---

## Constraints

- Repo-local venv: `/home/mila/t/tanga/CODE/irt-contamination-detection/.venv/bin/python`.
- GPU node required for anything touching the model. **A full scoring pass over ~1,200 items
  takes ~2 minutes (~12 it/s)** — so a 14k-item pass is ~20 minutes. Treat new *scoring*
  experiments as nearly free and new *training* runs as the real cost.
- No new external deps; no external API calls. Results must reproduce from the repo.
- Fixed seeds everywhere; one entry-point script per stage; never overwrite v1 results.
- Target: NeurIPS 2026 "Can We Trust AI Evaluation?" workshop, ≤8 pages, non-archival.

## Reporting rules

- Report outcomes faithfully. If a detector comes out at chance, that is the finding — say it
  plainly rather than tuning until something looks positive.
- Every stage's numbers get appended to `progress.md` in the existing style, with the date, the
  n, and the exact command that produced them.
- Distinguish "this needs a new experiment" from "this needs better writing" every time.
