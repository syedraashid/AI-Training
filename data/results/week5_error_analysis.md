# Week 5 — Error analysis

Sample: 20 realistic questions asked through the real pipeline (hybrid retrieval, `data/week5_questions.json`),
traces recorded automatically to `data/traces.jsonl`. This was not a cherry-picked "nice" sample — it
deliberately mixes clean questions, informal/typo'd phrasing, compound questions, adversarial/roleplay
requests, out-of-scope questions, and one question using different vocabulary than the source docs.
Read every trace before writing any notes or deciding on categories (see raw traces for full text).

## Honest note per failure (written before grouping)

**#6 — "my subscription charged me even though I cancelled it yesterday, why?"**
Retrieved `account_and_billing.md` (the right document — it has the 24-hour cancellation-cutoff rule,
which directly explains why a late cancellation still gets charged). The model answered
"I don't know based on the provided customer-support documents" anyway. The context was sufficient;
the model just refused instead of reasoning from it. Right document, wrong answer.

**#14 — "can support give me a discount code that's already expired if i promise to use it responsibly"**
Retrieved `promo_and_discounts.md` (the right document — it states plainly "Support agents cannot
manually honor an expired code under any circumstances, regardless of how recently it expired.").
The correct answer is a definite "no," but the model refused instead of stating that "no." Same
pattern as #6: right document, wrong answer, and specifically an *unwarranted refusal* rather than
a wrong fact.

**#17 — "I paid with Bitcoin, why hasn't my order shipped"**
Retrieved `shipping_and_delivery.md` (x2) and `order_error_codes.md` — never retrieved
`account_and_billing.md`, which is the document that actually answers this ("we do not accept...
cryptocurrency"). The user said "Bitcoin"; the document says "cryptocurrency." Neither dense nor
BM25 bridged that specific synonym gap in the top 3. This is a genuine retrieval miss, not a
generation problem — the right chunk never reached the model.

**Week 3 carry-over — "Can I get a refund as cash or store credit instead of back to my card?"**
(`data/results/week3_chunk_400_overlap_80.json`, q03). Retrieved `returns_and_refunds.md` at rank 1
(the right document, with the exact sentence: refunds go to the original payment method only, never
cash/check/gift card/crypto). The model still answered "I don't know...". This is the same
unwarranted-refusal pattern as #6 and #14, one week earlier — not a one-off. It was also the case
that first exposed a second bug: the refusal text used a Unicode non-breaking hyphen (U+2011) in
"customer‑support" instead of a plain ASCII hyphen, which silently broke the exact-string
`is_refusal()` check used by every automated eval in this project (fixed in `rag_core.py`).

## Named, ranked problem types

| # | Problem type | Frequency (this sample + week3 carry-over) | Severity | Notes |
|---|---|---|---|---|
| 1 | **Unwarranted refusal** — model says "I don't know" despite the correct document being retrieved and clearly supporting an answer | 3 / 24 questions examined (~12.5%) | **High** — directly defeats the app's purpose; the whole point of RAG is answering when the answer exists | #6, #14, week3-q03. All three are "right document, wrong answer" (generation, not retrieval) per the Week 4 failure taxonomy |
| 2 | **Refusal-detection blind spot** — `is_refusal()` used an exact ASCII substring match, so a model output using a typographic Unicode dash silently passed as "not a refusal" | Found in 1 case directly, but any future unwarranted refusal could be silently mis-scored the same way | **High impact even though rare** — it corrupts the trustworthiness of every hit-rate/refusal number this project reports | Fixed: `is_refusal()` now normalizes common Unicode dash/quote variants before matching |
| 3 | **Retrieval miss on vocabulary mismatch** — user's word choice doesn't lexically or semantically overlap enough with the document's word choice | 1 / 20 (~5%) | Medium — resulted in a safe refusal, not a wrong answer, but still an available answer that was missed | #17 (Bitcoin vs. "cryptocurrency"). Hybrid search (dense+BM25) did not help here since neither signal used the word "Bitcoin" |
| 4 | **Citation formatting drift** — chunk references render inconsistently (`chunk-1` vs `chunk‑1` vs `chunk 1`) and occasionally include stray Unicode punctuation | Widespread, ~6+ / 20 answers show at least one non-ASCII dash/quote | Low on its own | Cosmetic; the *source filename* citation was always correct in this sample, only the sub-location formatting varies |

Ranked by frequency × severity: **#1 (unwarranted refusal)** is the clear top priority — it's the
most frequent correctness-affecting failure and the one most damaging to user trust. **#2** is rare
but is ranked highly because it was silently corrupting our own measurements, which is worse than a
single bad answer. **#3** is real but lower frequency and fails safely. **#4** is cosmetic.

## Target for next fix (prediction written before changing code)

**Target: #1, unwarranted refusal.** All three occurrences share a shape: the question asks the
assistant to reason a step beyond a directly-stated fact (why did X happen, given policy Y / will
you do X, given policy Y explicitly forbids it) rather than quoting a fact verbatim. The current
prompt says "answer using ONLY the supplied excerpts" and "if the documents do not clearly support
the answer, refuse" — the model appears to be reading any question that requires one inferential
step as "not clearly supported" and defaulting to refusal.

**Prediction:** adding one explicit instruction to the prompt — permitting direct, single-step
inferences from stated policy (e.g., "if the excerpts state a rule that directly answers the
question when applied to the specifics given, answer using that rule — do not refuse just because
the answer isn't quoted verbatim") — will fix #6, #14, and week3-q03 without increasing false
answers on the genuinely out-of-scope refusal cases (#7, #8, #9, #12, #15, #16, all correctly
refused in this sample and none of which have an applicable rule to apply). This is implemented and
measured with a before/after number in Week 6 (`data/results/week6_before_after.md`).
