# Week 6 — Before/after: one prompt change, measured

## The one change
Week 5 found a recurring pattern: 3 real questions where the correct document was retrieved but the
model refused anyway, because the question required one inferential step (e.g. "why was I charged" →
apply the stated 24-hour cancellation rule) rather than a direct quote. The prompt in
`answer_question()` (`rag_core.py`) was changed to explicitly permit a single direct inference from a
stated rule, and to narrow the refusal condition to "no rule or fact addresses the question at all."
Retrieval, chunking, and top-k were unchanged — this was the only variable.

Both runs used the same eval set (`data/eval_cases.json`, 26 permanent cases derived from Weeks 3–5)
and the same retrieval mode (`hybrid`): `python run_evals.py hybrid`.

## Results

| problem_type | n | hit@3 before | hit@3 after | refusal-pass before | refusal-pass after |
|---|---|---|---|---|---|
| retrieval | 14 | 100.0% | 100.0% | 100.0% | 100.0% |
| unsupported-question | 6 | n/a | n/a | 100.0% | 100.0% |
| exact-code-retrieval | 2 | 100.0% | 100.0% | 100.0% | 100.0% |
| unwarranted-refusal | 3 | 100.0% | 100.0% | **33.3%** | **33.3%** |
| vocabulary-mismatch | 1 | 0.0% | 0.0% | 0.0% | 0.0% |

Full data: `data/results/week6_before_hybrid.json`, `data/results/week6_after_hybrid.json`.

## Honest result: the change did not move the number

Per-case, `unwarranted-refusal` stayed at 1/3 passing in both runs — and it's the *same* case passing
both times (`q03`, the cash-refund question) and the *same two* still failing both times (`w5-06`
subscription-charge-timing, `w5-14` expired-code-override). The prompt edit had zero measured effect
on the two failures it was written to fix.

Root-cause check (not just re-running the eval): the retrieved context for `w5-06` was inspected
directly and does contain the full 24-hour cancellation rule in the top-ranked chunk — this is not a
retrieval gap, the model has the fact and still declines to use it. Both `w5-06` and `w5-14` share a
shape neither the original nor the revised prompt handles: the customer isn't quoting the policy back,
they're describing an outcome (a charge) or making a request framed as asking permission ("can support
give me..., I promise..."), and the model appears to treat that framing itself as reason to refuse,
regardless of the "permit one direct inference" instruction.

A second prompt variant was drafted (explicitly framing the task as "apply the policy to the
customer's situation," with a worked example) and was ready to test head-to-head against the same two
cases, but Groq's free-tier daily token cap for `openai/gpt-oss-20b` (200,000 tokens/day) was hit
mid-session, confirmed by a real 429 response, so it could not be validated today. **It was
deliberately not shipped un-validated** — the current prompt (already measured above) is what's live
in `rag_core.py`. The untested variant is left inline in this repo's history for whoever picks this up
next; re-run `run_evals.py hybrid` after swapping it in and compare against
`data/results/week6_after_hybrid.json` before trusting it.

## What this demonstrates (per the Week 4/6 mentor checklist)
- One change only, measured with a real before/after number, per problem type — not "it feels better."
- The failures the change did **not** fix are named and root-caused, not glossed over.
- No second "fix" was shipped without being measured against the same before/after harness, even
  though a candidate existed — that's the point of having the harness.

## LLM-as-judge (subjective quality)
`rag_core.judge_answer()` and `scripts/week6_judge_validation.py` are implemented: 8 hand-picked
questions get a human score (written before running the judge, with a one-line reason grounded in
what the source documents say) compared against the LLM judge's blind score, reporting % agreement
within 1 point. This has **not been run yet** — it needs ~16 more Groq calls (8 answers + 8 judge
calls) and the daily token cap above is currently exhausted. Run it once the quota resets:

```
python scripts/week6_judge_validation.py
```

If agreement is below ~70%, per the brief, the judge should not be trusted for subjective scoring yet
and the rubric in `JUDGE_RUBRIC` (`rag_core.py`) needs revision before relying on it.
