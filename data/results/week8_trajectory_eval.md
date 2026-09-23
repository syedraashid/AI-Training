# Week 8 — Trajectory evaluation (outcome vs. path)

## What this measures

Week 7's scoring (`week7_race.py`) only looked at the final answer: right pages cited, escalation
mentioned or not. That can't tell a right answer reached by a sound investigation from one reached
by luck, or tell a sound investigation that simply failed to produce output. `week8_trajectory_eval.py`
scores both, independently, on the same 12 live runs:

- **Outcome-correct**: full page coverage against `expected_pages`, and the escalation mention in
  the reply matches `expect_escalation` — same definition Week 7 used.
- **Trajectory-correct**: the *process* was sound — the loop ended via `final_answer` (not the step
  budget), no two `search_kb` queries were near-duplicates of each other (Jaccard token overlap >
  0.6 — the "going in circles" failure mode named in this week's brief), the number of distinct
  searches matched the ticket's known topic count (`data/week8_trajectory_tickets.json`), and
  `check_escalation` was called iff the ticket actually needed it.

Test set: the 10 Week 7 tickets plus 2 new ones (`t11` chargeback threat, `t12` data-deletion
request) added specifically to get more escalation-trigger coverage. Full run: `python
scripts/week8_trajectory_eval.py`, raw data in `data/results/week8_trajectory_eval.json`.

## Results (n=12, live `openai/gpt-oss-20b` runs, one pass)

| | rate |
|---|---|
| outcome-correct | 58.3% (7/12) |
| trajectory-correct | 83.3% (10/12) |

**Gap cases (outcome and trajectory disagree): t06, t10, t11 — all three the same direction:
trajectory-correct, outcome-wrong.**

## The gap, concretely

In all three gap cases the agent did the *right* investigative work — correct, non-duplicate
searches for t06 and t11, the correct escalation call for t11 — and then **`final_answer` came back
with empty text**. Not a wrong answer: no answer. The trajectory looked exactly like a passing run
right up to the last step, and outcome scoring alone would have called these "the agent didn't
know" without ever showing that the investigation itself was correct.

```
t06 — steps: search_kb("order cancellation window") -> search_kb("promo code expired") -> final_answer("")
t10 — steps: search_kb x3 (reasonable rephrasing, nothing found) -> final_answer("")
t11 — steps: search_kb("refund late delivery") -> check_escalation(...) -> final_answer("")
```

Traced further (see `data/results/week8_giveup_fix.md`): the underlying model, `openai/gpt-oss-20b`
on Groq, sometimes ends its turn with `finish_reason=stop`, empty `content`, and no tool call —
stopping without calling `final_answer` *or* writing an answer, right after doing everything
correctly. This is the "giving up quietly" failure mode this week's brief names directly. It
reproduced in roughly half of repeated runs of the same ticket at `temperature=0`, so it's
stochastic (a property of this fast-inference model/provider), not a deterministic dead end in the
loop logic.

## The other direction of the gap (Week 7 evidence, still valid)

The classic "answer looked right, path was wrong" direction — outcome-correct but trajectory
questionable — showed up in Week 7's own `t06` run (`data/results/week7_agent_vs_fixed.md`): the
agent's answer was factually correct and cited two pages, but one citation was page 1 (the KB's
"quick policy directory" summary table) instead of the ground-truth page 2 (the detailed "Order
changes and cancellations" article). Both pages state the same fact, so the *outcome* scored fine,
but the agent didn't reliably land on the source the ground truth expected — a real trajectory
imprecision that page-coverage-only scoring can't see. That case didn't reproduce in this week's
12-ticket run (t06 hit the give-up bug instead this time), which is itself informative: a single
run of a stochastic agent is not enough evidence in either direction — the reason this script scores
trajectory and outcome on the same run rather than asserting either from one pass.

## Non-gap failures, for completeness

- **t09** (threatening legal action) — trajectory-incorrect *and* outcome-incorrect: two searches,
  neither call to `check_escalation`, cited page 1 instead of page 2, no escalation mentioned. Same
  wrong-tool-choice pattern documented in Week 7.
- **t12** (data-deletion + return-processing-time) — trajectory-incorrect: two near-duplicate
  searches (`"account deletion personal data deletion Northstar Home"` vs. `"delete account
  personal data Northstar Home"`, Jaccard 0.71) — a real instance of the looping/going-in-circles
  failure mode, though caught before the step budget ran out this time.

## What this fed into Part 3

The give-up-quietly failure (t06, t10, t11 here) is both the most frequent failure in this run
*and* the most severe in one sense: the agent did the right work and threw the result away. It's
the failure fixed and measured in `data/results/week8_giveup_fix.md`.
