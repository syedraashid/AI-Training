# Week 8 — Fixing "gives up quietly" (the worst failure found)

## Why this one

`week8_trajectory_eval.py` found three of twelve tickets (t06, t10, t11) where the agent did the
correct investigative work — right searches, right escalation decision — and then `final_answer`
came back **empty**. Traced with a raw request/response dump (not through `run_agent`, to see the
underlying API response directly):

```
finish_reason: stop
content: null
tool_calls: null
reasoning: "We have the relevant info: order cancellation possible during first 2 hours ...
            Provide answer citing pages."
```

`openai/gpt-oss-20b` (via Groq) decided what to say (visible in its own `reasoning` field) and then
didn't say it — no tool call, no content, turn just ends. This is a genuinely different failure from
Week 7's t09 (agent picks the wrong tool repeatedly): here tool choice and retrieval were correct
throughout, and the failure is entirely in the last step. It's also the most common failure in this
week's 12-ticket batch (3/12), and arguably the worst kind: the support agent operator gets *nothing*
back after the system did all the right work, with no visible error to explain why.

Reproduced directly: 5 repeated runs of the same ticket (`t06`, `agent_core.run_agent`, unmodified,
`temperature=0`) came back empty **3 of 5 times**. Same ticket, same messages, same model call —
purely stochastic on Groq's fast-inference serving, not a deterministic dead end in the loop's own
logic.

## The fix

`agent_core.py`, `GIVE_UP_RETRIES` (default 2): when a model turn comes back with no tool call *and*
no non-empty content, the loop does not accept it as the final answer. It regenerates the identical
request (same message history, not yet appended) up to the retry budget, and only falls back to
accepting the empty turn if every retry gives up too. This mirrors the existing `LLM_CALL_RETRIES`
handling already added for Groq's separate `BadRequestError` (tool-call-parsing) failures — both are
"this specific response was bad, ask again" fixes, not new agent logic. Each `final_answer` step now
logs `gave_up_retries_used` so the trace shows exactly how many times this fired.

## Measurement

`scripts/week8_giveup_fix.py` runs repeated trials per ticket with `give_up_retries=0` (before, i.e.
Week 7 behavior) vs. the default `give_up_retries=2` (after), and reports the empty-answer rate.

**Status: partial.** Groq's daily token quota (200,000 TPD) was exhausted mid-run on the third test
ticket — a real constraint, not a script bug (`groq.RateLimitError`, `tokens per day` limit,
confirmed at 199,912/200,000 used). Two of three planned tickets completed with n=10 trials each
before hitting the cap; the third was not run. Numbers below are exactly what those two completed
sweeps produced — nothing extrapolated or estimated for the missing ticket.

| ticket | before (no retry) | after (fixed) |
|---|---|---|
| "order cancellation + expired promo code" (t06) | 30% empty (3/10) | 30% empty (3/10), avg 1.4 retries used |
| "tracking delay + warranty coverage" (t07) | 20% empty (2/10) | 0% empty (0/10), avg 0.2 retries used |
| "password reset + cash refund" (t08) | not measured — quota exhausted before this ticket ran | not measured |

**Combined over the two measured tickets: 25% empty before (5/20) -> 15% empty after (3/20).** A
real reduction, but not a full fix — one of the two tickets (t06) showed *no* improvement even
though retries visibly fired (avg 1.4 retries used per run), meaning some of this model's empty
turns are retried into another empty turn rather than a real answer. The fix catches the failure
often enough to matter but should not be read as "solved."

## What could still get through

- **The give-up-quietly failure isn't eliminated, just reduced** — t06's 30%->30% result is the
  honest counter-example to a "problem fixed" narrative. `GIVE_UP_RETRIES` is a fixed budget (2); a
  request that reliably reproduces an empty turn for a specific ticket phrasing will still exhaust it
  and fall through to the "step budget reached" honest-failure message — better than a silent blank
  reply, but still not a successful resolution for that ticket.
- **No measurement exists yet for a third of the target test set** (the quota cap), so the 25%->15%
  number is real but not the full picture; a follow-up run once quota resets should complete it.
- **The retry budget costs tokens on every hit** — `total_tokens` per run goes up whenever a retry
  fires (visible in the `after` column's avg-retries-used), which is a real tradeoff against Week 7's
  cost comparison with the fixed workflow (`week7_agent_vs_fixed.md`): this fix makes the agent
  slightly more reliable and correspondingly more expensive, not free.

## Reproduce
```
python scripts/week8_trajectory_eval.py
python scripts/week8_giveup_fix.py
```
