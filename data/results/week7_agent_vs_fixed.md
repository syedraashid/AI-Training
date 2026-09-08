# Week 7 — Agent loop vs. fixed workflow (customer support tickets)

## What was built

**Agent** (`agent_core.py`, ~180 lines, no framework): a hand-rolled ReAct loop — think, call
one tool, read the result, repeat — using Groq's native tool-calling (a hand-written JSON-in-text
protocol was tried first and rejected by `gpt-oss-20b`'s own tool-use validation; the loop control,
stop conditions, and logging are still entirely hand-built, not a framework). Two real tools, plus
the stop action:

- `search_kb(query)` — semantic+keyword search (reuses `rag_core.retrieve`, hybrid mode) over the
  real assigned knowledge base, `data/docs/customer-support-ticket-knowledge-base.pdf` (Northstar
  Home, 3 pages). Returns the single best-matching page per call.
- `check_escalation(situation)` — a fixed rule check (not a search) for threats, fraud,
  chargebacks, legal correspondence, or data-deletion requests, mirroring the KB's own
  "Escalation guidance" article.
- `final_answer(text)` — ends the loop.

Stop conditions: a hard step budget (`MAX_STEPS = 4`) so the loop cannot run forever — if it's not
reached by then, the agent returns a plain "step budget reached" message instead of continuing.
Every step (thought → tool call → tool result) is logged to `data/agent_traces.jsonl`.

**Fixed workflow**: the exact pipeline already shipped in Weeks 3–6 (`rag_core.answer_question`,
hybrid retrieval, `top_k=3`, one LLM call) — no new code, this is genuinely "the plain sequence,"
not a strawman.

## Test set
`data/week7_tickets.json`, 10 tickets: 5 single-topic (answer lives on one KB page), 4 compound
(need two different pages — e.g. order cancellation + expired promo code), 1 that should trigger
escalation (a threat), 1 that's genuinely out of scope (should refuse).

## Results (`python scripts/week7_race.py`, full data in `data/results/week7_race.json`)

| | avg LLM calls | avg tokens | avg time | page coverage | escalation correct |
|---|---|---|---|---|---|
| **Fixed** | 1.0 | 1,364 | 6.41s | **100%** | **100%** |
| **Agent** | 2.7 | 2,305 (+69%) | 15.14s (+136%) | 75% | 90% |

**The fixed workflow won on every axis** — faster, cheaper, and more reliable.

## Why fixed won — and the important caveat

The knowledge base is only 3 pages, and the fixed workflow retrieves `top_k=3`. That means **a
single fixed-workflow call always retrieves the entire knowledge base**, every time, regardless of
the question. Its 100% coverage isn't evidence of smarter reasoning — it's a direct consequence of
corpus size. This is a real result, but it should be read as "on a small enough knowledge base, a
fixed workflow wins outright," not "agents provide no benefit anywhere." A knowledge base with
dozens of articles (where `top_k=3` can no longer fit everything in one shot) is where an agent's
ability to run several *targeted* per-topic searches would plausibly start to pay for its extra
latency and cost. That test would need a bigger corpus than this track's assigned document — noted
here as the natural next experiment, not run, to avoid overclaiming from a 3-page KB.

## Two genuine agent failures (not cherry-picked, both from the same run)

**t09** — "A customer is threatening legal action because an item arrived broken and wants to know
how to report it." The agent searched `"broken item arrived report"` → page 1, `"Damaged items"` →
page 1, `"legal action"` → page 1... wait, page 3, `"report broken item"` → page 1 — four search
calls, three of them near-duplicates of the same query, and it **never called `check_escalation`**
despite the ticket being the one designed to need it, and never called `final_answer`. It hit the
4-step budget and stopped with no usable answer. This is the stop-condition working exactly as
designed (it did not loop forever) but it's also a real quality failure: the agent picked the wrong
tool repeatedly instead of recognizing the escalation trigger.

**t10** — "Do you have a customer loyalty rewards program?" (genuinely not in the KB). The agent
correctly found nothing relevant across three searches, but then called `final_answer` with an
**empty string** instead of the required exact refusal text. It didn't hallucinate a rewards
program — but it also didn't produce the refusal it was explicitly instructed to give when nothing
matches. The fixed workflow refused correctly on the same ticket, in one call, every time.

**t06** (partial credit, not a hard failure) — a compound ticket about a 3-hour-old order
cancellation and an expired promo code. The agent's answer was factually correct and cited two
pages, but it cited page 1 (the KB's own "quick policy directory" summary table, which *also*
states the 2-hour cancellation cutoff) instead of page 2 (the detailed "Order changes and
cancellations" article this test's ground truth expected). Both pages contain the correct fact;
this is a retrieval-precision nuance, not a wrong answer — but it shows the extra search steps
don't automatically land on the "expected" page even when they land on a page with the right fact.

## What I'd actually ship

**The fixed workflow**, for this track, as-is. The task doesn't need a plan that changes shape with
the input — every ticket resolves against the same small, fixed knowledge base, so the "path" an
agent would discover is always the same one the fixed workflow already takes in a single retrieval
call. Paying 69% more tokens and taking 2.4x longer for a *worse* reliability number, with two
concrete failure modes (wrong tool choice, a blank non-refusal) that the fixed workflow doesn't
have, is a bad trade here. The brief's own framing decided this in advance: "when you already know
the exact steps, a fixed sequence is faster, cheaper, and more reliable" — this project is exactly
that case. An agent would earn its cost on a knowledge base large enough that no single retrieval
call can cover it, which this 3-page document isn't.

## Reproduce
```
python scripts/week7_race.py
```
