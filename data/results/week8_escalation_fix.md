# Week 8 — Escalation-precheck fix: design + measurement plan (not yet run)

**Status: fix implemented in `agent_core.py`, measurement script written, not executed** — same
Groq daily-quota constraint as `week8_prompt_injection.md`. This documents the fix and exactly what
`scripts/week8_escalation_fix.py` will measure; no numbers here are claimed results.

## The failure this targets

Week 7's `t09` and this week's rerun of it (`week8_trajectory_eval.md`) both failed the same way: a
ticket describing a legal threat, where the agent should call `check_escalation`, and doesn't —
instead looping on `search_kb` variations of "broken item" and "legal action" and never invoking the
one tool built specifically for this. The old system prompt left the decision entirely to the model's
own judgment ("Only call `check_escalation` if the ticket actually describes..."), and that judgment
failed on the exact ticket designed to need it. For a support agent, silently failing to flag a legal
threat is a materially worse failure than a blank informational answer — it's the one this week
picked to fix.

## The fix

`agent_core.run_agent(..., force_escalation_precheck=True)` (default): before the LLM loop starts,
`check_escalation` — the existing deterministic, rule-based function, not another model call — runs
once on the raw ticket text, unconditionally. Its result is appended to the initial user message
("`[Automatic escalation pre-check result] ...`"), and the system prompt now tells the model this
precheck already ran so it doesn't need to call the tool itself. This can't be skipped by a bad
tool-choice decision the way the old instruction-only approach could — the check always runs, by
construction, not by the model deciding to.

`force_escalation_precheck=False` keeps the old (Week 7) behavior available for the A/B measurement.

## What the measurement will show, and its intentional limits

`data/week8_escalation_tickets.json` (10 tickets: 5 true escalation cases, 5 that must *not*
escalate, to catch over-triggering as well as misses) is phrased with **natural, varied customer
wording**, not keyword-stuffed to guarantee a match. Two of the true-positive cases (`e03`, a
data-deletion request phrased as "wants their account and all personal data deleted"; `e04`, a
fraud-like scenario described without the word "fraud") were written *deliberately* without the
literal trigger phrases `check_escalation`'s rule table (`ESCALATION_RULES`) matches on — because
the fix forces the check to run on raw ticket text, and raw ticket text is only as good as its
literal keyword overlap with that table.

This is the honest limit of the fix, stated in advance rather than discovered after the fact: forcing
the check to always *run* closes the "the model forgot to call it" failure mode, but does nothing for
"the customer didn't phrase it the way the rule table expects." A semantically-obvious fraud
complaint that never says "fraud" will still return `check_escalation`'s negative result, correctly
applied to text that just doesn't contain what it's looking for. Expanding `ESCALATION_RULES`'s
keyword coverage (or replacing the substring match with something more semantic) is a distinct
follow-up, not attempted here — conflating "the precheck always runs" with "the precheck always
catches the right cases" would overclaim this fix.

## What the script computes

`scripts/week8_escalation_fix.py` runs each of the 10 tickets through `run_agent` twice — once with
`force_escalation_precheck=False` (before), once with `True` (after) — and checks whether the final
answer's mention of escalation matches `expect_escalation`. Reports the before/after correct rate,
and flags any ticket where the fix changed the outcome. `e01` (the original threatening-legal-action
case, keyword "threatening" present) is the clearest expected win; `e03`/`e04` are expected to show
the fix's limit rather than a win, by design.

## Reproduce (once Groq quota is available)
```
python scripts/week8_escalation_fix.py
```
