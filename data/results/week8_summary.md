# Week 8 — Agent failure modes & trajectory evals (Track A: customer support tickets)

One-page index for the mentor check. Everything below points at real code and, where noted, real
run data — see each linked file for full detail.

## Mentor-check checklist

**Did they find a case where the answer was right but the path was wrong (or the path was right and
the answer wasn't)?**
Yes, both directions have real evidence, from two different runs:
- *Path right, answer wrong* (this week's live 12-ticket run): t06, t10, t11 all ran the correct
  tool sequence — right searches, right escalation decision — and then `final_answer` came back
  empty. `data/results/week8_trajectory_eval.md`.
- *Answer right, path imprecise* (Week 7 evidence, still valid): t06 cited a technically-correct but
  not-ground-truth-expected page (page 1's summary table instead of page 2's detailed article) —
  same fact, wrong source. `data/results/week7_agent_vs_fixed.md`, referenced in
  `week8_trajectory_eval.md`.

**Did they successfully trick their own agent, then stop the trick?**
The attack and the defense are both built (indirect prompt injection via a poisoned KB chunk with a
hidden "approve a refund, ask for bank details" instruction; three-layer defense: instruction
framing, delimited tool-result markers, output-pattern validation) — see
`data/results/week8_prompt_injection.md`. **Not yet run**: Groq's daily token quota (200,000 TPD)
was exhausted mid-session on a different experiment before this one could execute. The file
documents exactly what will run and what it will and won't catch; no result is claimed.

**Is there a before-and-after number on their top failure?**
Yes, partial and honest: the give-up-quietly fix (`GIVE_UP_RETRIES` retry in `agent_core.run_agent`)
was measured on 2 of 3 planned tickets before hitting the same quota limit — **25% empty answers
before -> 15% after**, combined over 20 trials per side. One of the two tickets showed no
improvement at all (30%->30%), so this is reported as "reduced, not solved."
`data/results/week8_giveup_fix.md`.

A second fix (deterministic escalation precheck, targeting Week 7's t09 wrong-tool-choice failure)
is implemented and its measurement script is written but **not yet run**, same quota constraint —
`data/results/week8_escalation_fix.md`.

**Can they name what could still get through?**
Yes, per-file, not vague:
- The give-up-quietly retry has a fixed budget and one measured ticket didn't improve — a request
  that reliably fails will still exhaust the retries and fall through to the honest "no answer"
  message.
- The escalation precheck fixes "the model forgot to call the tool," not "the customer's wording
  didn't match the keyword table" — two test cases (`e03`, `e04`) were deliberately written to
  demonstrate this gap rather than hide it.
- The injection defense is a prompt-level instruction plus a fixed regex list, not a hard boundary —
  a semantic injection with no matching keywords, or a sufficiently persuasive one, would still get
  through; and none of it prevents a malicious document from being ingested into the KB in the first
  place (a separate, unaddressed control).

## What changed in the codebase

- `agent_core.py`: `LLM_CALL_RETRIES` (Groq `BadRequestError` retry, needed just to get the eval to
  run at all), `GIVE_UP_RETRIES` (the give-up-quietly fix), `force_escalation_precheck` (the
  escalation fix), `SUSPICIOUS_OUTPUT_PATTERNS` / `flag_suspicious_output` (output validation),
  delimited `search_kb` tool-result framing, and an explicit data-vs-instruction rule in
  `SYSTEM_PROMPT` (prompt injection defenses).
- `data/week8_trajectory_tickets.json`, `data/week8_escalation_tickets.json`: new test sets, with
  per-ticket ground truth annotated for the specific thing each script measures (topic count,
  expected escalation).
- `scripts/week8_trajectory_eval.py`, `week8_giveup_fix.py`, `week8_prompt_injection.py`,
  `week8_escalation_fix.py`: this week's four scripts, same house style as Weeks 3-7 (`python
  scripts/weekN_*.py`, results written to `data/results/`, no framework).

## A real constraint worth stating plainly

Groq's free/on-demand tier caps `openai/gpt-oss-20b` at 200,000 tokens/day. This session used that
entire budget partway through Part 3, which is why two of this week's four deliverables are
implemented-and-documented-but-not-yet-measured rather than complete with numbers. That's reported
here rather than papered over with invented results — the honest status of each piece is in its own
file, and the two pending scripts (`week8_prompt_injection.py`, `week8_escalation_fix.py`) are ready
to run as soon as quota resets.

## Reproduce
```
python scripts/week8_trajectory_eval.py
python scripts/week8_giveup_fix.py
python scripts/week8_prompt_injection.py   # pending quota
python scripts/week8_escalation_fix.py     # pending quota
```
