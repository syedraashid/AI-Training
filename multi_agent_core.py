"""Week 10: the "triage squad" -- a manager plus two narrow specialists -- built specifically to
race against the single hand-built agent in agent_core.py on the same tests (see
data/results/week10_multi_agent_race.md for the result, not assumed here).

Why this split, not some other one: the two specialists mirror the two kinds of tools the single
agent already has, so each one is genuinely narrow rather than an arbitrary division of labor --

- kb_specialist: search_kb only. Answers policy/informational questions from the KB.
- risk_specialist: check_escalation plus the Week 9 MCP tools (get_ticket_history,
  get_order_status). Decides escalation and answers account-specific questions.

Both specialists ARE agent_core.run_agent, called with allowed_tool_names restricting what the
model can see -- not a second hand-rolled loop. That reuse is deliberate: it means any difference
in results between "single agent" and "team" comes from the team *structure* (routing + narrow
tools + a synthesis hand-off), not from two different, unequally-debugged pieces of code.

The manager has two jobs, each its own plain (tool-free) LLM call -- deliberately not another
tool-using loop, since classifying which specialist(s) are needed and merging their reports back
into one reply are both single-shot text tasks, not multi-step investigations:

1. ROUTING: read the ticket, decide which specialist(s) it needs (one, or both).
2. SYNTHESIS: after the chosen specialist(s) report back, merge their findings into one
   customer-facing reply, preserving every citation.

Every one of those calls -- routing, each specialist's own internal loop, synthesis -- is a full
hand-off that re-sends the ticket (and, for synthesis, both specialists' full output) from
scratch. That's the "hidden cost" this week's brief names directly, not a simplification this
module is trying to minimize -- the point of the race is to measure it honestly.
"""

from __future__ import annotations

import json
import time
from typing import Any

import agent_core
import rag_core

GENERATION_MODEL = "openai/gpt-oss-20b"

KB_SPECIALIST_TOOLS = {"search_kb", "final_answer"}
RISK_SPECIALIST_TOOLS = {"check_escalation", "get_ticket_history", "get_order_status", "final_answer"}

ROUTING_PROMPT = """You are the triage manager for a Northstar Home customer-support team of two specialists:

- kb_specialist: answers questions using the support knowledge base (policies -- returns, refunds,
  warranty, shipping, promotions, password reset, order cancellation, error codes).
- risk_specialist: decides whether a ticket must be escalated (threats, suspected fraud,
  chargebacks, legal correspondence, data-deletion requests) and can look up a customer's past
  ticket/order history by email or order id.

Read the ticket and decide which specialist(s) it needs -- a ticket can need one or both when it
raises more than one kind of question. When genuinely unsure, include both rather than guessing
wrong and missing something the ticket needed.

Reply with STRICT JSON only, no other text, in exactly this shape:
{{"specialists": ["kb_specialist"], "reasoning": "<one sentence>"}}
(specialists may list one or both names, in any order)

Ticket: {ticket}
"""

SYNTHESIS_PROMPT = """You are the triage manager for Northstar Home. Your specialists have reported
back on one ticket. Combine their findings into ONE reply for the customer -- preserve every
citation exactly as a specialist gave it, do not invent anything beyond what they reported, and if
either specialist found nothing relevant to their part, do not mention that absence. If a
specialist's result is ESCALATE, make sure your reply says this needs escalation.

Original ticket: {ticket}

{specialist_reports}

Reply with the final customer-facing text only, no preamble, no explanation of your own process.
"""


def _plain_completion(prompt: str) -> tuple[str, dict[str, int]]:
    response = rag_core.chat_completion_with_backoff(
        model=GENERATION_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0,
    )
    text = (response.choices[0].message.content or "").strip()
    usage = response.usage
    tokens = {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
    }
    return text, tokens


def _route(ticket: str) -> tuple[list[str], dict[str, int]]:
    text, tokens = _plain_completion(ROUTING_PROMPT.format(ticket=ticket))
    cleaned = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(cleaned)
        specialists = [s for s in parsed.get("specialists", []) if s in ("kb_specialist", "risk_specialist")]
    except (json.JSONDecodeError, AttributeError):
        specialists = []
    if not specialists:
        # Unparseable routing decision -- fail safe to "ask both" rather than silently dropping
        # the ticket, the same safe-stop philosophy as agent_core's own retry fallbacks.
        specialists = ["kb_specialist", "risk_specialist"]
    return specialists, tokens


def _run_specialist(name: str, ticket: str) -> dict[str, Any]:
    if name == "kb_specialist":
        return agent_core.run_agent(
            ticket, generation_model=GENERATION_MODEL, allowed_tool_names=KB_SPECIALIST_TOOLS,
            force_escalation_precheck=False, use_mcp_tools=False,
        )
    return agent_core.run_agent(
        ticket, generation_model=GENERATION_MODEL, allowed_tool_names=RISK_SPECIALIST_TOOLS,
        force_escalation_precheck=True, use_mcp_tools=True,
    )


def _synthesize(ticket: str, specialist_results: dict[str, dict[str, Any]]) -> tuple[str, dict[str, int]]:
    reports = "\n\n".join(
        f"--- {name} report ---\n{result['answer']}" for name, result in specialist_results.items()
    )
    text, tokens = _plain_completion(SYNTHESIS_PROMPT.format(ticket=ticket, specialist_reports=reports))
    return text, tokens


def run_team(ticket: str) -> dict[str, Any]:
    started = time.perf_counter()
    total_tokens = prompt_tokens = completion_tokens = 0
    num_llm_calls = 0

    specialists, routing_tokens = _route(ticket)
    total_tokens += routing_tokens["total_tokens"]
    prompt_tokens += routing_tokens["prompt_tokens"]
    completion_tokens += routing_tokens["completion_tokens"]
    num_llm_calls += 1

    specialist_results: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for name in specialists:
        result = _run_specialist(name, ticket)
        specialist_results[name] = result
        total_tokens += result["total_tokens"]
        prompt_tokens += result["prompt_tokens"]
        completion_tokens += result["completion_tokens"]
        num_llm_calls += result["num_llm_calls"]
        sources.extend(result["sources"])

    answer, synthesis_tokens = _synthesize(ticket, specialist_results)
    total_tokens += synthesis_tokens["total_tokens"]
    prompt_tokens += synthesis_tokens["prompt_tokens"]
    completion_tokens += synthesis_tokens["completion_tokens"]
    num_llm_calls += 1

    elapsed = time.perf_counter() - started
    return {
        "ticket": ticket,
        "answer": answer,
        "sources": sources,
        "routing": specialists,
        "specialist_results": {
            name: {"answer": r["answer"], "num_llm_calls": r["num_llm_calls"], "sources": r["sources"]}
            for name, r in specialist_results.items()
        },
        "num_llm_calls": num_llm_calls,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": rag_core.estimate_cost_usd(prompt_tokens, completion_tokens, GENERATION_MODEL),
        "elapsed_s": round(elapsed, 3),
    }
