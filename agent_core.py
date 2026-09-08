"""Week 7: a hand-built ReAct-style agent loop for resolving support tickets that may
touch more than one knowledge-base article. Uses the model's native tool-calling (Groq's
gpt-oss-20b rejects free-text "pretend tool calls" outside that API), but the loop itself
-- deciding when to stop, executing tools, logging every step -- is hand-rolled, no agent
framework. Every step is logged to data/agent_traces.jsonl so the loop is never a black box.

Loop: think -> pick one tool -> read its result -> repeat until final_answer or the
step budget runs out (a safe stop, not an infinite loop).
"""

from __future__ import annotations

import json
import time
from typing import Any

import rag_core

MAX_STEPS = 4
AGENT_TRACES_PATH = rag_core.DATA_DIR / "agent_traces.jsonl"

# Kept deliberately small and rule-based (not another LLM call) so it's cheap, deterministic,
# and testable on its own -- a good contrast to the semantic search_kb tool.
ESCALATION_RULES = {
    "threat": "threats",
    "threatening": "threats",
    "fraud": "suspected fraud",
    "chargeback": "a chargeback",
    "lawsuit": "legal correspondence",
    "sue": "legal correspondence",
    "legal action": "legal correspondence",
    "attorney": "legal correspondence",
    "delete my data": "a customer requesting deletion of personal data",
    "delete my account": "a customer requesting deletion of personal data",
    "gdpr": "a customer requesting deletion of personal data",
}

SYSTEM_PROMPT = f"""You are a support-ticket resolution agent for Northstar Home. You work in a
loop: think, call exactly one tool, read its result, and repeat until you have everything the
ticket needs -- then call final_answer. Never answer from memory; every fact must come from a
tool result you were given in this conversation.

Call search_kb once per distinct topic the ticket raises -- do not call it twice for the same
topic. Only call check_escalation if the ticket actually describes a threat, suspected fraud, a
chargeback, legal correspondence, or a data-deletion request.

final_answer must cite every article page a tool returned, in the format [filename | page/section].
If no tool result addresses the ticket at all, final_answer's text must be exactly: {rag_core.REFUSAL_TEXT}"""

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": (
                "Semantic+keyword search over the Northstar Home support knowledge base. "
                "Returns the single best-matching article page for one topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A short query about ONE topic, e.g. 'warranty coverage' or 'order cancellation window'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_escalation",
            "description": (
                "Fixed rule check (not a search) for whether a described situation must be "
                "escalated per policy: threats, suspected fraud, a chargeback, legal "
                "correspondence, or a data-deletion request."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "situation": {"type": "string", "description": "A one-sentence description of the situation."}
                },
                "required": ["situation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Ends the task and returns the reply to give the support agent.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "The final reply, citing every source used."}},
                "required": ["text"],
            },
        },
    },
]


def check_escalation(situation: str) -> str:
    hits = sorted({reason for kw, reason in ESCALATION_RULES.items() if kw in situation.lower()})
    if hits:
        return f"ESCALATE: matches escalation policy ({', '.join(hits)}). [Source: Escalation guidance]"
    return "No escalation trigger matched for this situation. [Source: Escalation guidance]"


def search_kb(query: str) -> tuple[str, dict[str, Any] | None]:
    results, _ = rag_core.retrieve(query, top_k=1, mode="hybrid")
    if not results:
        return "No matching article found.", None
    chunk = results[0]
    return f"[{chunk['source']} | {chunk['location']}]\n{chunk['text']}", chunk


def run_agent(
    ticket: str, generation_model: str = "openai/gpt-oss-20b", max_steps: int = MAX_STEPS
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Ticket: {ticket}"},
    ]
    steps: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    total_tokens = 0
    started = time.perf_counter()
    answer = None

    for step_num in range(1, max_steps + 1):
        response = rag_core.client().chat.completions.create(
            model=generation_model, messages=messages, tools=TOOLS_SCHEMA, tool_choice="auto", temperature=0
        )
        total_tokens += response.usage.total_tokens if response.usage else 0
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            answer = message.content or ""
            steps.append({"step": step_num, "action": "final_answer", "action_input": answer, "observation": None})
            break

        call = message.tool_calls[0]
        try:
            args = json.loads(call.function.arguments)
        except json.JSONDecodeError:
            args = {}
        name = call.function.name

        if name == "final_answer":
            answer = args.get("text", "")
            steps.append({"step": step_num, "action": name, "action_input": args, "observation": None})
            break
        elif name == "search_kb":
            observation, chunk = search_kb(args.get("query", ""))
            if chunk:
                sources.append(chunk)
        elif name == "check_escalation":
            observation = check_escalation(args.get("situation", ""))
        else:
            observation = f"Unknown tool '{name}'."

        steps.append({"step": step_num, "action": name, "action_input": args, "observation": observation})
        messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})

    stopped_safely = answer is not None
    if answer is None:
        answer = "Step budget reached before a final answer was produced -- stopping safely rather than looping further."

    elapsed = time.perf_counter() - started
    result = {
        "ticket": ticket,
        "answer": answer,
        "steps": steps,
        "sources": [{"source": s["source"], "location": s["location"]} for s in sources],
        "num_llm_calls": len(steps) if stopped_safely else max_steps,
        "total_tokens": total_tokens,
        "elapsed_s": round(elapsed, 3),
        "stopped_safely": stopped_safely,
    }
    _log_trace(result)
    return result


def _log_trace(result: dict[str, Any]) -> None:
    rag_core.DATA_DIR.mkdir(exist_ok=True)
    with AGENT_TRACES_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(result) + "\n")
