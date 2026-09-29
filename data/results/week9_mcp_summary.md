# Week 9 — MCP, multi-agent & A2A (Track A: customer support tickets)

"Bolt on the ticket-history server without touching the agent." One-page index for the mentor
check; see inline references for detail.

## Mentor-check checklist

**Does the agent use a tool through MCP, discovered rather than hard-coded?**
Yes. `agent_core.discover_mcp_tools()` calls `list_tools()` on every configured MCP server at the
start of every `run_agent()` call and builds that tool's Groq function-calling schema from
whatever comes back — `get_ticket_history` and `get_order_status` never appear as names anywhere
in `agent_core.py`'s own code, only in the server (`mcp_server/ticket_history_server.py`). Verified
live: `python -c "import agent_core; print(agent_core.discover_mcp_tools())"` prints the tool
schema fetched over HTTP from a separate running process.

**Can they add a second tool without changing the agent's code?**
Yes, and done in that literal order to prove it rather than assert it: `get_ticket_history` was
built and integration-tested first; `get_order_status` was added to
`mcp_server/ticket_history_server.py` afterward, as its own separate edit; `agent_core.py` was not
touched between the two. Restarting only the server process and re-running
`discover_mcp_tools()` picked up both tools automatically (`['get_order_status',
'get_ticket_history']`).

**Did they build their own server that another person's agent could call?**
Yes — `mcp_server/ticket_history_server.py`, running over streamable-HTTP
(`http://127.0.0.1:8931/mcp`), so it's reachable by any MCP client, not just a subprocess this
project spawns. `scripts/week9_external_client_demo.py` proves it: it never imports `agent_core`,
knows only the server's URL, and gets back the same structured data — real output below.

**Can they explain, in plain words, where the AI runs and where it doesn't?**
The AI (the Groq `chat.completions.create` call, the only place a model is invoked anywhere in
this project) runs exclusively inside `agent_core.py`'s process — the MCP "host." **The MCP server
process has no model in it at all.** `mcp_server/ticket_history_server.py` imports no LLM client,
sends no prompt, and makes no decision — it's a JSON lookup over `data/mock_ticket_history.json`.
It has no idea whether the caller is `agent_core.py`'s agent, `week9_external_client_demo.py`, or
literally anything else — proven concretely by that demo script getting identical data back with
zero knowledge of the agent's system prompt, retry logic, or tool schema.

## Architecture

```
agent_core.py (the "host" -- runs the LLM)              mcp_server/ticket_history_server.py
  run_agent()                                              (a separate OS process, HTTP :8931)
    |-- discover_mcp_tools()  --list_tools (MCP)-------->  no model, no prompt --
    |     (fresh every run)   <--tool schemas-------------  get_ticket_history
    |-- Groq chat.completions.create(tools=merged schema)   get_order_status
    |-- call_mcp_tool()       --call_tool (MCP)--------->  pure JSON lookup over
    |                         <--tool result--------------  data/mock_ticket_history.json
    v
  final_answer

scripts/week9_external_client_demo.py -- an independent fastmcp Client, no agent_core import,
talking to the same http://127.0.0.1:8931/mcp -- stands in for "someone else's agent."
```

## Real output

Tool discovery (both tools, after `get_order_status` was added as a separate later edit):
```
discovered tools: ['get_order_status', 'get_ticket_history']
```

Independent client (`scripts/week9_external_client_demo.py`), raw MCP protocol objects — genuinely
"looking at the raw messages once":
```json
{
  "name": "get_ticket_history",
  "description": "Look up a customer's past support tickets by email. ...",
  "input_schema": {"type": "object", "properties": {"customer_email": {"type": "string"}}, "required": ["customer_email"]}
}
```
```json
{
  "content": [{"type": "text", "text": "{\"customer_email\":\"priya@example.com\",\"past_tickets\":[{\"id\":\"TCK-1190\",...}],\"found\":true}"}],
  "structured_content": {"customer_email": "priya@example.com", "past_tickets": [...], "found": true},
  "is_error": false
}
```

Full agent, live, real ticket, real tool discovery, real answer:
```
Ticket: "A customer with email alice@example.com wants a summary of her past support tickets."
Steps: check_escalation (forced precheck) -> get_ticket_history(customer_email=alice@example.com) -> final_answer
Answer: "Here's a summary of your past support tickets:
1. Ticket ID TCK-1001 - Damaged item on delivery - opened 2026-06-02, resolved 2026-06-05.
2. Ticket ID TCK-1042 - Order cancellation request - opened and resolved 2026-07-14.
If you need more details or have any other questions, let me know!"
```

## Keeping it safe (this week's own framing)

Two access-control layers, neither an afterthought:
- **`MCP_SERVERS` is a fixed allowlist** (`agent_core.py`) — the agent only ever talks to a server
  named here, never one named in a ticket or a tool result. Adding a tool to an already-trusted
  server needs zero code changes here (see above); trusting a *new* server is a deliberate one-line
  edit to this list, not something discovery does automatically.
- **`_vet_mcp_tool` / `SUSPICIOUS_TOOL_METADATA_PATTERNS`**: before a discovered tool's schema is
  ever handed to the model, its own name and description are checked against a small pattern list
  ("ignore previous instructions", "also call", a request for bank/card details, etc.) — "MCP tool
  poisoning" (a hidden instruction in a tool's own metadata, not in its output) is a real, named
  attack class, and this is the direct analog of Week 8's `flag_suspicious_output`, aimed at the
  input side instead of the output side. A tool that matches is refused outright, not merely
  flagged — logged and never added to the schema at all.

## A real bug found and fixed, and one found and not fully fixed

**Fixed**: the model's response after a *compact* (no-whitespace) JSON tool result reliably
produced content Groq's serving layer couldn't parse (`BadRequestError`, `output_parse_failed`) —
100% reproducible for `get_ticket_history`, not the ~50% stochastic rate Week 8 found for
`search_kb`. Pretty-printing the tool result (`json.dumps(data, indent=2)` in `call_mcp_tool`)
fixed it completely, confirmed clean across three separate runs. Separately, a persistent
`BadRequestError` used to propagate uncaught past `LLM_CALL_RETRIES` and crash the whole ticket —
`run_agent` now folds a persistent parse failure into the same give-up-retry budget as an empty
turn, so it degrades to a safe "no answer" instead of crashing, matching Week 8's established
philosophy that a bad model turn should never take down the loop.

**Found, not fixed**: `get_order_status` reproduces essentially the same dead end 100% of the time
regardless of retries, independent of the JSON-null formatting hypothesis tested and ruled out (a
raw `null` for an undelivered order's `delivered` field was replaced with the string `"not yet
delivered"`; the failure persisted identically). The model's own `failed_generation` each time is a
single sentence of commentary ("Now we need to respond with final_answer citing the tool
result.") with no actual tool-call payload following it — the generation breaks off before
completing, a Groq/`openai/gpt-oss-20b` serving-side instability this project has no visibility
into or control over. `get_ticket_history` works reliably; `get_order_status` currently does not,
for reasons not fully isolated. This is reported rather than hidden — see below.

## What could still get through

- **`get_order_status` doesn't reliably produce a real answer.** The loop still stops safely (no
  crash, confirmed), but a ticket asking about order status specifically will very likely get the
  honest "no answer produced" fallback rather than a resolution. This is a genuine open reliability
  gap, not a design choice.
- **The tool-metadata vetting is a coarse regex net**, the same limitation Week 8 already stated for
  output validation — a differently-worded malicious tool description, with no matching keywords,
  would still load and reach the model as legitimate.
- **The local server has no auth.** `http://127.0.0.1:8931/mcp` accepts any request from anything
  that can reach localhost — fine for this training exercise, not for a real deployment. "Remote MCP
  & auth" is this week's own listed topic and is explicitly *not* implemented here, stated plainly
  rather than glossed over.
- **`MCP_SERVERS` is static.** Nothing in this project discovers *servers* dynamically — only tools
  on servers already in that fixed list. A real multi-team deployment would need a registry or
  config file, not a hardcoded Python list; that's a reasonable next step, not attempted here.

## Reproduce
```
python mcp_server/ticket_history_server.py        # separate terminal, leave running
python scripts/week9_external_client_demo.py       # "someone else's agent"
python -c "import agent_core; print(agent_core.discover_mcp_tools())"
```
