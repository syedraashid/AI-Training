"""Week 9: the Track A MCP server -- "bolt on the ticket-history server without touching the
agent." A small, standalone service that exposes one real capability of this project (looking
up a customer's past support tickets) over the Model Context Protocol, using fastmcp.

This process has no model in it at all -- no Groq client, no prompt, nothing that "thinks." It
is pure data lookup over a fixed mock dataset (data/mock_ticket_history.json, fictional
Northstar Home customers). That's the point this week's brief makes about where the AI runs:
never here. This server has no idea whether the caller is agent_core.py's agent, the standalone
demo client in scripts/week9_external_client_demo.py, or literally any other MCP client -- it
answers the same way regardless, which is exactly what makes it reusable.

Runs over streamable-HTTP so a genuinely separate process (this project's own agent, or
"someone else's agent") can reach it over the network, not just as a spawned subprocess --
the realistic shape of a deployed MCP server, even at localhost.

Usage:
    python mcp_server/ticket_history_server.py
    (leave running in its own terminal; agent_core.py and the demo client both connect to
    http://127.0.0.1:8931/mcp)
"""

import json
from pathlib import Path

from fastmcp import FastMCP

ROOT = Path(__file__).parent.parent
TICKET_HISTORY_PATH = ROOT / "data" / "mock_ticket_history.json"
HOST = "127.0.0.1"
PORT = 8931

mcp = FastMCP(
    "northstar-ticket-history",
    instructions=(
        "Read-only lookup over Northstar Home's (fictional) support ticket and order history. "
        "Does not modify any records and does not decide anything -- the calling agent "
        "interprets the returned data."
    ),
)


def _load() -> dict:
    return json.loads(TICKET_HISTORY_PATH.read_text(encoding="utf-8"))


def _no_nulls(record: dict) -> dict:
    """Same fix as get_order_status's clean_order, applied generically -- see that function's
    comment for why bare JSON null in a tool result was worth designing away here."""
    return {k: (v if v is not None else "not yet closed") for k, v in record.items()}


@mcp.tool
def get_ticket_history(customer_email: str) -> dict:
    """Look up a customer's past support tickets by email. Returns each ticket's id, topic,
    status, and open/close dates. Returns an empty list if the email has no history on file --
    that is not an error, just no prior tickets."""
    customer = _load()["customers"].get(customer_email.strip().lower())
    if not customer:
        return {"customer_email": customer_email, "past_tickets": [], "found": False}
    tickets = [_no_nulls(t) for t in customer["past_tickets"]]
    return {"customer_email": customer_email, "past_tickets": tickets, "found": True}


# Added after the agent-side MCP integration (agent_core.py) was already built and working
# against get_ticket_history alone -- this is the "add a second tool without touching the
# agent's code" proof from this week's mentor check. agent_core.py has not changed since: it
# discovers whatever tools this server currently exposes, every run, by calling list_tools --
# it never enumerates get_ticket_history or get_order_status by name.
@mcp.tool
def get_order_status(order_id: str) -> dict:
    """Look up the shipping status of one order by its order id (e.g. 'ORD-5521'). Returns
    not-found if the order id doesn't match anything on file."""
    for customer in _load()["customers"].values():
        for order in customer["orders"]:
            if order["order_id"].strip().lower() == order_id.strip().lower():
                # "delivered" is only meaningful once it's happened -- return a plain string
                # rather than a bare JSON null for "hasn't yet", not because null is invalid
                # MCP output, but because a null literal in this tool's result was traced (see
                # data/results/week9_mcp_summary.md) to a genuinely different downstream failure
                # than the compact-JSON one call_mcp_tool's indent=2 already fixes: the model's
                # own generation broke off mid-turn with no completed tool call at all. Shaping
                # the data so nothing "hasn't happened yet" is represented as a bare null avoids
                # it, and is arguably better API design regardless.
                clean_order = {k: (v if v is not None else "not yet delivered") for k, v in order.items()}
                return {"order_id": order_id, "found": True, **clean_order}
    return {"order_id": order_id, "found": False}


if __name__ == "__main__":
    mcp.run(transport="http", host=HOST, port=PORT)
