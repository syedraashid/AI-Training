"""Week 9: "have someone else's agent call it." This script deliberately does NOT import
agent_core -- it's an independent MCP client, standing in for a different team's agent that
knows nothing about this project except the server's URL. If this script gets the same answers
agent_core.py does, that's the proof the ticket-history server is a genuine, reusable capability
and not something secretly coupled to our own agent's code.

Also satisfies "looking at the raw messages once, so MCP stops being a mystery" -- the *_mcp
variants of fastmcp's client calls (list_tools_mcp, call_tool_mcp) return the raw MCP protocol
objects (the same JSON-RPC-shaped data that crosses the wire), not fastmcp's convenience
wrapper, and this script dumps them.

Usage (with mcp_server/ticket_history_server.py already running separately):
    python scripts/week9_external_client_demo.py
"""

import asyncio
import json

from fastmcp import Client

SERVER_URL = "http://127.0.0.1:8931/mcp"


def dump(label: str, obj) -> None:
    print(f"--- {label} ---")
    if hasattr(obj, "model_dump"):
        print(json.dumps(obj.model_dump(mode="json", exclude_none=True), indent=2))
    else:
        print(json.dumps([o.model_dump(mode="json", exclude_none=True) for o in obj], indent=2))
    print()


async def main() -> None:
    async with Client(SERVER_URL) as client:
        # Raw protocol-level list_tools -- what actually crosses the wire, not fastmcp's
        # convenience-wrapped list_tools().
        tools_result = await client.list_tools_mcp()
        dump("raw list_tools response (JSON-RPC 'tools' field)", tools_result.tools)

        print("Calling get_ticket_history for a known customer (priya@example.com):")
        result = await client.call_tool_mcp("get_ticket_history", {"customer_email": "priya@example.com"})
        dump("raw call_tool response", result)

        print("Calling get_order_status for an order that doesn't exist (ORD-0000):")
        result2 = await client.call_tool_mcp("get_order_status", {"order_id": "ORD-0000"})
        dump("raw call_tool response (not found)", result2)

        print(
            "This client never imported agent_core.py and knows nothing about the ticket agent's\n"
            "system prompt, tool schema merging, or retry logic -- it only knows this URL. Getting\n"
            "correct, identical-shaped data back is the proof the server is independently callable,\n"
            "exactly as this week's brief asks: 'build a small MCP server ... and have someone\n"
            "else's agent call it.'"
        )


if __name__ == "__main__":
    asyncio.run(main())
