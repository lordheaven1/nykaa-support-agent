"""
mcp_client.py
Asynchronous client communicating with the Nykaa MCP Server via HTTP protocol.
Verifies tool introspection (list_tools) and executes 3 test scenarios:
1. Delayed open order (ORD-0008)
2. Delivered order (ORD-0001)
3. Nonexistent order (ORD-9999) - demonstrates error dictionary propagation
"""

import asyncio
import json
from fastmcp import Client


async def main():
    server_url = "http://127.0.0.1:8000/mcp"
    print("=" * 80)
    print(f"CONNECTING TO MCP SERVER AT: {server_url}")
    print("=" * 80)

    async with Client(server_url) as client:
        # 1. Tool Introspection
        print("\n--- [1] LIST TOOLS ---")
        tools_list = await client.list_tools()
        for t in tools_list:
            print(f"Tool Name:        {t.name}")
            print(f"Tool Description: {t.description.strip()}")
            if hasattr(t, "parameters"):
                print(f"Tool Schema:      {json.dumps(t.parameters, indent=2)}")

        # 2. Call Tool: Delayed open order
        print("\n--- [2] CALL TOOL: Delayed Open Order (ORD-0008) ---")
        res_delayed = await client.call_tool("check_order_status", {"record_id": "ORD-0008"})
        print("MCP Raw Result Type:", type(res_delayed).__name__)
        print("MCP Result Representation:", repr(res_delayed))
        if hasattr(res_delayed, "data"):
            print("Decoded Data:", json.dumps(res_delayed.data, indent=2))
        elif hasattr(res_delayed, "structured_content"):
            print("Structured Content:", json.dumps(res_delayed.structured_content, indent=2))
        elif hasattr(res_delayed, "content"):
            print("Content:", res_delayed.content)

        # 3. Call Tool: Delivered resolved order
        print("\n--- [3] CALL TOOL: Delivered Order (ORD-0001) ---")
        res_delivered = await client.call_tool("check_order_status", {"record_id": "ORD-0001"})
        print("MCP Raw Result Type:", type(res_delivered).__name__)
        print("MCP Result Representation:", repr(res_delivered))
        if hasattr(res_delivered, "data"):
            print("Decoded Data:", json.dumps(res_delivered.data, indent=2))
        elif hasattr(res_delivered, "structured_content"):
            print("Structured Content:", json.dumps(res_delivered.structured_content, indent=2))
        elif hasattr(res_delivered, "content"):
            print("Content:", res_delivered.content)

        # 4. Call Tool: Nonexistent order (Protocol resilience check)
        print("\n--- [4] CALL TOOL: Nonexistent Order (ORD-9999) ---")
        res_nonexistent = await client.call_tool("check_order_status", {"record_id": "ORD-9999"})
        print("MCP Raw Result Type:", type(res_nonexistent).__name__)
        print("MCP Result Representation:", repr(res_nonexistent))
        if hasattr(res_nonexistent, "data"):
            print("Decoded Data:", json.dumps(res_nonexistent.data, indent=2))
        elif hasattr(res_nonexistent, "structured_content"):
            print("Structured Content:", json.dumps(res_nonexistent.structured_content, indent=2))
        elif hasattr(res_nonexistent, "content"):
            print("Content:", res_nonexistent.content)

    print("\n" + "=" * 80)
    print("ALL MCP TOOL CALLS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
