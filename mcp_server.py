"""
mcp_server.py
Model Context Protocol (MCP) server exposing Nykaa customer support tools.
Exposes check_order_status tool over HTTP SSE at http://127.0.0.1:8000/mcp.
"""

from typing import Any, Dict
from fastmcp import FastMCP
import tools

# Initialize MCP Server instance
mcp = FastMCP("nykaa-order-status")


@mcp.tool(name="check_order_status")
def check_order_status(record_id: str) -> Dict[str, Any]:
    """Look up real-time status and escalation priority for a Nykaa customer order.

    Args:
        record_id (str): The unique order identifier formatted as 'ORD-XXXX'
            (e.g., 'ORD-0012'). Case-insensitive with automatic whitespace trimming.

    Returns:
        dict: A structured dictionary containing:
            - record_id (str): Normalized order identifier.
            - status (str): Current status ('Placed', 'Shipped', 'Delivered', 'Returned', 'Refunded').
            - order_value_inr (int): Order value in Indian Rupees.
            - delayed_shipment (bool): Whether shipment SLA was breached.
            - days_since_created (int): Age of order in days [0-30].
            - escalation_score (float): Calculated urgency score [0.0 to 1.0].
            - escalate (bool): True if escalation_score >= 0.3733 (80th percentile).
            - reason (str): Human-readable dispatch and escalation rationale.

        If the record_id does not exist in the dataset, returns an error dictionary
        with 'error': True, 'record_id', and a helpful diagnostic 'message'.
    """
    return tools.check_order_status(record_id)


if __name__ == "__main__":
    # Server listens at http://127.0.0.1:8000/mcp
    print("Starting Nykaa MCP Server on http://127.0.0.1:8000/mcp ...")
    mcp.run(transport="http", host="127.0.0.1", port=8000)
