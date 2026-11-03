"""
schema.py
Strict JSON schema definition and validator for agent responses.
Enforces typed contract across all routing paths (rag, order_status, refused).
"""

from typing import Any, Dict
import jsonschema
from jsonschema import validate, ValidationError

AGENT_RESPONSE_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "AgentResponse",
    "type": "object",
    "properties": {
        "trace_id": {
            "type": "string",
            "minLength": 1,
            "description": "Unique identifier for tracing request lifecycle",
        },
        "route": {
            "type": "string",
            "enum": ["rag", "order_status", "refused"],
            "description": "Selected routing decision taken by the graph",
        },
        "answer": {
            "type": "string",
            "description": "Customer-facing generated or tool-derived response text",
        },
        "sources": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Referenced policy document IDs",
        },
        "grounded": {
            "type": "boolean",
            "description": "Whether the response is strictly grounded in retrieved KB context",
        },
        "guardrails_triggered": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of guardrails activated during execution",
        },
        "data": {
            "type": ["object", "null"],
            "description": "Structured payload returned by tool execution",
        },
    },
    "required": [
        "trace_id",
        "route",
        "answer",
        "sources",
        "grounded",
        "guardrails_triggered",
        "data",
    ],
    "additionalProperties": False,
}


def validate_response(response_dict: Dict[str, Any]) -> None:
    """
    Validates dictionary against AGENT_RESPONSE_SCHEMA.
    Raises jsonschema.ValidationError if response does not strictly adhere.
    """
    validate(instance=response_dict, schema=AGENT_RESPONSE_SCHEMA)


if __name__ == "__main__":
    valid_sample = {
        "trace_id": "tr-test-1234",
        "route": "rag",
        "answer": "Beauty items have a 5-day return window.",
        "sources": ["return_window"],
        "grounded": True,
        "guardrails_triggered": [],
        "data": None,
    }
    validate_response(valid_sample)
    print("Schema validation test: PASS")
