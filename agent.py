"""
agent.py
LangGraph stateful orchestration engine for Nykaa Domain Support Agent.
Implements 7 functional nodes, 2 conditional routing edges, strict schema enforcement,
PII redaction, intent classification, memory persistence, and grounded generation.
"""

import os
import re
import uuid
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

import guardrails
import memory
from generate import answer
from schema import validate_response
from tools import check_order_status

ORDER_ID_REGEX = re.compile(r"\b(ORD-\d{4})\b", re.IGNORECASE)
ORDER_INTENT_KEYWORDS = ["status", "where is my order", "track", "delayed", "is it delayed", "delivery time"]


class AgentState(TypedDict, total=False):
    query: str
    conversation_id: str
    trace_id: str
    masked_query: str
    history: List[Dict[str, Any]]
    last_order_id: Optional[str]
    extracted_order_id: Optional[str]
    route: str
    guardrails_triggered: List[str]
    result: Dict[str, Any]
    response: Dict[str, Any]


# ==========================================
# Graph Nodes
# ==========================================

def node_load_memory(state: AgentState) -> Dict[str, Any]:
    conv_id = state.get("conversation_id", "default")
    mem = memory.load(conv_id)
    return {
        "history": mem.get("messages", []),
        "last_order_id": mem.get("last_order_id"),
        "guardrails_triggered": [],
    }


def node_input_guard(state: AgentState) -> Dict[str, Any]:
    raw_query = state.get("query", "")
    triggered = list(state.get("guardrails_triggered", []))

    # 1. PII Masking
    masked, pii_findings = guardrails.mask_pii(raw_query)
    for f in pii_findings:
        triggered.append(f"PII_{f['type']}")

    # 2. Prompt Injection Detection
    is_inj, pattern_matched = guardrails.detect_injection(masked)
    if is_inj:
        triggered.append("PROMPT_INJECTION")
        return {
            "masked_query": masked,
            "route": "refused",
            "guardrails_triggered": triggered,
            "result": {
                "answer": "I cannot fulfill this request as it violates safety guidelines.",
                "sources": [],
                "grounded": False,
                "data": None,
            },
        }

    # 3. Near-Miss Corporate / Out-of-Scope Detection
    is_oos, matched_kw = guardrails.is_out_of_scope(masked)
    if is_oos:
        triggered.append("OUT_OF_SCOPE")
        return {
            "masked_query": masked,
            "route": "refused",
            "guardrails_triggered": triggered,
            "result": {
                "answer": "I don't know based on the Nykaa knowledge base.",
                "sources": [],
                "grounded": False,
                "data": None,
            },
        }

    return {
        "masked_query": masked,
        "guardrails_triggered": triggered,
    }


def node_classify_intent(state: AgentState) -> Dict[str, Any]:
    query_text = state.get("masked_query", "").strip()
    match = ORDER_ID_REGEX.search(query_text)

    if match:
        extracted = match.group(1).upper()
        return {
            "route": "order_status",
            "extracted_order_id": extracted,
            "last_order_id": extracted,
        }

    # Follow-up resolution using conversational memory
    q_lower = query_text.lower()
    has_order_keyword = any(kw in q_lower for kw in ORDER_INTENT_KEYWORDS)
    remembered_id = state.get("last_order_id")

    if has_order_keyword and remembered_id:
        return {
            "route": "order_status",
            "extracted_order_id": remembered_id,
        }

    return {
        "route": "rag",
        "extracted_order_id": None,
    }


def node_order_tool(state: AgentState) -> Dict[str, Any]:
    order_id = state.get("extracted_order_id")
    if not order_id:
        return {
            "result": {
                "answer": "Please provide a valid Nykaa order ID (e.g., ORD-0012) to check your status.",
                "sources": [],
                "grounded": False,
                "data": None,
            }
        }

    tool_res = check_order_status(order_id)
    if "error" in tool_res:
        answer_text = f"Order lookup failed: {tool_res['message']}"
    else:
        esc_str = " (Flagged for priority agent escalation)" if tool_res["escalate"] else ""
        answer_text = (
            f"Order {tool_res['record_id']} is currently '{tool_res['status']}'. "
            f"Order value: Rs {tool_res['order_value_inr']}. {tool_res['reason']}{esc_str}"
        )

    return {
        "result": {
            "answer": answer_text,
            "sources": [],
            "grounded": False,
            "data": tool_res,
        }
    }


def node_rag_tool(state: AgentState) -> Dict[str, Any]:
    query_text = state.get("masked_query", "")
    rag_res = answer(query_text, collection_name="kb_sentence")
    return {
        "result": {
            "answer": rag_res["answer"],
            "sources": rag_res["sources"],
            "grounded": rag_res["grounded"],
            "top_similarity": rag_res.get("top_similarity", 0.0), "top_chunk_text": rag_res.get("top_chunk_text", ""),
            "data": None,
        }
    }


def node_output_guard(state: AgentState) -> Dict[str, Any]:
    result = dict(state.get("result", {}))
    triggered = list(state.get("guardrails_triggered", []))

    is_grounded = guardrails.check_grounded(result, min_threshold=0.30)
    if not is_grounded:
        triggered.append("groundedness")
        result["answer"] = "I don't know based on the Nykaa knowledge base."
        result["grounded"] = False
        result["sources"] = []

    return {
        "result": result,
        "guardrails_triggered": triggered,
    }


def node_finalize(state: AgentState) -> Dict[str, Any]:
    trace_id = state.get("trace_id") or f"tr-{uuid.uuid4().hex[:12]}"
    route = state.get("route", "refused")
    result = state.get("result", {})
    triggered = sorted(list(set(state.get("guardrails_triggered", []))))

    response_payload = {
        "trace_id": trace_id,
        "route": route,
        "answer": result.get("answer", "I don't know based on the Nykaa knowledge base."),
        "sources": result.get("sources", []),
        "grounded": bool(result.get("grounded", False)),
        "guardrails_triggered": triggered,
        "data": result.get("data", None),
    }

    # Strict JSON Schema validation
    validate_response(response_payload)

    # Persist masked turn into JSON conversation store
    conv_id = state.get("conversation_id", "default")
    messages = list(state.get("history", []))
    messages.append({
        "user": state.get("masked_query", ""),
        "agent": response_payload["answer"],
        "route": route,
    })
    memory.save(conv_id, {
        "messages": messages,
        "last_order_id": state.get("last_order_id"),
    })

    return {"response": response_payload}


# ==========================================
# Conditional Edge Deciders
# ==========================================

def route_after_input_guard(state: AgentState) -> str:
    if state.get("route") == "refused":
        return "finalize"
    return "classify_intent"


def route_after_classify(state: AgentState) -> str:
    if state.get("route") == "order_status":
        return "order_tool"
    return "rag_tool"


# ==========================================
# Graph Assembly
# ==========================================

def build_graph(checkpointer=None):
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("load_memory", node_load_memory)
    workflow.add_node("input_guard", node_input_guard)
    workflow.add_node("classify_intent", node_classify_intent)
    workflow.add_node("order_tool", node_order_tool)
    workflow.add_node("rag_tool", node_rag_tool)
    workflow.add_node("output_guard", node_output_guard)
    workflow.add_node("finalize", node_finalize)

    # Wire edges
    workflow.add_edge(START, "load_memory")
    workflow.add_edge("load_memory", "input_guard")

    # Edge 1: Input guard conditional split
    workflow.add_conditional_edges(
        "input_guard",
        route_after_input_guard,
        {
            "finalize": "finalize",
            "classify_intent": "classify_intent",
        },
    )

    # Edge 2: Intent router conditional split
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "order_tool": "order_tool",
            "rag_tool": "rag_tool",
        },
    )

    workflow.add_edge("order_tool", "finalize")
    workflow.add_edge("rag_tool", "output_guard")
    workflow.add_edge("output_guard", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile(checkpointer=checkpointer)


# Singleton compiled graph instance
AGENT = build_graph()


def run_agent(query: str, conversation_id: str = "default", trace_id: Optional[str] = None) -> Dict[str, Any]:
    """Top-level invocation wrapper returning validated response payload."""
    initial_state: AgentState = {
        "query": query,
        "conversation_id": conversation_id,
        "trace_id": trace_id or f"tr-{uuid.uuid4().hex[:12]}",
    }
    final_state = AGENT.invoke(initial_state)
    return final_state["response"]


if __name__ == "__main__":
    resp = run_agent("How long do I have to return a lipstick?", conversation_id="test-cli")
    print("Agent CLI check:")
    print("  Route:   ", resp["route"])
    print("  Answer:  ", resp["answer"])
    print("  Sources: ", resp["sources"])

