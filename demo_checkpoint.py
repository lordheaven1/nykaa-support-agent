"""
demo_checkpoint.py
Two-stage demonstration of LangGraph state checkpointing and cross-process resumption.
Uses SqliteSaver backed by checkpoints.sqlite with interrupt_before=["rag_tool"].

Usage:
    python demo_checkpoint.py stage1
    python demo_checkpoint.py stage2
"""

import json
import os
import sqlite3
import sys
from typing import Any, Callable, Dict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

# Import domain components with exact node names
from agent import (
    AgentState,
    node_classify_intent,
    node_finalize,
    node_input_guard,
    node_load_memory,
    node_order_tool,
    node_output_guard,
    node_rag_tool,
    route_after_classify,
    route_after_input_guard,
)


# Node execution logger
def track_exec(name: str, fn: Callable) -> Callable:
    def wrapped(state: AgentState) -> Dict[str, Any]:
        print(f"[EXEC] {name}")
        return fn(state)
    return wrapped


def build_checkpointed_graph(checkpointer: SqliteSaver):
    """Builds the agent state machine with [EXEC] wrappers and interrupt_before."""
    workflow = StateGraph(AgentState)

    workflow.add_node("load_memory", track_exec("load_memory", node_load_memory))
    workflow.add_node("input_guard", track_exec("input_guard", node_input_guard))
    workflow.add_node("classify_intent", track_exec("classify_intent", node_classify_intent))
    workflow.add_node("order_tool", track_exec("order_tool", node_order_tool))
    workflow.add_node("rag_tool", track_exec("rag_tool", node_rag_tool))
    workflow.add_node("output_guard", track_exec("output_guard", node_output_guard))
    workflow.add_node("finalize", track_exec("finalize", node_finalize))

    workflow.set_entry_point("load_memory")
    workflow.add_edge("load_memory", "input_guard")

    workflow.add_conditional_edges(
        "input_guard",
        route_after_input_guard,
        {"refused": "finalize", "classify_intent": "classify_intent"},
    )

    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {"order_tool": "order_tool", "rag_tool": "rag_tool"},
    )

    workflow.add_edge("order_tool", "finalize")
    workflow.add_edge("rag_tool", "output_guard")
    workflow.add_edge("output_guard", "finalize")
    workflow.add_edge("finalize", END)

    # Interrupt right before rag_tool
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["rag_tool"],
    )


THREAD_ID = "ckpt-demo-01"
CONV_ID = "conv-ckpt-demo-01"
DB_PATH = "checkpoints.sqlite"
MEMORY_PATH = "memory_store.json"


def run_stage1():
    print("=" * 80)
    print("TASK 15 - STAGE 1: EXECUTION UNTIL INTERRUPT")
    print(f"Target Thread ID: {THREAD_ID}")
    print("=" * 80)

    # Clean existing SQLite checkpoints file for a fresh demo run
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            print(f"Cleaned stale {DB_PATH}")
        except Exception:
            pass

    # Clean prior demo conversation entry in memory_store.json before starting
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            store = json.load(f)
        if CONV_ID in store:
            del store[CONV_ID]
            with open(MEMORY_PATH, "w", encoding="utf-8") as f:
                json.dump(store, f, indent=2)

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    saver = SqliteSaver(conn)
    app = build_checkpointed_graph(saver)

    initial_state: AgentState = {
        "query": "How long do I have to return an unopened face cream?",
        "conversation_id": CONV_ID,
        "trace_id": "tr-ckpt-demo-01",
        "masked_query": "",
        "history": [],
        "last_order_id": None,
        "extracted_order_id": None,
        "route": "rag",
        "guardrails_triggered": [],
        "result": None,
        "response": None,
    }

    config = {"configurable": {"thread_id": THREAD_ID}}

    print("\n--- Starting Graph Invocation (Stage 1) ---")
    app.invoke(initial_state, config=config)

    print("\n--- Inspecting Interrupted State from Checkpointer ---")
    snapshot = app.get_state(config)
    print(f"Next Node in Execution Queue: {snapshot.next}")
    print(f"Interrupted Route:             {snapshot.values.get('route')}")
    print(f"Masked Query in Checkpoint:    {snapshot.values.get('masked_query')}")
    print(f"Last Order ID in Checkpoint:   {snapshot.values.get('last_order_id')}")
    print(f"Trace ID in Checkpoint:        {snapshot.values.get('trace_id')}")

    assert snapshot.next == ("rag_tool",), f"Expected ('rag_tool',), got {snapshot.next}"
    print("\nStage 1 successfully interrupted before 'rag_tool'. Process exiting cleanly.")
    conn.close()


def run_stage2():
    print("=" * 80)
    print("TASK 15 - STAGE 2: PROCESS RESUMPTION FROM DISK")
    print(f"Connecting to: {DB_PATH} with Thread ID: {THREAD_ID}")
    print("=" * 80)

    assert os.path.exists(DB_PATH), f"Error: {DB_PATH} not found!"

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    saver = SqliteSaver(conn)
    app = build_checkpointed_graph(saver)

    config = {"configurable": {"thread_id": THREAD_ID}}

    print("\n--- Reading Checkpoint from Disk Before Resuming ---")
    snapshot = app.get_state(config)
    print(f"Loaded 'next' from DB:       {snapshot.next}")
    print(f"Loaded 'route' from DB:      {snapshot.values.get('route')}")
    print(f"Loaded 'query' from DB:      {snapshot.values.get('masked_query')}")

    assert snapshot.next == ("rag_tool",), f"Expected ('rag_tool',), got {snapshot.next}"

    print("\n--- Resuming Graph Invocation (Stage 2) with invoke(None, config) ---")
    final_state = app.invoke(None, config=config)

    print("\n--- Final State Validation ---")
    print(f"Final Route:                 {final_state.get('route')}")
    print(f"Grounded:                    {final_state.get('result', {}).get('grounded')}")
    print(f"Sources:                     {final_state.get('result', {}).get('sources')}")
    print(f"Answer:                      {final_state.get('response', {}).get('answer')}")

    # Inspect SQLite database directly and print individual rows
    print("\n--- Direct SQLite Checkpoint Audit ---")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT thread_id, checkpoint_id, parent_checkpoint_id FROM checkpoints WHERE thread_id = ? ORDER BY rowid ASC",
        (THREAD_ID,),
    )
    rows = cursor.fetchall()
    print(f"Total Checkpoint Snapshots for thread '{THREAD_ID}': {len(rows)}")
    for idx, (t_id, ckpt_id, parent_id) in enumerate(rows, 1):
        print(f"  Row {idx}: thread_id={t_id} | checkpoint_id={ckpt_id} | parent_id={parent_id}")

    # Verify memory_store.json has exactly 1 entry for CONV_ID
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        store = json.load(f)
    turns = store.get(CONV_ID, {}).get("messages", [])
    print(f"\nMemory Store Audit: conversation '{CONV_ID}' has {len(turns)} saved turn(s) (expected: 1).")
    assert len(turns) == 1, f"Expected 1 turn in memory_store.json, found {len(turns)}"

    conn.close()
    print("\nStage 2 completed successfully: remaining nodes executed without early node replay.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("stage1", "stage2"):
        print("Usage: python demo_checkpoint.py [stage1|stage2]")
        sys.exit(1)

    if sys.argv[1] == "stage1":
        run_stage1()
    else:
        run_stage2()
