"""
resilience.py
Resilience harness demonstrating:
1. Exponential backoff retries with jitter via LangGraph RetryPolicy (Task 16a)
2. Per-node timeouts via async cancellation decorator (Task 16b)
3. Global pipeline timeouts via asyncio.wait_for (Task 16c)

All nodes are asynchronous to guarantee genuine coroutine cancellation
without thread-pool hanging or unkillable worker loops.
"""

import asyncio
import time
from typing import Any, Callable, Dict, Optional, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import RetryPolicy

# Real domain tool import
import tools


# ==========================================
# 1. Custom Exceptions & Fault Injector
# ==========================================

class TransientError(Exception):
    """Custom exception representing recoverable network or upstream tool failure."""
    pass


class NodeTimeoutError(Exception):
    """Raised when an individual node exceeds its per-node execution timeout."""
    pass


class FaultInjector:
    """Injects TransientError for the first n calls, then succeeds."""
    def __init__(self, fail_first_n: int = 0):
        self.fail_first_n = fail_first_n
        self.call_count = 0
        self.call_timestamps = []

    def reset(self, fail_first_n: int):
        self.fail_first_n = fail_first_n
        self.call_count = 0
        self.call_timestamps = []

    def record_and_check(self, node_name: str):
        self.call_count += 1
        now = time.perf_counter()
        self.call_timestamps.append(now)
        elapsed_since_first = now - self.call_timestamps[0] if len(self.call_timestamps) > 1 else 0.0

        if self.call_count <= self.fail_first_n:
            print(f"[{time.strftime('%H:%M:%S')}] [ATTEMPT {self.call_count}] {node_name} FAILING (TransientError injected, elapsed={elapsed_since_first:.3f}s)")
            raise TransientError(f"Simulated upstream outage on attempt {self.call_count}")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] [ATTEMPT {self.call_count}] {node_name} SUCCEEDED (elapsed={elapsed_since_first:.3f}s)")


# Global injector instance
injector = FaultInjector()


# ==========================================
# 2. Per-Node Timeout Decorator
# ==========================================

def with_timeout(seconds: float):
    """Wraps an async node coroutine with asyncio.wait_for, raising NodeTimeoutError on breach."""
    def decorator(fn: Callable):
        async def wrapped(state: "ResilienceState") -> Dict[str, Any]:
            t0 = time.perf_counter()
            try:
                return await asyncio.wait_for(fn(state), timeout=seconds)
            except asyncio.TimeoutError:
                dur = time.perf_counter() - t0
                print(f"[{time.strftime('%H:%M:%S')}] [TIMEOUT] Node '{fn.__name__}' timed out after {dur:.3f}s (limit: {seconds:.1f}s)")
                raise NodeTimeoutError(f"Node '{fn.__name__}' exceeded {seconds:.1f}s timeout limit")
        return wrapped
    return decorator


# ==========================================
# 3. State & Node Definitions
# ==========================================

class ResilienceState(TypedDict):
    order_id: str
    result: Optional[Dict[str, Any]]
    steps_executed: list


async def node_prepare(state: ResilienceState) -> Dict[str, Any]:
    print(f"[{time.strftime('%H:%M:%S')}] [EXEC] prepare")
    steps = list(state.get("steps_executed", []))
    steps.append("prepare")
    return {"steps_executed": steps}


async def node_flaky_lookup(state: ResilienceState) -> Dict[str, Any]:
    print(f"[{time.strftime('%H:%M:%S')}] [EXEC] flaky_lookup")
    injector.record_and_check("flaky_lookup")
    # Executes real domain order status tool
    order_info = tools.check_order_status(state["order_id"])
    steps = list(state.get("steps_executed", []))
    steps.append("flaky_lookup")
    return {"result": order_info, "steps_executed": steps}


# Deliberately slow node for timeout tests
async def node_slow_step(state: ResilienceState) -> Dict[str, Any]:
    print(f"[{time.strftime('%H:%M:%S')}] [EXEC] slow_step: sleeping for 5.0s...")
    await asyncio.sleep(5.0)
    steps = list(state.get("steps_executed", []))
    steps.append("slow_step")
    return {"steps_executed": steps}


# Chained nodes for global timeout tests (each node takes ~0.8s)
async def node_chain_1(state: ResilienceState) -> Dict[str, Any]:
    print(f"[{time.strftime('%H:%M:%S')}] [EXEC] chain_node_1 (takes ~0.8s)")
    await asyncio.sleep(0.8)
    return {"steps_executed": state.get("steps_executed", []) + ["chain_1"]}

async def node_chain_2(state: ResilienceState) -> Dict[str, Any]:
    print(f"[{time.strftime('%H:%M:%S')}] [EXEC] chain_node_2 (takes ~0.8s)")
    await asyncio.sleep(0.8)
    return {"steps_executed": state.get("steps_executed", []) + ["chain_2"]}

async def node_chain_3(state: ResilienceState) -> Dict[str, Any]:
    print(f"[{time.strftime('%H:%M:%S')}] [EXEC] chain_node_3 (takes ~0.8s)")
    await asyncio.sleep(0.8)
    return {"steps_executed": state.get("steps_executed", []) + ["chain_3"]}

async def node_chain_4(state: ResilienceState) -> Dict[str, Any]:
    print(f"[{time.strftime('%H:%M:%S')}] [EXEC] chain_node_4 (takes ~0.8s)")
    await asyncio.sleep(0.8)
    return {"steps_executed": state.get("steps_executed", []) + ["chain_4"]}


async def node_finalize(state: ResilienceState) -> Dict[str, Any]:
    print(f"[{time.strftime('%H:%M:%S')}] [EXEC] finalize")
    steps = list(state.get("steps_executed", []))
    steps.append("finalize")
    return {"steps_executed": steps}


# ==========================================
# 4. Graph Builders
# ==========================================

# Task 16a: Exponential Backoff Policy
RETRY_POLICY = RetryPolicy(
    initial_interval=0.2,   # 200ms initial gap
    backoff_factor=2.0,     # doubles each retry: 0.2 -> 0.4 -> 0.8
    max_interval=2.0,       # interval cap
    max_attempts=4,         # 1 initial + up to 3 retries
    jitter=True,            # randomized interval jitter
    retry_on=TransientError # strictly retries transient errors
)


def build_retry_graph():
    wf = StateGraph(ResilienceState)
    wf.add_node("prepare", node_prepare)
    wf.add_node("flaky_lookup", node_flaky_lookup, retry_policy=RETRY_POLICY)
    wf.add_node("finalize", node_finalize)

    wf.set_entry_point("prepare")
    wf.add_edge("prepare", "flaky_lookup")
    wf.add_edge("flaky_lookup", "finalize")
    wf.add_edge("finalize", END)
    return wf.compile()


def build_node_timeout_graph():
    wf = StateGraph(ResilienceState)
    wf.add_node("prepare", node_prepare)
    # Per-node timeout set to 1.0s on a node that tries to sleep for 5.0s
    wf.add_node("slow_step", with_timeout(1.0)(node_slow_step))
    wf.add_node("finalize", node_finalize)

    wf.set_entry_point("prepare")
    wf.add_edge("prepare", "slow_step")
    wf.add_edge("slow_step", "finalize")
    wf.add_edge("finalize", END)
    return wf.compile()


def build_global_timeout_graph():
    wf = StateGraph(ResilienceState)
    wf.add_node("chain_1", with_timeout(1.0)(node_chain_1))
    wf.add_node("chain_2", with_timeout(1.0)(node_chain_2))
    wf.add_node("chain_3", with_timeout(1.0)(node_chain_3))
    wf.add_node("chain_4", with_timeout(1.0)(node_chain_4))

    wf.set_entry_point("chain_1")
    wf.add_edge("chain_1", "chain_2")
    wf.add_edge("chain_2", "chain_3")
    wf.add_edge("chain_3", "chain_4")
    wf.add_edge("chain_4", END)
    return wf.compile()
