"""
demo_resilience.py
Runs the three Task 16 verification scenarios:
1. Task 16a: Exponential backoff retries (success on attempt 3, then exhausted failure)
2. Task 16b: Per-node timeout (1.0s timeout cancels 5.0s sleep)
3. Task 16c: Global timeout (2.0s workflow limit cancels multi-node chain)
"""

import asyncio
import sys
import time

from resilience import (
    NodeTimeoutError,
    RETRY_POLICY,
    TransientError,
    build_global_timeout_graph,
    build_node_timeout_graph,
    build_retry_graph,
    injector,
)


async def run_task16a_retry():
    print("=" * 80)
    print("TASK 16a: EXPONENTIAL BACKOFF RETRY VERIFICATION")
    print("=" * 80)
    print("Policy Configuration:")
    print(f"  - initial_interval: {RETRY_POLICY.initial_interval}s")
    print(f"  - backoff_factor:   {RETRY_POLICY.backoff_factor}")
    print(f"  - max_interval:     {RETRY_POLICY.max_interval}s")
    print(f"  - max_attempts:     {RETRY_POLICY.max_attempts}")
    print(f"  - jitter:           {RETRY_POLICY.jitter}")
    print(f"  - retry_on:         {RETRY_POLICY.retry_on.__name__}")
    print("-" * 80)

    app = build_retry_graph()

    # Scenario 1: Recovers on Attempt 3
    print("\n--- Scenario 1: Node Fails 2 Times -> Recovers on Attempt 3 ---")
    injector.reset(fail_first_n=2)
    t0 = time.perf_counter()
    state = await app.ainvoke({"order_id": "ORD-0001", "result": None, "steps_executed": []})
    dur = time.perf_counter() - t0
    print(f"\nResult: SUCCESS after {injector.call_count} attempts (Total duration: {dur:.3f}s)")
    print(f"Resolved Order Status: {state['result']['status']}, Escalation: {state['result']['escalation_score']}")
    print(f"Steps Completed: {state['steps_executed']}")

    # Scenario 2: Exhausted Retries (Fails 5 times with max_attempts=4)
    print("\n--- Scenario 2: Retries Exhausted (Node Fails 5 Times, Max Attempts = 4) ---")
    injector.reset(fail_first_n=5)
    t0 = time.perf_counter()
    try:
        await app.ainvoke({"order_id": "ORD-0001", "result": None, "steps_executed": []})
        print("ERROR: Expected TransientError to be raised, but succeeded unexpectedly!")
    except TransientError as e:
        dur = time.perf_counter() - t0
        print(f"\nCaught Expected Exhaustion Error: {type(e).__name__}: {e}")
        print(f"Total Attempts Recorded: {injector.call_count} of max {RETRY_POLICY.max_attempts}")
        print(f"Total Duration Until Failure: {dur:.3f}s")
        print("Retries exhausted cleanly without unhandled crash.")


async def run_task16b_node_timeout():
    print("=" * 80)
    print("TASK 16b: PER-NODE TIMEOUT VERIFICATION")
    print("=" * 80)
    print("Configuration: Node configured with @with_timeout(1.0s), executing 5.0s sleep")
    print("-" * 80)

    app = build_node_timeout_graph()
    t0 = time.perf_counter()
    try:
        await app.ainvoke({"order_id": "ORD-0001", "result": None, "steps_executed": []})
        print("ERROR: Expected NodeTimeoutError, but step completed!")
    except NodeTimeoutError as e:
        dur = time.perf_counter() - t0
        print(f"\nCaught Expected NodeTimeoutError: {e}")
        print(f"Elapsed Time: {dur:.3f}s (Verified ~1.0s cancellation, NOT hanging for 5.0s)")
        assert 0.9 <= dur <= 1.5, f"Unexpected duration: {dur}s"
        print("Per-node timeout cancelled coroutine cleanly.")


async def run_task16c_global_timeout():
    print("=" * 80)
    print("TASK 16c: GLOBAL PIPELINE TIMEOUT VERIFICATION")
    print("=" * 80)
    print("Configuration:")
    print("  - 4 Sequential nodes, each taking ~0.8s (Total work = ~3.2s)")
    print("  - Each node timeout = 1.0s (No single node breaches its local limit)")
    print("  - Global pipeline timeout = 2.0s via asyncio.wait_for")
    print("-" * 80)

    app = build_global_timeout_graph()
    t0 = time.perf_counter()
    try:
        await asyncio.wait_for(
            app.ainvoke({"order_id": "ORD-0001", "result": None, "steps_executed": []}),
            timeout=2.0
        )
        print("ERROR: Expected asyncio.TimeoutError, but workflow completed!")
    except asyncio.TimeoutError:
        dur = time.perf_counter() - t0
        print(f"\nCaught Expected Global asyncio.TimeoutError after {dur:.3f}s")
        print("Global pipeline aborted at ~2.0s: later nodes never started.")
        assert 1.9 <= dur <= 2.5, f"Unexpected duration: {dur}s"
        print("Global timeout cancellation verified successfully.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("retry", "node_timeout", "global_timeout"):
        print("Usage: python demo_resilience.py [retry|node_timeout|global_timeout]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "retry":
        asyncio.run(run_task16a_retry())
    elif cmd == "node_timeout":
        asyncio.run(run_task16b_node_timeout())
    elif cmd == "global_timeout":
        asyncio.run(run_task16c_global_timeout())
