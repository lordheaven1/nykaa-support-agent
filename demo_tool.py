"""
demo_tool.py
Demonstrates Task 6 tool execution across delayed open, non-delayed open,
resolved, whitespace-padded, and nonexistent order IDs.
"""

from tools import check_order_status, STATS, ESCALATION_THRESHOLD

TEST_ORDER_IDS = [
    "ORD-0008",         # Shipped, delayed, high recency
    "ORD-0010",         # Shipped, not delayed
    "ORD-0001",         # Delivered (resolved)
    "  ord-0008  ",     # Lowercase with padding
    "ORD-9999",         # Nonexistent ID
]


def main():
    print("=" * 70)
    print("TASK 6: ORDER STATUS & ESCALATION SCORING DEMO")
    print("=" * 70)
    print(f"Dataset Distribution (60 orders):")
    print(f"  P50: {STATS['p50']:.4f} | P75: {STATS['p75']:.4f} | P80: {STATS['p80']:.4f} | P90: {STATS['p90']:.4f}")
    print(f"Empirical Escalation Threshold: {ESCALATION_THRESHOLD:.4f} (80th percentile)")
    print(f"Orders exceeding threshold: {STATS['exceed_count']} of {STATS['total_orders']}")
    print("=" * 70)

    for idx, test_id in enumerate(TEST_ORDER_IDS, 1):
        res = check_order_status(test_id)
        print(f"\n[Test {idx}]: Query ID = \"{test_id}\"")
        if "error" in res:
            print(f"  - Error:     {res['error']}")
            print(f"  - Message:   {res['message']}")
        else:
            print(f"  - Order ID:  {res['record_id']}")
            print(f"  - Status:    {res['status']}")
            print(f"  - Value:     Rs {res['order_value_inr']}")
            print(f"  - Delayed:   {res['delayed_shipment']} ({res['days_since_created']} days old)")
            print(f"  - Score:     {res['escalation_score']:.4f}")
            print(f"  - Escalate:  {res['escalate']}")
            print(f"  - Reason:    {res['reason']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
