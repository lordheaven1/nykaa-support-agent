"""
dataset.py
Deterministic order dataset generator for Nykaa customer support agent.
Track: E-commerce & Retail (Nykaa)
"""

import random
from typing import Any, Dict, List

# ---------------------------------------------------------
# CONSTANTS & DESIGN CHOICES
# ---------------------------------------------------------
SEED = 42
N_RECORDS = 60

CATEGORIES = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]
CATEGORY_WEIGHTS = [0.25, 0.15, 0.15, 0.15, 0.30]

STATUSES = ["Placed", "Shipped", "Delivered", "Returned", "Refunded"]
STATUS_WEIGHTS = [0.15, 0.25, 0.40, 0.10, 0.10]

# Category price ranges in INR:
# Reasoning: Ranges reflect authentic Nykaa basket sizes, where cosmetic items
# and home decor are lower-to-mid ticket while personal styling appliances and
# tech gadgets command higher ticket values.
PRICE_RANGES = {
    "Beauty": (200, 3500),
    "Apparel": (700, 6000),
    "Footwear": (900, 5000),
    "Electronics": (1500, 12000),
    "Home": (500, 4000),
}


def generate_orders(
    n_records: int = N_RECORDS, seed: int = SEED
) -> List[Dict[str, Any]]:
    """
    Generates a deterministic list of orders using an isolated random generator.
    Each record draws attributes in a fixed sequence:
    1. category
    2. status
    3. order_value_inr
    4. days_since_created
    5. delayed_shipment roll
    """
    rng = random.Random(seed)
    orders: List[Dict[str, Any]] = []

    for i in range(1, n_records + 1):
        # Non-numeric ID to avoid collisions with phone/card masking
        record_id = f"ORD-{i:04d}"

        # 1. Category draw
        category = rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0]

        # 2. Status draw
        status = rng.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0]

        # 3. Order value draw (rounded to nearest 10 INR)
        min_p, max_p = PRICE_RANGES[category]
        order_value_inr = round(rng.randint(min_p, max_p) / 10.0) * 10

        # 4. Age draw (uniform integer 0-30)
        days_since_created = rng.randint(0, 30)

        # 5. Delayed shipment determination:
        # Rule: Missed promised dispatch or delivery SLA date.
        # Applies only to orders not yet returned or refunded.
        if status in ["Returned", "Refunded"]:
            delayed_shipment = False
        elif status in ["Placed", "Shipped"]:
            if days_since_created >= 5:
                delayed_shipment = rng.random() < 0.45
            else:
                delayed_shipment = rng.random() < 0.05
        elif status == "Delivered":
            if days_since_created >= 8:
                delayed_shipment = rng.random() < 0.20
            else:
                delayed_shipment = False
        else:
            delayed_shipment = False

        orders.append(
            {
                "record_id": record_id,
                "category": category,
                "status": status,
                "order_value_inr": order_value_inr,
                "days_since_created": days_since_created,
                "delayed_shipment": delayed_shipment,
            }
        )

    return orders


def validate_orders(orders: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validates required rubric constraints and returns summary metrics."""
    total = len(orders)
    category_counts: Dict[str, int] = {c: 0 for c in CATEGORIES}
    status_counts: Dict[str, int] = {s: 0 for s in STATUSES}
    delayed_count = 0
    ids_seen = set()

    for o in orders:
        category_counts[o["category"]] += 1
        status_counts[o["status"]] += 1
        if o["delayed_shipment"]:
            delayed_count += 1

        # Check bounds & uniqueness
        assert (
            0 <= o["days_since_created"] <= 30
        ), f"Invalid age: {o['days_since_created']}"
        assert o["record_id"] not in ids_seen, f"Duplicate ID: {o['record_id']}"
        ids_seen.add(o["record_id"])

    delayed_pct = (delayed_count / total) * 100.0

    # Rubric Assertions
    assert total >= 40, f"Expected >= 40 records, found {total}"
    for cat, count in category_counts.items():
        assert (
            count >= 3
        ), f"Category '{cat}' has {count} records (expected >= 3)"
    for st, count in status_counts.items():
        assert count >= 1, f"Status '{st}' has {count} records (expected >= 1)"
    assert (
        10.0 <= delayed_pct <= 30.0
    ), f"Delayed % {delayed_pct:.2f}% outside [10.0%, 30.0%]"

    return {
        "total": total,
        "category_counts": category_counts,
        "status_counts": status_counts,
        "delayed_count": delayed_count,
        "delayed_pct": delayed_pct,
    }


# Module-level constant exported for imports in later tasks
ORDERS = generate_orders()


if __name__ == "__main__":
    print("=" * 65)
    print("NYKAA DOMAIN DATASET VALIDATION (TASK 1)")
    print("=" * 65)
    stats = validate_orders(ORDERS)

    print(f"Total Records: {stats['total']} [PASS >= 40]")
    print(f"Random Seed:   {SEED}\n")

    print("Category Counts (Rule: >= 3 each):")
    for cat, cnt in stats["category_counts"].items():
        print(f"  - {cat:<14}: {cnt:<3} [PASS]")

    print("\nStatus Counts (Rule: >= 1 each):")
    for st, cnt in stats["status_counts"].items():
        print(f"  - {st:<14}: {cnt:<3} [PASS]")

    print(
        f"\nDelayed Shipment Rate: {stats['delayed_pct']:.2f}% ({stats['delayed_count']}/{stats['total']})"
    )
    print("Target Band: [10.0%, 30.0%] -> PASS")
    print("All Record IDs Unique: True [PASS]")
    print("Days Range [0, 30] Valid: True [PASS]")
    print("=" * 65)